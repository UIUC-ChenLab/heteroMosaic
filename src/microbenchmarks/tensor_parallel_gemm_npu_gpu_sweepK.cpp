#include "hipkernels/w4a16_gemm_packed.hpp"
#include "third_party/nlohmann/json.hpp"
#include "unified_llm_w4a16_hetero/npuSetup.hpp"
#include "unified_llm_w4a16_hetero/unified_llm_w4a16.hpp"

#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

constexpr int ITER = 512;
constexpr int WARMUP_CYCLES = 8;
constexpr int GPU_THREAD_CORE = 2;
constexpr int NPU_THREAD_CORE = 3;
constexpr float CHECK_RTOL = 0.20f;
constexpr float CHECK_ATOL = 0.20f;
constexpr int GROUP_SIZE = 128;
constexpr int HW_NUM_TILES = 32;
constexpr int ACTIVE_HW_SLOT = 0;
constexpr int ACTIVE_INST_SLOT = 0;

struct SweepCase {
    int64_t M;
    int64_t K;
    int64_t N;
    bool enabled;
    std::string artifact_subdir;
};

struct SweepPoint {
    int64_t npu_k;
    fs::path pdi_path;
    fs::path inst_path;
};

struct BenchmarkStats {
    double avg_latency_us = 0.0;
    double min_latency_us = 0.0;
    double max_latency_us = 0.0;
    double throughput_gops = 0.0;
    std::vector<int64_t> latencies_us;
};

struct RuntimeCaseState {
    torch::Tensor input;
    torch::Tensor split_output;
    torch::Tensor reference_output;
    torch::Tensor qweight;
    torch::Tensor scales;
    torch::Tensor zeros;
    torch::Tensor packed_params;
};

struct RuntimePointState {
    int64_t gpu_k = 0;
    int64_t npu_k = 0;
    torch::Tensor gpu_input;
    torch::Tensor npu_input;
    torch::Tensor npu_output;
    torch::Tensor gpu_packed_params;
    torch::Tensor npu_packed_params;
    uint32_t npu_input_handle = 0;
    uint32_t npu_output_handle = 0;
    uint32_t npu_weight_handle = 0;
};

std::string now_utc_iso8601() {
    const std::time_t now = std::time(nullptr);
    std::tm tm_now {};
    gmtime_r(&now, &tm_now);

    std::ostringstream oss;
    oss << std::put_time(&tm_now, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

void set_thread_affinity(std::thread &thread, int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    const int rc = pthread_setaffinity_np(thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "Warning: failed to set thread affinity to core " << core_id << ": " << std::strerror(rc) << std::endl;
    }
}

std::string make_result_key(const SweepCase &sweep_case, const fs::path &artifact_dir, int64_t npu_k) {
    std::ostringstream oss;
    oss << sweep_case.M << "x" << sweep_case.K << "x" << sweep_case.N << "__" << artifact_dir.filename().string() << "__npuK_" << npu_k;
    return oss.str();
}

fs::path get_executable_dir(const char *argv0) {
    std::error_code ec;
    fs::path proc_exe = fs::read_symlink("/proc/self/exe", ec);
    if (!ec && !proc_exe.empty()) {
        return proc_exe.parent_path();
    }

    fs::path arg_path = argv0 ? fs::path(argv0) : fs::current_path();
    if (arg_path.is_relative()) {
        arg_path = fs::absolute(arg_path, ec);
    }
    if (ec || arg_path.empty()) {
        return fs::current_path();
    }
    return arg_path.parent_path();
}

std::string get_root_dir(const char *argv0) {
    const char *env_root = std::getenv("HETEROMOSAIC_ROOT");
    if (env_root) {
        return std::string(env_root);
    }

    const fs::path exec_dir = get_executable_dir(argv0);
    if (exec_dir.filename() == "bin" && exec_dir.parent_path().filename() == "build") {
        return exec_dir.parent_path().parent_path().string();
    }
    if (fs::exists(exec_dir / "py/heteroMosaic_figs/results")) {
        return exec_dir.string();
    }
    if (fs::exists(exec_dir.parent_path() / "py/heteroMosaic_figs/results")) {
        return exec_dir.parent_path().string();
    }

    return "/home/greg/Desktop/heteroMosaic";
}

fs::path get_default_results_path(const std::string &root_dir) {
    return fs::path(root_dir) / "py/heteroMosaic_figs/results/tensor_parallel_gemm_npu_gpu_sweepK_results.json";
}

json load_results_document(const fs::path &results_path) {
    if (!fs::exists(results_path)) {
        return json {
            {"schema_version", 1},
            {"results", json::object()},
        };
    }

    std::ifstream input(results_path);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open results JSON: " + results_path.string());
    }

    json doc = json::parse(input);
    if (!doc.contains("results") || !doc["results"].is_object()) {
        doc["results"] = json::object();
    }
    if (!doc.contains("schema_version")) {
        doc["schema_version"] = 1;
    }
    return doc;
}

void write_results_document(const fs::path &results_path, json &doc) {
    doc["updated_at_utc"] = now_utc_iso8601();

    const fs::path parent = results_path.parent_path();
    if (!parent.empty()) {
        fs::create_directories(parent);
    }

    const fs::path tmp_path = results_path.string() + ".tmp";
    {
        std::ofstream output(tmp_path);
        if (!output.is_open()) {
            throw std::runtime_error("Failed to open temp results file: " + tmp_path.string());
        }
        output << doc.dump(2) << '\n';
    }

    fs::rename(tmp_path, results_path);
}

bool parse_pdi_filename(const std::string &name, int64_t &m, int64_t &k, int64_t &n) {
    long long parsed_m = 0;
    long long parsed_k = 0;
    long long parsed_n = 0;
    const int matched =
        std::sscanf(name.c_str(), "final_%lldx%lldx%lld_64x128x64_8c_bf16_int4AWQ_bf16.pdi", &parsed_m, &parsed_k, &parsed_n);
    if (matched != 3) {
        return false;
    }
    m = parsed_m;
    k = parsed_k;
    n = parsed_n;
    return true;
}

bool parse_inst_filename(const std::string &name, int64_t &m, int64_t &k, int64_t &n) {
    long long parsed_m = 0;
    long long parsed_k = 0;
    long long parsed_n = 0;
    const int matched =
        std::sscanf(name.c_str(), "insts_%lldx%lldx%lld_64x128x64_8c_bf16_int4AWQ_bf16.txt", &parsed_m, &parsed_k, &parsed_n);
    if (matched != 3) {
        return false;
    }
    m = parsed_m;
    k = parsed_k;
    n = parsed_n;
    return true;
}

std::vector<SweepPoint> discover_sweep_points(const SweepCase &sweep_case, const fs::path &artifact_dir) {
    if (!fs::exists(artifact_dir)) {
        throw std::runtime_error("Artifact directory does not exist: " + artifact_dir.string());
    }

    std::map<int64_t, fs::path> pdi_by_npu_k;
    std::map<int64_t, fs::path> inst_by_npu_k;

    for (const auto &entry : fs::directory_iterator(artifact_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        const std::string filename = entry.path().filename().string();
        int64_t file_m = 0;
        int64_t file_k = 0;
        int64_t file_n = 0;
        if (parse_pdi_filename(filename, file_m, file_k, file_n)) {
            if (file_m == sweep_case.M && file_n == sweep_case.N && file_k > 0 && file_k <= sweep_case.K) {
                pdi_by_npu_k[file_k] = entry.path();
            }
        } else if (parse_inst_filename(filename, file_m, file_k, file_n)) {
            if (file_m == sweep_case.M && file_n == sweep_case.N && file_k > 0 && file_k <= sweep_case.K) {
                inst_by_npu_k[file_k] = entry.path();
            }
        }
    }

    std::vector<SweepPoint> points;
    for (const auto &[npu_k, pdi_path] : pdi_by_npu_k) {
        const auto inst_it = inst_by_npu_k.find(npu_k);
        if (inst_it == inst_by_npu_k.end()) {
            continue;
        }
        points.push_back(SweepPoint {npu_k, pdi_path, inst_it->second});
    }

    std::sort(points.begin(), points.end(), [](const SweepPoint &lhs, const SweepPoint &rhs) { return lhs.npu_k < rhs.npu_k; });
    return points;
}

void close_gem_handle_if_needed(int fd, uint32_t handle) {
    if (fd < 0 || handle == 0) {
        return;
    }

    drm_gem_close close_args {};
    close_args.handle = handle;
    if (ioctl(fd, DRM_IOCTL_GEM_CLOSE, &close_args) != 0) {
        std::cerr << "Warning: failed to close GEM handle " << handle << ": " << std::strerror(errno) << std::endl;
    }
}

void destroy_hw_context_slot(int fd, hwctxt &ctx) {
    if (fd >= 0 && ctx.hw_ctx.handle != 0) {
        amdxdna_drm_destroy_ctx destroy_ctx {};
        destroy_ctx.handle = ctx.hw_ctx.handle;
        if (ioctl(fd, DRM_IOCTL_AMDXDNA_DESTROY_CTX, &destroy_ctx) != 0) {
            std::cerr << "Warning: failed to destroy HW context " << ctx.hw_ctx.handle << ": " << std::strerror(errno) << std::endl;
        }
    }

    close_gem_handle_if_needed(fd, ctx.pdi_handle);
    ctx = {};
}

void destroy_inst_context_slot(int fd, instctxt &ctx) {
    close_gem_handle_if_needed(fd, ctx.dpu_0_handle);
    ctx = {};
}

void load_active_context(const SweepPoint &point) {
    destroy_hw_context_slot(xdna_drv_fd, hwctxt_array[ACTIVE_HW_SLOT]);
    destroy_inst_context_slot(xdna_drv_fd, instctxt_array[ACTIVE_INST_SLOT]);

    if (createHWctxt(xdna_drv_fd, hwctxt_array[ACTIVE_HW_SLOT], point.pdi_path.c_str(), HW_NUM_TILES) != 0) {
        throw std::runtime_error("Failed to create HW context for " + point.pdi_path.string());
    }

    if (createInstctxt(xdna_drv_fd, instctxt_array[ACTIVE_INST_SLOT], point.inst_path.c_str(), true) != 0) {
        throw std::runtime_error("Failed to create instruction context for " + point.inst_path.string());
    }

    FlushCpuCache((const void *)instctxt_array[ACTIVE_INST_SLOT].dpu_0_vaddr, 0,
                  instctxt_array[ACTIVE_INST_SLOT].num_dpu_0_insts * sizeof(uint32_t));
    FlushCpuCache((const void *)hwctxt_array[ACTIVE_HW_SLOT].pdi_vaddr, 0, hwctxt_array[ACTIVE_HW_SLOT].pdi_size);
}

torch::Tensor build_input_tensor(int64_t M, int64_t K) {
    auto options = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    return (torch::rand({M, K}, options) * 0.1f).contiguous();
}

torch::Tensor build_qweight_tensor(int64_t K, int64_t N) {
    return torch::randint(0, 16, {K, N}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA)).contiguous();
}

torch::Tensor build_scales_tensor(int64_t K, int64_t N) {
    const int64_t num_groups = K / GROUP_SIZE;
    auto scales = torch::rand({num_groups, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    return (scales.to(torch::kBFloat16) * 0.1f).contiguous();
}

torch::Tensor build_zeros_tensor(int64_t K, int64_t N) {
    const int64_t num_groups = K / GROUP_SIZE;
    return torch::randint(0, 16, {num_groups, N}, torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA)).contiguous();
}

torch::Tensor build_packed_weights_from_components(const torch::Tensor &qweight, const torch::Tensor &scales, const torch::Tensor &zeros,
                                                   int64_t K, int64_t N) {
    QuantizedLinearImpl layer(K, N, false);
    layer.to(torch::Device(torch::kCUDA));

    layer.set_quantized_weights(qweight, scales, zeros, torch::Tensor());
    auto packed_params = layer.get_packed_params();
    if (!packed_params.defined()) {
        throw std::runtime_error("Packed params are undefined after set_quantized_weights()");
    }

    return packed_params.contiguous();
}

torch::Tensor build_packed_weights(int64_t K, int64_t N) {
    auto qweight = build_qweight_tensor(K, N);
    auto scales = build_scales_tensor(K, N);
    auto zeros = build_zeros_tensor(K, N);
    return build_packed_weights_from_components(qweight, scales, zeros, K, N);
}

torch::Tensor build_full_gpu_reference(const torch::Tensor &input, const torch::Tensor &packed_params, int64_t M, int64_t K, int64_t N) {
    auto reference = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
    hipkernels::w4a16_gemm_fused_packed(reference, input, packed_params, K, N);
    torch::cuda::synchronize();
    return reference;
}

std::chrono::microseconds run_gpu_k(torch::Tensor &output, const RuntimePointState &point_state, int64_t N) {
    if (point_state.gpu_k <= 0) {
        return std::chrono::microseconds(0);
    }

    const auto start = std::chrono::high_resolution_clock::now();
    hipkernels::w4a16_gemm_fused_packed(output, point_state.gpu_input, point_state.gpu_packed_params, point_state.gpu_k, N);
    torch::cuda::synchronize();
    const auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start);
}

std::chrono::microseconds run_npu_k(RuntimePointState &point_state) {
    if (point_state.npu_k <= 0) {
        return std::chrono::microseconds(0);
    }

    void *output_ptr = point_state.npu_output.data_ptr();
    void *input_ptr = point_state.npu_input.data_ptr();
    void *weight_ptr = point_state.npu_packed_params.data_ptr();

    const auto start = std::chrono::high_resolution_clock::now();
    const int ret = npuMatmul_zero(ACTIVE_HW_SLOT, ACTIVE_INST_SLOT, output_ptr, input_ptr, weight_ptr, point_state.npu_output_handle,
                                   point_state.npu_input_handle, point_state.npu_weight_handle, (hipEvent_t) nullptr);
    const auto end = std::chrono::high_resolution_clock::now();

    if (ret != 0) {
        throw std::runtime_error("npuMatmul_zero failed");
    }

    return std::chrono::duration_cast<std::chrono::microseconds>(end - start);
}

void run_split_once(torch::Tensor &output, RuntimePointState &point_state, int64_t N) {
    std::chrono::microseconds gpu_time(0);
    std::chrono::microseconds npu_time(0);

    std::optional<std::thread> gpu_thread;
    if (point_state.gpu_k > 0) {
        gpu_thread.emplace([&]() { gpu_time = run_gpu_k(output, point_state, N); });
        set_thread_affinity(*gpu_thread, GPU_THREAD_CORE);
    }

    std::optional<std::thread> npu_thread;
    if (point_state.npu_k > 0) {
        npu_thread.emplace([&]() { npu_time = run_npu_k(point_state); });
        set_thread_affinity(*npu_thread, NPU_THREAD_CORE);
    }

    if (gpu_thread.has_value()) {
        gpu_thread->join();
    }
    if (npu_thread.has_value()) {
        npu_thread->join();
    }

    if (point_state.npu_k > 0) {
        output.add_(point_state.npu_output);
        torch::cuda::synchronize();
    }
}

BenchmarkStats benchmark_split(torch::Tensor &output, RuntimePointState &point_state, int64_t M, int64_t K, int64_t N) {
    BenchmarkStats stats;
    stats.min_latency_us = std::numeric_limits<double>::max();
    stats.max_latency_us = 0.0;

    for (int iter = 0; iter < ITER; ++iter) {
        output.zero_();
        if (point_state.npu_k > 0) {
            point_state.npu_output.zero_();
        }
        torch::cuda::synchronize();

        const auto iter_start = std::chrono::high_resolution_clock::now();
        run_split_once(output, point_state, N);
        const auto iter_end = std::chrono::high_resolution_clock::now();
        const auto iter_latency =
            std::chrono::duration_cast<std::chrono::microseconds>(iter_end - iter_start).count();

        if (iter >= WARMUP_CYCLES) {
            stats.latencies_us.push_back(iter_latency);
            stats.min_latency_us = std::min(stats.min_latency_us, static_cast<double>(iter_latency));
            stats.max_latency_us = std::max(stats.max_latency_us, static_cast<double>(iter_latency));
        }
    }

    if (stats.latencies_us.empty()) {
        throw std::runtime_error("No benchmark latencies recorded");
    }

    double total_latency = 0.0;
    for (const int64_t latency : stats.latencies_us) {
        total_latency += static_cast<double>(latency);
    }

    stats.avg_latency_us = total_latency / static_cast<double>(stats.latencies_us.size());
    const double ops = 2.0 * static_cast<double>(M) * static_cast<double>(K) * static_cast<double>(N);
    stats.throughput_gops = ops / (stats.avg_latency_us * 1000.0);
    return stats;
}

RuntimePointState prepare_gpu_baseline_point_state(const RuntimeCaseState &case_state, int64_t K) {
    RuntimePointState point_state;
    point_state.gpu_k = K;
    point_state.npu_k = 0;
    point_state.gpu_input = case_state.input;
    point_state.gpu_packed_params = case_state.packed_params;
    return point_state;
}

RuntimePointState prepare_point_state(const RuntimeCaseState &case_state, const SweepCase &sweep_case, int64_t npu_k) {
    if (npu_k <= 0 || npu_k > sweep_case.K) {
        throw std::runtime_error("Invalid npuK " + std::to_string(npu_k));
    }
    if (npu_k % GROUP_SIZE != 0) {
        throw std::runtime_error("npuK must be a multiple of group size: " + std::to_string(npu_k));
    }

    RuntimePointState point_state;
    point_state.npu_k = npu_k;
    point_state.gpu_k = sweep_case.K - npu_k;

    if (point_state.gpu_k > 0) {
        if (point_state.gpu_k % GROUP_SIZE != 0) {
            throw std::runtime_error("gpuK must be a multiple of group size: " + std::to_string(point_state.gpu_k));
        }
        const int64_t gpu_groups = point_state.gpu_k / GROUP_SIZE;
        point_state.gpu_input = case_state.input.slice(1, 0, point_state.gpu_k).contiguous();
        point_state.gpu_packed_params =
            build_packed_weights_from_components(case_state.qweight.slice(0, 0, point_state.gpu_k).contiguous(),
                                                 case_state.scales.slice(0, 0, gpu_groups).contiguous(),
                                                 case_state.zeros.slice(0, 0, gpu_groups).contiguous(), point_state.gpu_k, sweep_case.N);
    }

    const int64_t npu_group_start = point_state.gpu_k / GROUP_SIZE;
    const int64_t npu_group_end = sweep_case.K / GROUP_SIZE;
    point_state.npu_input = case_state.input.slice(1, point_state.gpu_k, sweep_case.K).contiguous();
    point_state.npu_output =
        torch::zeros({sweep_case.M, sweep_case.N}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
    point_state.npu_packed_params =
        build_packed_weights_from_components(case_state.qweight.slice(0, point_state.gpu_k, sweep_case.K).contiguous(),
                                             case_state.scales.slice(0, npu_group_start, npu_group_end).contiguous(),
                                             case_state.zeros.slice(0, npu_group_start, npu_group_end).contiguous(), point_state.npu_k,
                                             sweep_case.N);
    torch::cuda::synchronize();

    point_state.npu_input_handle =
        import_dma_buf_to_xdna(point_state.npu_input.data_ptr(), point_state.npu_input.numel(), point_state.npu_input.element_size());
    point_state.npu_output_handle =
        import_dma_buf_to_xdna(point_state.npu_output.data_ptr(), point_state.npu_output.numel(), point_state.npu_output.element_size());
    point_state.npu_weight_handle = import_dma_buf_to_xdna(point_state.npu_packed_params.data_ptr(), point_state.npu_packed_params.numel(),
                                                           point_state.npu_packed_params.element_size());

    if (point_state.npu_input_handle == 0 || point_state.npu_output_handle == 0 || point_state.npu_weight_handle == 0) {
        throw std::runtime_error("Failed to import one or more K-split DMA-BUF handles to XDNA");
    }

    return point_state;
}

void ensure_gpu_baseline_result(json &results_doc, const fs::path &results_path, const SweepCase &sweep_case, const fs::path &artifact_dir,
                                RuntimeCaseState &case_state, bool force_rerun) {
    const std::string result_key = make_result_key(sweep_case, artifact_dir, 0);
    const bool already_completed =
        results_doc["results"].contains(result_key) && results_doc["results"][result_key].value("status", "") == "completed";
    if (already_completed && !force_rerun) {
        std::cout << "Skipping completed iGPU baseline for " << sweep_case.M << "x" << sweep_case.K << "x" << sweep_case.N << std::endl;
        return;
    }

    results_doc["results"][result_key] = {
        {"M", sweep_case.M},
        {"K", sweep_case.K},
        {"N", sweep_case.N},
        {"artifact_dir", artifact_dir.string()},
        {"npu_k", 0},
        {"gpu_k", sweep_case.K},
        {"pdi_path", ""},
        {"inst_path", ""},
        {"status", "in_progress"},
        {"updated_at_utc", now_utc_iso8601()},
    };
    write_results_document(results_path, results_doc);

    try {
        RuntimePointState baseline_point_state = prepare_gpu_baseline_point_state(case_state, sweep_case.K);
        case_state.split_output.zero_();
        torch::cuda::synchronize();
        run_split_once(case_state.split_output, baseline_point_state, sweep_case.N);

        const auto diff = (case_state.reference_output - case_state.split_output).abs();
        const double max_abs_diff = diff.max().item<double>();
        const bool allclose = torch::allclose(case_state.reference_output, case_state.split_output, CHECK_RTOL, CHECK_ATOL);

        BenchmarkStats stats = benchmark_split(case_state.split_output, baseline_point_state, sweep_case.M, sweep_case.K, sweep_case.N);

        results_doc["results"][result_key] = {
            {"M", sweep_case.M},
            {"K", sweep_case.K},
            {"N", sweep_case.N},
            {"artifact_dir", artifact_dir.string()},
            {"npu_k", 0},
            {"gpu_k", sweep_case.K},
            {"pdi_path", ""},
            {"inst_path", ""},
            {"status", "completed"},
            {"avg_latency_us", stats.avg_latency_us},
            {"min_latency_us", stats.min_latency_us},
            {"max_latency_us", stats.max_latency_us},
            {"throughput_gops", stats.throughput_gops},
            {"allclose", allclose},
            {"max_abs_diff", max_abs_diff},
            {"latencies_us", stats.latencies_us},
            {"updated_at_utc", now_utc_iso8601()},
        };
        write_results_document(results_path, results_doc);

        std::cout << "Completed iGPU baseline avg_latency_us=" << stats.avg_latency_us << " throughput_gops=" << stats.throughput_gops
                  << " allclose=" << (allclose ? "true" : "false") << " max_abs_diff=" << max_abs_diff << std::endl;
    } catch (const std::exception &baseline_error) {
        results_doc["results"][result_key] = {
            {"M", sweep_case.M},
            {"K", sweep_case.K},
            {"N", sweep_case.N},
            {"artifact_dir", artifact_dir.string()},
            {"npu_k", 0},
            {"gpu_k", sweep_case.K},
            {"pdi_path", ""},
            {"inst_path", ""},
            {"status", "failed"},
            {"error", baseline_error.what()},
            {"updated_at_utc", now_utc_iso8601()},
        };
        write_results_document(results_path, results_doc);
        throw;
    }
}

json make_metadata(const std::string &root_dir, const fs::path &results_path, const std::vector<SweepCase> &cases) {
    json enabled_cases = json::array();
    for (const auto &sweep_case : cases) {
        enabled_cases.push_back({
            {"M", sweep_case.M},
            {"K", sweep_case.K},
            {"N", sweep_case.N},
            {"enabled", sweep_case.enabled},
            {"artifact_subdir", sweep_case.artifact_subdir},
        });
    }

    return json {
        {"schema_version", 1},
        {"created_at_utc", now_utc_iso8601()},
        {"updated_at_utc", now_utc_iso8601()},
        {"root_dir", root_dir},
        {"results_path", results_path.string()},
        {"benchmark", {
            {"iter", ITER},
            {"warmup_cycles", WARMUP_CYCLES},
            {"num_matrices", 1},
            {"reference", "full_gpu_w4a16_packed"},
            {"check_rtol", CHECK_RTOL},
            {"check_atol", CHECK_ATOL},
        }},
        {"cases", enabled_cases},
        {"results", json::object()},
    };
}

void ensure_document_metadata(json &doc, const std::string &root_dir, const fs::path &results_path, const std::vector<SweepCase> &cases) {
    if (!doc.contains("benchmark")) {
        doc = make_metadata(root_dir, results_path, cases);
        return;
    }

    doc["schema_version"] = 1;
    doc["root_dir"] = root_dir;
    doc["results_path"] = results_path.string();
    doc["benchmark"] = {
        {"iter", ITER},
        {"warmup_cycles", WARMUP_CYCLES},
        {"num_matrices", 1},
        {"reference", "full_gpu_w4a16_packed"},
        {"check_rtol", CHECK_RTOL},
        {"check_atol", CHECK_ATOL},
    };

    json enabled_cases = json::array();
    for (const auto &sweep_case : cases) {
        enabled_cases.push_back({
            {"M", sweep_case.M},
            {"K", sweep_case.K},
            {"N", sweep_case.N},
            {"enabled", sweep_case.enabled},
            {"artifact_subdir", sweep_case.artifact_subdir},
        });
    }
    doc["cases"] = enabled_cases;
    if (!doc.contains("results") || !doc["results"].is_object()) {
        doc["results"] = json::object();
    }
}

RuntimeCaseState prepare_case_state(const SweepCase &sweep_case) {
    torch::manual_seed(42);
    RuntimeCaseState state;
    state.input = build_input_tensor(sweep_case.M, sweep_case.K);
    state.qweight = build_qweight_tensor(sweep_case.K, sweep_case.N);
    state.scales = build_scales_tensor(sweep_case.K, sweep_case.N);
    state.zeros = build_zeros_tensor(sweep_case.K, sweep_case.N);
    state.packed_params = build_packed_weights_from_components(state.qweight, state.scales, state.zeros, sweep_case.K, sweep_case.N);
    torch::cuda::synchronize();

    state.reference_output = build_full_gpu_reference(state.input, state.packed_params, sweep_case.M, sweep_case.K, sweep_case.N);
    state.split_output =
        torch::zeros({sweep_case.M, sweep_case.N}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
    torch::cuda::synchronize();

    return state;
}

} // namespace

int main(int argc, char **argv) {
    try {
        setbuf(stdout, nullptr);

        std::string root_dir = get_root_dir(argv[0]);
        fs::path results_path = get_default_results_path(root_dir);
        bool force_rerun = false;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg.rfind("--results_json=", 0) == 0) {
                results_path = arg.substr(std::string("--results_json=").size());
            } else if (arg.rfind("--force_rerun=", 0) == 0) {
                force_rerun = std::stoi(arg.substr(std::string("--force_rerun=").size())) != 0;
            } else {
                std::cerr << "Unknown argument: " << arg << std::endl;
                return 1;
            }
        }

        std::vector<SweepCase> sweep_cases = {
            {2048, 2048, 2048, true, "hw_bins/npu2/8192x2048x2048/bf16_int4AWQ_bf16_K"},
            {4096, 4096, 4096, true, "hw_bins/npu2/8192x4096x4096/bf16_int4AWQ_bf16_K"},
            {8192, 8192, 8192, true, "hw_bins/npu2/8192x8192x8192/bf16_int4AWQ_bf16_K"},
            {8192, 4096, 14336, true, "hw_bins/npu2/8192x4096x14336/bf16_int4AWQ_bf16_K"},
            {8192, 14336, 4096, true, "hw_bins/npu2/8192x14336x4096/bf16_int4AWQ_bf16_K"},
        };

        debug_verbosity = 0;
        set_npu_debug_verbosity(0);
        use_packed_weights = true;
        pad_packed_weights = false;

        if (!torch::cuda::is_available()) {
            std::cerr << "HIP/CUDA is not available." << std::endl;
            return 1;
        }

        if (initialize_xdna_driver() != 0) {
            std::cerr << "Failed to initialize XDNA driver." << std::endl;
            return 1;
        }
        init_npu();

        json results_doc = load_results_document(results_path);
        ensure_document_metadata(results_doc, root_dir, results_path, sweep_cases);
        write_results_document(results_path, results_doc);

        for (const auto &sweep_case : sweep_cases) {
            if (!sweep_case.enabled) {
                continue;
            }

            const fs::path artifact_dir = fs::path(root_dir) / sweep_case.artifact_subdir;
            std::cout << "\n=== Case " << sweep_case.M << "x" << sweep_case.K << "x" << sweep_case.N << " ===" << std::endl;
            std::cout << "Artifact dir: " << artifact_dir << std::endl;

            const auto sweep_points = discover_sweep_points(sweep_case, artifact_dir);
            if (sweep_points.empty()) {
                std::cerr << "No valid K-split sweep points found for " << artifact_dir << std::endl;
                continue;
            }

            std::vector<SweepPoint> pending_points;
            pending_points.reserve(sweep_points.size());
            for (const auto &point : sweep_points) {
                const std::string result_key = make_result_key(sweep_case, artifact_dir, point.npu_k);
                const bool already_completed =
                    results_doc["results"].contains(result_key) &&
                    results_doc["results"][result_key].value("status", "") == "completed";
                if (already_completed && !force_rerun) {
                    std::cout << "Skipping completed point npuK=" << point.npu_k << std::endl;
                    continue;
                }
                pending_points.push_back(point);
            }

            RuntimeCaseState case_state = prepare_case_state(sweep_case);
            ensure_gpu_baseline_result(results_doc, results_path, sweep_case, artifact_dir, case_state, force_rerun);

            if (pending_points.empty()) {
                std::cout << "All sweep points already completed for this case." << std::endl;
                continue;
            }

            for (const auto &point : pending_points) {
                const int64_t gpu_k = sweep_case.K - point.npu_k;
                const std::string result_key = make_result_key(sweep_case, artifact_dir, point.npu_k);

                json result_entry = {
                    {"M", sweep_case.M},
                    {"K", sweep_case.K},
                    {"N", sweep_case.N},
                    {"artifact_dir", artifact_dir.string()},
                    {"npu_k", point.npu_k},
                    {"gpu_k", gpu_k},
                    {"pdi_path", point.pdi_path.string()},
                    {"inst_path", point.inst_path.string()},
                    {"status", "in_progress"},
                    {"updated_at_utc", now_utc_iso8601()},
                };
                results_doc["results"][result_key] = result_entry;
                write_results_document(results_path, results_doc);

                std::cout << "Running point npuK=" << point.npu_k << " gpuK=" << gpu_k << std::endl;

                try {
                    load_active_context(point);
                    RuntimePointState point_state = prepare_point_state(case_state, sweep_case, point.npu_k);

                    case_state.split_output.zero_();
                    point_state.npu_output.zero_();
                    torch::cuda::synchronize();
                    run_split_once(case_state.split_output, point_state, sweep_case.N);

                    const auto diff = (case_state.reference_output - case_state.split_output).abs();
                    const double max_abs_diff = diff.max().item<double>();
                    const bool allclose = torch::allclose(case_state.reference_output, case_state.split_output, CHECK_RTOL, CHECK_ATOL);

                    BenchmarkStats stats = benchmark_split(case_state.split_output, point_state, sweep_case.M, sweep_case.K, sweep_case.N);

                    results_doc["results"][result_key] = {
                        {"M", sweep_case.M},
                        {"K", sweep_case.K},
                        {"N", sweep_case.N},
                        {"artifact_dir", artifact_dir.string()},
                        {"npu_k", point.npu_k},
                        {"gpu_k", gpu_k},
                        {"pdi_path", point.pdi_path.string()},
                        {"inst_path", point.inst_path.string()},
                        {"status", "completed"},
                        {"avg_latency_us", stats.avg_latency_us},
                        {"min_latency_us", stats.min_latency_us},
                        {"max_latency_us", stats.max_latency_us},
                        {"throughput_gops", stats.throughput_gops},
                        {"allclose", allclose},
                        {"max_abs_diff", max_abs_diff},
                        {"latencies_us", stats.latencies_us},
                        {"updated_at_utc", now_utc_iso8601()},
                    };
                    write_results_document(results_path, results_doc);

                    std::cout << "Completed npuK=" << point.npu_k << " avg_latency_us=" << stats.avg_latency_us
                              << " throughput_gops=" << stats.throughput_gops << " allclose=" << (allclose ? "true" : "false")
                              << " max_abs_diff=" << max_abs_diff << std::endl;
                } catch (const std::exception &point_error) {
                    results_doc["results"][result_key] = {
                        {"M", sweep_case.M},
                        {"K", sweep_case.K},
                        {"N", sweep_case.N},
                        {"artifact_dir", artifact_dir.string()},
                        {"npu_k", point.npu_k},
                        {"gpu_k", gpu_k},
                        {"pdi_path", point.pdi_path.string()},
                        {"inst_path", point.inst_path.string()},
                        {"status", "failed"},
                        {"error", point_error.what()},
                        {"updated_at_utc", now_utc_iso8601()},
                    };
                    write_results_document(results_path, results_doc);
                    std::cerr << "Point failed for npuK=" << point.npu_k << ": " << point_error.what() << std::endl;
                }
            }
        }

        std::cout.flush();
        std::cerr.flush();
        std::_Exit(0);
    } catch (const std::exception &error) {
        std::cerr << "Fatal error: " << error.what() << std::endl;
        std::cerr.flush();
        std::_Exit(1);
    }
}
