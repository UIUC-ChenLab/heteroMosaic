#include "third_party/nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

constexpr int ITER = 16;
constexpr int WARMUP_CYCLES = 4;
constexpr int64_t MIN_M = 256;
constexpr int64_t MAX_M = 4096;
constexpr int64_t STEP_M = 256;

struct SweepCase {
    int64_t M;
    int64_t K;
    int64_t N;
    bool enabled;
    std::string label;
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
    torch::Tensor weight;
    torch::Tensor output;
};

std::string now_utc_iso8601() {
    const std::time_t now = std::time(nullptr);
    std::tm tm_now {};
    gmtime_r(&now, &tm_now);

    std::ostringstream oss;
    oss << std::put_time(&tm_now, "%Y-%m-%dT%H:%M:%SZ");
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
    if (fs::exists(exec_dir / "py/tileFuse_figs/results")) {
        return exec_dir.string();
    }
    if (fs::exists(exec_dir.parent_path() / "py/tileFuse_figs/results")) {
        return exec_dir.parent_path().string();
    }

    return "/home/greg/Desktop/heteroMosaic";
}

fs::path get_default_results_path(const std::string &root_dir) {
    return fs::path(root_dir) / "py/tileFuse_figs/results/tensor_parallel_gemm_torch_bf16_sweepM_results.json";
}

std::string make_result_key(const SweepCase &sweep_case) {
    std::ostringstream oss;
    oss << sweep_case.M << "x" << sweep_case.K << "x" << sweep_case.N << "__torch_bf16";
    return oss.str();
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

json make_metadata(const std::string &root_dir, const fs::path &results_path, const std::vector<SweepCase> &cases) {
    json enabled_cases = json::array();
    for (const auto &sweep_case : cases) {
        enabled_cases.push_back({
            {"M", sweep_case.M},
            {"K", sweep_case.K},
            {"N", sweep_case.N},
            {"enabled", sweep_case.enabled},
            {"label", sweep_case.label},
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
            {"operator", "torch::matmul"},
            {"input_dtype", "bf16"},
            {"weight_dtype", "bf16"},
            {"device", "cuda"},
            {"sweep_dim", "M"},
            {"min_m", MIN_M},
            {"max_m", MAX_M},
            {"step_m", STEP_M},
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
        {"operator", "torch::matmul"},
        {"input_dtype", "bf16"},
        {"weight_dtype", "bf16"},
        {"device", "cuda"},
        {"sweep_dim", "M"},
        {"min_m", MIN_M},
        {"max_m", MAX_M},
        {"step_m", STEP_M},
    };

    json enabled_cases = json::array();
    for (const auto &sweep_case : cases) {
        enabled_cases.push_back({
            {"M", sweep_case.M},
            {"K", sweep_case.K},
            {"N", sweep_case.N},
            {"enabled", sweep_case.enabled},
            {"label", sweep_case.label},
        });
    }
    doc["cases"] = enabled_cases;
    if (!doc.contains("results") || !doc["results"].is_object()) {
        doc["results"] = json::object();
    }
}

RuntimeCaseState prepare_case_state(const SweepCase &sweep_case) {
    torch::manual_seed(42);

    const auto options = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    RuntimeCaseState state;
    state.input = (torch::rand({sweep_case.M, sweep_case.K}, options) * 0.1f).contiguous();
    state.weight = (torch::rand({sweep_case.K, sweep_case.N}, options) * 0.1f).contiguous();
    state.output = torch::empty({sweep_case.M, sweep_case.N}, options);
    torch::cuda::synchronize();
    return state;
}

torch::Tensor run_matmul_once(const torch::Tensor &input, const torch::Tensor &weight) {
    return torch::matmul(input, weight);
}

BenchmarkStats benchmark_matmul(RuntimeCaseState &state, const SweepCase &sweep_case) {
    for (int warmup = 0; warmup < WARMUP_CYCLES; ++warmup) {
        state.output = run_matmul_once(state.input, state.weight);
    }
    torch::cuda::synchronize();

    BenchmarkStats stats;
    stats.min_latency_us = std::numeric_limits<double>::max();
    stats.max_latency_us = 0.0;
    stats.latencies_us.reserve(ITER);

    for (int iter = 0; iter < ITER; ++iter) {
        const auto iter_start = std::chrono::high_resolution_clock::now();
        state.output = run_matmul_once(state.input, state.weight);
        torch::cuda::synchronize();
        const auto iter_end = std::chrono::high_resolution_clock::now();
        const auto iter_latency = std::chrono::duration_cast<std::chrono::microseconds>(iter_end - iter_start).count();

        stats.latencies_us.push_back(iter_latency);
        stats.min_latency_us = std::min(stats.min_latency_us, static_cast<double>(iter_latency));
        stats.max_latency_us = std::max(stats.max_latency_us, static_cast<double>(iter_latency));
    }

    double total_latency = 0.0;
    for (const int64_t latency : stats.latencies_us) {
        total_latency += static_cast<double>(latency);
    }

    stats.avg_latency_us = total_latency / static_cast<double>(stats.latencies_us.size());
    const double ops =
        2.0 * static_cast<double>(sweep_case.M) * static_cast<double>(sweep_case.K) * static_cast<double>(sweep_case.N);
    stats.throughput_gops = ops / (stats.avg_latency_us * 1000.0);
    return stats;
}

void run_case(json &results_doc, const fs::path &results_path, const SweepCase &sweep_case, bool force_rerun) {
    const std::string result_key = make_result_key(sweep_case);
    const bool already_completed =
        results_doc["results"].contains(result_key) && results_doc["results"][result_key].value("status", "") == "completed";
    if (already_completed && !force_rerun) {
        std::cout << "Skipping completed case " << sweep_case.M << "x" << sweep_case.K << "x" << sweep_case.N << std::endl;
        return;
    }

    results_doc["results"][result_key] = {
        {"M", sweep_case.M},
        {"K", sweep_case.K},
        {"N", sweep_case.N},
        {"label", sweep_case.label},
        {"status", "in_progress"},
        {"updated_at_utc", now_utc_iso8601()},
    };
    write_results_document(results_path, results_doc);

    std::cout << "\n=== Case " << sweep_case.M << "x" << sweep_case.K << "x" << sweep_case.N << " ===" << std::endl;
    std::cout << "Operator: torch::matmul, dtype: bf16 x bf16, iterations: " << ITER << std::endl;

    RuntimeCaseState state = prepare_case_state(sweep_case);
    BenchmarkStats stats = benchmark_matmul(state, sweep_case);

    if (state.output.size(0) != sweep_case.M || state.output.size(1) != sweep_case.N || state.output.scalar_type() != torch::kBFloat16) {
        throw std::runtime_error("Unexpected torch::matmul output shape or dtype");
    }

    results_doc["results"][result_key] = {
        {"M", sweep_case.M},
        {"K", sweep_case.K},
        {"N", sweep_case.N},
        {"label", sweep_case.label},
        {"status", "completed"},
        {"avg_latency_us", stats.avg_latency_us},
        {"min_latency_us", stats.min_latency_us},
        {"max_latency_us", stats.max_latency_us},
        {"throughput_gops", stats.throughput_gops},
        {"latencies_us", stats.latencies_us},
        {"updated_at_utc", now_utc_iso8601()},
    };
    write_results_document(results_path, results_doc);

    std::cout << "Completed avg_latency_us=" << stats.avg_latency_us << " min_latency_us=" << stats.min_latency_us
              << " max_latency_us=" << stats.max_latency_us << " throughput_gops=" << stats.throughput_gops << std::endl;
}

std::vector<SweepCase> build_sweep_cases() {
    std::vector<SweepCase> sweep_cases;
    for (int64_t m = MIN_M; m <= MAX_M; m += STEP_M) {
        sweep_cases.push_back({m, 4096, 14336, true, "up_projection"});
    }
    for (int64_t m = MIN_M; m <= MAX_M; m += STEP_M) {
        sweep_cases.push_back({m, 14336, 4096, true, "down_projection"});
    }
    return sweep_cases;
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

        const std::vector<SweepCase> sweep_cases = build_sweep_cases();

        if (!torch::cuda::is_available()) {
            std::cerr << "HIP/CUDA is not available." << std::endl;
            return 1;
        }

        json results_doc = load_results_document(results_path);
        ensure_document_metadata(results_doc, root_dir, results_path, sweep_cases);
        write_results_document(results_path, results_doc);

        for (const auto &sweep_case : sweep_cases) {
            if (sweep_case.enabled) {
                run_case(results_doc, results_path, sweep_case, force_rerun);
            }
        }

        std::cout << "\nResults written to " << results_path << std::endl;
        std::cout.flush();
        std::cerr.flush();
        std::_Exit(0);
    } catch (const std::exception &error) {
        std::cerr << "Fatal error: " << error.what() << std::endl;
        std::cerr.flush();
        std::_Exit(1);
    }
}
