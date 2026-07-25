#include <bits/stdc++.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "npu_matmul_func/npu_matmul_func.hpp"

int npu_debug_verbosity = 1;

void set_npu_debug_verbosity(int level) { npu_debug_verbosity = level; }

void allocate_heap_and_error(int drv_fd) {
    __u32 heap_handle;
    // Reserve some device memory for the heap
    if (alloc_heap(drv_fd, 64 * 1024 * 1024, &heap_handle) < 0) {
        perror("Error allocating device heap");
        if (npu_debug_verbosity >= 1)
            printf("Closing\n");
        close(drv_fd);
        if (npu_debug_verbosity >= 1)
            printf("Done\n");
    }
}

int createHWInstctxt(int drv_fd, struct hwInstctxt &thisctxt, const char *pdi_path, const char *dpu_inst_path,
                     const unsigned int num_tiles, bool npu_is_bin) {
    if (npu_debug_verbosity >= 1)
        std::cout << "Started Creating HW Context " << std::endl;

    int ret;

    if (npu_debug_verbosity >= 1)
        printf("Loading PDI\n");
    ret = load_pdi(drv_fd, &thisctxt.pdi_vaddr, &thisctxt.pdi_sram_vaddr, &thisctxt.pdi_handle, pdi_path,
                   &thisctxt.pdi_size);
    if (ret < 0) {
        if (npu_debug_verbosity >= 1) {
            printf("Error %i loading PDI\n", ret);
            printf("Closing\n");
        }
        close(drv_fd);
        if (npu_debug_verbosity >= 1)
            printf("Done\n");
        return -1;
    }

    if (npu_debug_verbosity >= 1)
        printf("Loading DPU instructions\n");
    ret = load_instructions(drv_fd, &thisctxt.dpu_0_vaddr, &thisctxt.dpu_0_sram_vaddr, &thisctxt.dpu_0_handle,
                            dpu_inst_path, &thisctxt.num_dpu_0_insts, npu_is_bin);
    if (ret < 0) {
        if (npu_debug_verbosity >= 1) {
            printf("Error %i loading DPU instructions\n", ret);
            printf("Closing\n");
        }
        close(drv_fd);
        if (npu_debug_verbosity >= 1)
            printf("Done\n");
        return -1;
    }

    if (npu_debug_verbosity >= 1) {
        printf("DPU 0 instructions @:             %p\n", (void *)thisctxt.dpu_0_vaddr);
        printf("PDI file @:                     %p\n", (void *)thisctxt.pdi_vaddr);
        printf("PDI handle @:                     %d\n", thisctxt.pdi_handle);
    }

    // Allocating a structure to store QOS information
    struct amdxdna_qos_info *qos = (struct amdxdna_qos_info *)malloc(sizeof(struct amdxdna_qos_info));
    qos->gops = 0;
    qos->fps = 0;
    qos->dma_bandwidth = 0;
    qos->latency = 0;
    qos->frame_exec_time = 0;
    qos->priority = 0;

    // This is the structure that we pass
    struct amdxdna_drm_create_ctx create_hw_ctx = {
        .ext = 0,
        .ext_flags = 0,
        .qos_p = (__u64)qos,
        .umq_bo = 0,
        .log_buf_bo = 0,
        .max_opc = 0x800, // Not sure what this is but this was the value used
        .num_tiles = num_tiles,
        .mem_size = 0,
        .umq_doorbell = 0,
    };
    ret = ioctl(drv_fd, DRM_IOCTL_AMDXDNA_CREATE_CTX, &create_hw_ctx);
    if (ret != 0) {
        perror("Failed to create hwctx - 0");
        return -1;
    }

    // Creating a structure to configure the CU
    struct amdxdna_cu_config cu_config = {
        .cu_bo = thisctxt.pdi_handle,
        .cu_func = 0,
    };

    // Creating a structure to configure the hardware context
    // Allocate memory for the struct plus the flexible array member
    size_t param_size = sizeof(struct amdxdna_ctx_param_config_cu) + sizeof(struct amdxdna_cu_config);
    struct amdxdna_ctx_param_config_cu *param_config_cu =
        (struct amdxdna_ctx_param_config_cu *)malloc(param_size);
    if (!param_config_cu) {
        perror("Failed to allocate memory for param_config_cu");
        return -1;
    }

    param_config_cu->num_cus = 1;
    param_config_cu->cu_configs[0] = cu_config;

    if (npu_debug_verbosity >= 1)
        printf("Size of param_config_cu: 0x%lx\n", param_size);

    // Configuring the hardware context with the PDI
    struct amdxdna_drm_config_ctx config_hw_ctx = {
        .handle = create_hw_ctx.handle,
        .param_type = DRM_AMDXDNA_CTX_CONFIG_CU,
        .param_val = (__u64)param_config_cu,              // Pass in the pointer to the param value
        .param_val_size = static_cast<__u32>(param_size), // Size of param config CU
    };
    ret = ioctl(drv_fd, DRM_IOCTL_AMDXDNA_CONFIG_CTX, &config_hw_ctx);
    if (ret != 0) {
        perror("Failed to config hwctx - 1");
        return -1;
    }
    thisctxt.hw_ctx = create_hw_ctx;

    free(param_config_cu);

    if (npu_debug_verbosity >= 1)
        std::cout << "Created HW Context " << std::endl;

    return 0;
}

int createHWctxt(int drv_fd, struct hwctxt &thisctxt, const char *pdi_path, const unsigned int num_tiles) {
    if (npu_debug_verbosity >= 1)
        std::cout << "Started Creating HW Context " << std::endl;

    int ret;

    if (npu_debug_verbosity >= 1)
        printf("Loading PDI\n");
    ret = load_pdi(drv_fd, &thisctxt.pdi_vaddr, &thisctxt.pdi_sram_vaddr, &thisctxt.pdi_handle, pdi_path,
                   &thisctxt.pdi_size);
    if (ret < 0) {
        if (npu_debug_verbosity >= 1) {
            printf("Error %i loading PDI\n", ret);
            printf("Closing\n");
        }
        close(drv_fd);
        if (npu_debug_verbosity >= 1)
            printf("Done\n");
        return -1;
    }
    if (npu_debug_verbosity >= 1) {
        printf("PDI file @:                     %p\n", (void *)thisctxt.pdi_vaddr);
        printf("PDI handle @:                     %d\n", thisctxt.pdi_handle);
    }

    // Allocating a structure to store QOS information
    struct amdxdna_qos_info *qos = (struct amdxdna_qos_info *)malloc(sizeof(struct amdxdna_qos_info));
    qos->gops = 0;
    qos->fps = 0;
    qos->dma_bandwidth = 0;
    qos->latency = 0;
    qos->frame_exec_time = 0;
    qos->priority = 0;

    // This is the structure that we pass
    struct amdxdna_drm_create_ctx create_hw_ctx = {
        .ext = 0,
        .ext_flags = 0,
        .qos_p = (__u64)qos,
        .umq_bo = 0,
        .log_buf_bo = 0,
        .max_opc = 0x800, // Not sure what this is but this was the value used
        .num_tiles = num_tiles,
        .mem_size = 0,
        .umq_doorbell = 0,
    };
    ret = ioctl(drv_fd, DRM_IOCTL_AMDXDNA_CREATE_CTX, &create_hw_ctx);
    if (ret != 0) {
        perror("Failed to create hwctx - 0");
        return -1;
    }

    // Creating a structure to configure the CU
    struct amdxdna_cu_config cu_config = {
        .cu_bo = thisctxt.pdi_handle,
        .cu_func = 0,
    };

    // Creating a structure to configure the hardware context
    // Allocate memory for the struct plus the flexible array member
    size_t param_size = sizeof(struct amdxdna_ctx_param_config_cu) + sizeof(struct amdxdna_cu_config);
    struct amdxdna_ctx_param_config_cu *param_config_cu =
        (struct amdxdna_ctx_param_config_cu *)malloc(param_size);
    if (!param_config_cu) {
        perror("Failed to allocate memory for param_config_cu");
        return -1;
    }

    param_config_cu->num_cus = 1;
    param_config_cu->cu_configs[0] = cu_config;

    if (npu_debug_verbosity >= 1)
        printf("Size of param_config_cu: 0x%lx\n", param_size);

    // Configuring the hardware context with the PDI
    struct amdxdna_drm_config_ctx config_hw_ctx = {
        .handle = create_hw_ctx.handle,
        .param_type = DRM_AMDXDNA_CTX_CONFIG_CU,
        .param_val = (__u64)param_config_cu,              // Pass in the pointer to the param value
        .param_val_size = static_cast<__u32>(param_size), // Size of param config CU
    };
    ret = ioctl(drv_fd, DRM_IOCTL_AMDXDNA_CONFIG_CTX, &config_hw_ctx);
    if (ret != 0) {
        perror("Failed to config hwctx - 1");
        return -1;
    }
    thisctxt.hw_ctx = create_hw_ctx;

    free(param_config_cu);

    if (npu_debug_verbosity >= 1)
        std::cout << "Created HW Context " << std::endl;

    return 0;
}

int createInstctxt(int drv_fd, struct instctxt &thisctxt, const char *dpu_inst_path, bool npu_is_bin) {
    if (npu_debug_verbosity >= 1)
        std::cout << "Started Creating Inst Context " << std::endl;
    int ret;

    if (npu_debug_verbosity >= 1)
        printf("Loading DPU instructions\n");
    ret = load_instructions(drv_fd, &thisctxt.dpu_0_vaddr, &thisctxt.dpu_0_sram_vaddr, &thisctxt.dpu_0_handle,
                            dpu_inst_path, &thisctxt.num_dpu_0_insts, npu_is_bin);
    if (ret < 0) {
        if (npu_debug_verbosity >= 1) {
            printf("Error %i loading DPU instructions\n", ret);
            printf("Closing\n");
        }
        close(drv_fd);
        if (npu_debug_verbosity >= 1)
            printf("Done\n");
        return -1;
    }
    if (npu_debug_verbosity >= 1)
        printf("DPU 0 instructions @:             %p\n", (void *)thisctxt.dpu_0_vaddr);

    return 0;
}

int create_matmul_bo_context(int drv_fd, struct matmul_bo_ctxt &new_bos, size_t sizeA, size_t sizeB, size_t sizeC) {
    int ret;

    // create bo
    ret = create_dev_bo(drv_fd, &new_bos.A_bo_vaddr, &new_bos.A_bo_sram_vaddr, &new_bos.A_bo_handle, sizeA);
    // print vaddr and handle
    if (npu_debug_verbosity >= 1) {
        std::cout << "A @:             " << new_bos.A_bo_vaddr << std::endl;
        std::cout << "A handle:        " << new_bos.A_bo_handle << std::endl;
    }
    if (ret < 0) {
        if (npu_debug_verbosity >= 1) {
            printf("Error %i creating data 0\n", ret);
            printf("Closing\n");
        }
        close(drv_fd);
        if (npu_debug_verbosity >= 1)
            printf("Done\n");
        return -1;
    }

    ret = create_dev_bo(drv_fd, &new_bos.B_bo_vaddr, &new_bos.B_bo_sram_vaddr, &new_bos.B_bo_handle, sizeB);
    // print vaddr and handle
    if (npu_debug_verbosity >= 1) {
        std::cout << "B @:             " << new_bos.B_bo_vaddr << std::endl;
        std::cout << "B handle:        " << new_bos.B_bo_handle << std::endl;
    }
    if (ret < 0) {
        if (npu_debug_verbosity >= 1) {
            printf("Error %i creating data 0\n", ret);
            printf("Closing\n");
        }
        close(drv_fd);
        if (npu_debug_verbosity >= 1)
            printf("Done\n");
        return -1;
    }

    ret = create_dev_bo(drv_fd, &new_bos.C_bo_vaddr, &new_bos.C_bo_sram_vaddr, &new_bos.C_bo_handle, sizeC);
    // print vaddr and handle
    if (npu_debug_verbosity >= 1) {
        std::cout << "C @:             " << new_bos.C_bo_vaddr << std::endl;
        std::cout << "C handle:        " << new_bos.C_bo_handle << std::endl;
    }
    if (ret < 0) {
        if (npu_debug_verbosity >= 1) {
            printf("Error %i creating data 0\n", ret);
            printf("Closing\n");
        }
        close(drv_fd);
        if (npu_debug_verbosity >= 1)
            printf("Done\n");
        return -1;
    }
    return 0;
}

int create_cmd_packet(int drv_fd, __u32 pdi_handle, uint64_t dpu_0_sram_vaddr, __u32 dpu_0_handle,
                      __u32 num_dpu_0_insts, uint64_t input_A, uint64_t input_B, uint64_t output_C,
                      __u32 input_A_handle, __u32 input_B_handle, __u32 output_C_handle,
                      struct amdxdna_drm_create_ctx &hw_ctx, struct amdxdna_drm_exec_cmd &cmd, uint32_t *bo_args,
                      uint32_t bo_args_count) {
    /////////////////////////////////////////////////////////////////////////////////
    // Step 2: Configuring the CMD BOs with the different instruction sequences

    int ret;
    struct amdxdna_drm_create_bo create_cmd_bo_0 = {
        .size = PACKET_SIZE,
        .type = AMDXDNA_BO_CMD,
    };
    int cmd_bo_ret = ioctl(drv_fd, DRM_IOCTL_AMDXDNA_CREATE_BO, &create_cmd_bo_0);
    if (cmd_bo_ret != 0) {
        perror("Failed to create cmd_0");
        return -1;
    }

    struct amdxdna_drm_get_bo_info cmd_bo_0_get_bo_info = {.handle = create_cmd_bo_0.handle};
    ret = ioctl(drv_fd, DRM_IOCTL_AMDXDNA_GET_BO_INFO, &cmd_bo_0_get_bo_info);
    if (ret != 0) {
        perror("Failed to get cmd BO 0 info");
        return -2;
    }

    // Writing the first packet to the queue
    struct amdxdna_cmd *cmd_0 = (struct amdxdna_cmd *)mmap(0, PACKET_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, drv_fd,
                                                           cmd_bo_0_get_bo_info.map_offset);
    cmd_0->state = 1; // ERT_CMD_STATE_NEW;
    cmd_0->extra_cu_masks = 0;
    cmd_0->count = 0xF;   // NOTE: For some reason this needs to be larger
    cmd_0->opcode = 0x0;  // ERT_START_CU;
    cmd_0->data[0] = 0x3; // NOTE: This one seems to be skipped
    cmd_0->data[1] = 0x3; // Transaction opcode
    cmd_0->data[2] = 0x0;
    cmd_0->data[3] = dpu_0_sram_vaddr;
    cmd_0->data[4] = 0x0;
    cmd_0->data[5] = num_dpu_0_insts;                // Size of DPU instruction
    cmd_0->data[6] = input_A & 0xFFFFFFFF;           // Input low
    cmd_0->data[7] = (input_A >> 32) & 0xFFFFFFFF;   // Input high
    cmd_0->data[8] = input_B & 0xFFFFFFFF;           // Output low
    cmd_0->data[9] = (input_B >> 32) & 0xFFFFFFFF;   // Output high
    cmd_0->data[10] = output_C & 0xFFFFFFFF;         // Output low
    cmd_0->data[11] = (output_C >> 32) & 0xFFFFFFFF; // Output high

    /////////////////////////////////////////////////////////////////////////////////
    // Step 3: Submit commands -- This requires creating a BO_EXEC that contains
    // the command chain that points to the instruction sequences just created

    // Allocate a command chain
    void *bo_cmd_chain_buf = NULL;
    cmd_bo_ret = posix_memalign(&bo_cmd_chain_buf, 4096, 4096);
    if (cmd_bo_ret != 0 || bo_cmd_chain_buf == NULL) {
        if (npu_debug_verbosity >= 1)
            printf("[ERROR] Failed to allocate cmd_bo buffer of size %d\n", 4096);
    }

    struct amdxdna_drm_create_bo create_cmd_chain_bo = {
        .size = 4096,
        .type = AMDXDNA_BO_CMD,
    };
    cmd_bo_ret = ioctl(drv_fd, DRM_IOCTL_AMDXDNA_CREATE_BO, &create_cmd_chain_bo);
    if (cmd_bo_ret != 0) {
        perror("Failed to create command chain BO");
        return -1;
    }

    struct amdxdna_drm_get_bo_info cmd_chain_bo_get_bo_info = {.handle = create_cmd_chain_bo.handle};
    ret = ioctl(drv_fd, DRM_IOCTL_AMDXDNA_GET_BO_INFO, &cmd_chain_bo_get_bo_info);
    if (ret != 0) {
        perror("Failed to get cmd BO 0 info");
        return -2;
    }

    struct amdxdna_cmd *cmd_chain = (struct amdxdna_cmd *)mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, drv_fd,
                                                               cmd_chain_bo_get_bo_info.map_offset);

    // Writing information to the command buffer
    struct amdxdna_cmd_chain *cmd_chain_payload = (struct amdxdna_cmd_chain *)(cmd_chain->data);
    cmd_chain->state = 1; // ERT_CMD_STATE_NEW;
    cmd_chain->extra_cu_masks = 0;
    cmd_chain->count = 0xA;   // TODO: Why is this the value?
    cmd_chain->opcode = 0x13; // ERT_CMD_CHAIN
    cmd_chain_payload->command_count = 1;
    cmd_chain_payload->submit_index = 0;
    cmd_chain_payload->error_index = 0;
    cmd_chain_payload->data[0] = create_cmd_bo_0.handle;

    // Reading the user buffers -> failes due to bo's not being the correct type
    // sync_bo(drv_fd, create_cmd_chain_bo.handle);
    // sync_bo(drv_fd, create_cmd_bo_0.handle);

    // uint32_t bo_args[4] = {dpu_0_handle, input_A_handle, input_B_handle, output_C_handle};
    // Perform a submit cmd
    struct amdxdna_drm_exec_cmd exec_cmd_0 = {
        .ext = 0,
        .ext_flags = 0,
        .ctx = hw_ctx.handle,
        .type = AMDXDNA_CMD_SUBMIT_EXEC_BUF,
        .cmd_handles = create_cmd_chain_bo.handle,
        .args = (__u64)bo_args,
        .cmd_count = 1,
        .arg_count = bo_args_count,
    };

    cmd = exec_cmd_0;

    return 0;
}

// Function to create and execute a command packet chain, and wait for completion
int execute_command_packet(int drv_fd, struct amdxdna_drm_create_ctx hw_ctx, __u32 pdi_handle, uint64_t dpu_sram_vaddr,
                           __u32 dpu_handle, __u32 num_dpu_0_insts, uint64_t input_A, uint64_t input_B,
                           uint64_t output_C, __u32 input_A_handle, __u32 input_B_handle, __u32 output_C_handle,
                           uint32_t *bo_args, uint32_t bo_args_count, double times[2]) {

    auto cmd_create_start = std::chrono::high_resolution_clock::now();
    struct amdxdna_drm_exec_cmd exec_cmd;
    // Create command packet chain
    int ret =
        create_cmd_packet(drv_fd, pdi_handle, dpu_sram_vaddr, dpu_handle, num_dpu_0_insts, input_A, input_B, output_C,
                          input_A_handle, input_B_handle, output_C_handle, hw_ctx, exec_cmd, bo_args, bo_args_count);

    if (ret != 0) {
        perror("Failed to create command packet chain");
        return -1;
    }
    auto cmd_create_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cmd_create_duration = cmd_create_end - cmd_create_start;
    times[0] = cmd_create_duration.count();

    auto run_start = std::chrono::high_resolution_clock::now();
    // Execute the command
    ret = ioctl(drv_fd, DRM_IOCTL_AMDXDNA_EXEC_CMD, &exec_cmd);
    if (ret != 0) {
        perror("Failed to submit work");
        return -1;
    }

    // Wait for the command to complete
    struct amdxdna_drm_wait_cmd wait_cmd = {
        .ctx = hw_ctx.handle,
        .timeout = 800, // 50ms timeout
        .seq = exec_cmd.seq,
    };

    ret = ioctl(drv_fd, DRM_IOCTL_AMDXDNA_WAIT_CMD, &wait_cmd);
    if (ret != 0) {
        perror("Failed to wait");
        return -1;
    }
    auto run_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> run_duration = run_end - run_start;
    times[1] = run_duration.count();

    return 0;
}

int create_cmd_packet_chain_bmm(int drv_fd, __u32 pdi_handle, uint64_t dpu_0_sram_vaddr, __u32 dpu_0_handle,
                                __u32 num_dpu_0_insts, uint64_t input_A, uint64_t input_B, uint64_t output_C,
                                __u32 input_A_handle, __u32 input_B_handle, __u32 output_C_handle,
                                struct amdxdna_drm_create_ctx &hw_ctx, struct amdxdna_drm_exec_cmd &cmd,
                                uint32_t *bo_args, uint32_t bo_args_count, uint32_t gemm_dims[4],
                                uint32_t input_data_bytes, uint32_t output_data_bytes) {

    //  gemm_dims[4]: B, M, K, N

    /////////////////////////////////////////////////////////////////////////////////
    // Step 2: Configuring the CMD BOs with the different instruction sequences

    int ret;

    // storage for the create and info structs
    std::vector<amdxdna_drm_create_bo> create_cmd_bo(gemm_dims[0]);
    std::vector<amdxdna_drm_get_bo_info> cmd_bo_get_bo_info(gemm_dims[0]);
    // std::vector<amdxdna_cmd *> cmd_ptrs(NUM_CMD_BOS);

    for (int i = 0; i < gemm_dims[0]; i++) {
        create_cmd_bo[i] = {
            .size = PACKET_SIZE,
            .type = AMDXDNA_BO_CMD,
        };
        ret = ioctl(drv_fd, DRM_IOCTL_AMDXDNA_CREATE_BO, &create_cmd_bo[i]);
        if (ret != 0) {
            perror("Failed to create cmd_0");
            return -1;
        }

        cmd_bo_get_bo_info[i] = {.handle = create_cmd_bo[i].handle};
        ret = ioctl(drv_fd, DRM_IOCTL_AMDXDNA_GET_BO_INFO, &cmd_bo_get_bo_info[i]);
        if (ret != 0) {
            perror("Failed to get cmd BO 0 info");
            return -2;
        }

        uint64_t input_A_batch = input_A + i * gemm_dims[1] * gemm_dims[2] * input_data_bytes;
        uint64_t input_B_batch = input_B + i * gemm_dims[2] * gemm_dims[3] * input_data_bytes;
        uint64_t output_C_batch = output_C + i * gemm_dims[1] * gemm_dims[3] * output_data_bytes;

        // Writing the first packet to the queue
        struct amdxdna_cmd *cmd = (struct amdxdna_cmd *)mmap(0, PACKET_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, drv_fd,
                                                             cmd_bo_get_bo_info[i].map_offset);
        cmd->state = 1; // ERT_CMD_STATE_NEW;
        cmd->extra_cu_masks = 0;
        cmd->count = 0xF;   // NOTE: For some reason this needs to be larger
        cmd->opcode = 0x0;  // ERT_START_CU;
        cmd->data[0] = 0x3; // NOTE: This one seems to be skipped
        cmd->data[1] = 0x3; // Transaction opcode
        cmd->data[2] = 0x0;
        cmd->data[3] = dpu_0_sram_vaddr;
        cmd->data[4] = 0x0;
        cmd->data[5] = num_dpu_0_insts;                      // Size of DPU instruction
        cmd->data[6] = input_A_batch & 0xFFFFFFFF;           // Input low
        cmd->data[7] = (input_A_batch >> 32) & 0xFFFFFFFF;   // Input high
        cmd->data[8] = input_B_batch & 0xFFFFFFFF;           // Output low
        cmd->data[9] = (input_B_batch >> 32) & 0xFFFFFFFF;   // Output high
        cmd->data[10] = output_C_batch & 0xFFFFFFFF;         // Output low
        cmd->data[11] = (output_C_batch >> 32) & 0xFFFFFFFF; // Output high
    }

    /////////////////////////////////////////////////////////////////////////////////
    // Step 3: Submit commands -- This requires creating a BO_EXEC that contains
    // the command chain that points to the instruction sequences just created

    // Allocate a command chain
    void *bo_cmd_chain_buf = NULL;
    ret = posix_memalign(&bo_cmd_chain_buf, 4096, 4096);
    if (ret != 0 || bo_cmd_chain_buf == NULL) {
        if (npu_debug_verbosity >= 1)
            printf("[ERROR] Failed to allocate cmd_bo buffer of size %d\n", 4096);
    }

    struct amdxdna_drm_create_bo create_cmd_chain_bo = {
        .size = 4096,
        .type = AMDXDNA_BO_CMD,
    };
    ret = ioctl(drv_fd, DRM_IOCTL_AMDXDNA_CREATE_BO, &create_cmd_chain_bo);
    if (ret != 0) {
        perror("Failed to create command chain BO");
        return -1;
    }

    struct amdxdna_drm_get_bo_info cmd_chain_bo_get_bo_info = {.handle = create_cmd_chain_bo.handle};
    ret = ioctl(drv_fd, DRM_IOCTL_AMDXDNA_GET_BO_INFO, &cmd_chain_bo_get_bo_info);
    if (ret != 0) {
        perror("Failed to get cmd BO 0 info");
        return -2;
    }

    struct amdxdna_cmd *cmd_chain = (struct amdxdna_cmd *)mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, drv_fd,
                                                               cmd_chain_bo_get_bo_info.map_offset);

    // Writing information to the command buffer
    struct amdxdna_cmd_chain *cmd_chain_payload = (struct amdxdna_cmd_chain *)(cmd_chain->data);
    cmd_chain->state = 1; // ERT_CMD_STATE_NEW;
    cmd_chain->extra_cu_masks = 0;
    cmd_chain->count = 0xA;   // TODO: Why is this the value?
    cmd_chain->opcode = 0x13; // ERT_CMD_CHAIN
    cmd_chain_payload->command_count = gemm_dims[0];
    cmd_chain_payload->submit_index = 0;
    cmd_chain_payload->error_index = 0;
    for (int i = 0; i < gemm_dims[0]; i++) {
        cmd_chain_payload->data[i] = create_cmd_bo[i].handle;
    }

    // Reading the user buffers -> failes due to bo's not being the correct type
    // sync_bo(drv_fd, create_cmd_chain_bo.handle);
    // sync_bo(drv_fd, create_cmd_bo_0.handle);

    // uint32_t bo_args[4] = {dpu_0_handle, input_A_handle, input_B_handle, output_C_handle};
    // Perform a submit cmd
    struct amdxdna_drm_exec_cmd exec_cmd_0 = {
        .ext = 0,
        .ext_flags = 0,
        .ctx = hw_ctx.handle,
        .type = AMDXDNA_CMD_SUBMIT_EXEC_BUF,
        .cmd_handles = create_cmd_chain_bo.handle,
        .args = (__u64)bo_args,
        .cmd_count = 1,
        .arg_count = bo_args_count,
    };

    cmd = exec_cmd_0;

    return 0;
}
