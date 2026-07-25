#pragma once

#include "npu_matmul_func/npu_util.hpp"

// These packets are variable width but using this as a
// maximum size for now
#define PACKET_SIZE 64

/*
 * Interpretation of the beginning of data payload for ERT_CMD_CHAIN in
 * amdxdna_cmd. The rest of the payload in amdxdna_cmd is cmd BO handles.
 */
struct amdxdna_cmd_chain {
    __u32 command_count;
    __u32 submit_index;
    __u32 error_index;
    __u32 reserved[3];
    __u64 data[] __counted_by(command_count);
};

/* Exec buffer command header format */
struct amdxdna_cmd {
    union {
        struct {
            __u32 state : 4;
            __u32 unused : 6;
            __u32 extra_cu_masks : 2;
            __u32 count : 11;
            __u32 opcode : 5;
            __u32 reserved : 4;
        };
        __u32 header;
    };
    __u32 data[] __counted_by(count);
};

struct hwInstctxt {
    uint64_t pdi_vaddr;
    uint64_t pdi_sram_vaddr;
    __u32 pdi_handle;
    uint64_t pdi_size;
    uint64_t dpu_0_vaddr;
    uint64_t dpu_0_sram_vaddr;
    __u32 dpu_0_handle;
    __u32 num_dpu_0_insts;
    struct amdxdna_drm_create_ctx hw_ctx;
};

struct hwctxt {
    uint64_t pdi_vaddr;
    uint64_t pdi_sram_vaddr;
    __u32 pdi_handle;
    uint64_t pdi_size;

    struct amdxdna_drm_create_ctx hw_ctx;
};

struct instctxt {
    uint64_t dpu_0_vaddr;
    uint64_t dpu_0_sram_vaddr;
    __u32 dpu_0_handle;
    __u32 num_dpu_0_insts;
};

struct matmul_bo_ctxt {
    uint64_t A_bo_vaddr;
    uint64_t A_bo_sram_vaddr;
    __u32 A_bo_handle;

    uint64_t B_bo_vaddr;
    uint64_t B_bo_sram_vaddr;
    __u32 B_bo_handle;

    uint64_t C_bo_vaddr;
    uint64_t C_bo_sram_vaddr;
    __u32 C_bo_handle;
};

void allocate_heap_and_error(int drv_fd);

int createHWInstctxt(int drv_fd, struct hwInstctxt &thisctxt, const char *pdi_path, const char *dpu_inst_path,
                     const unsigned int num_tiles, bool npu_is_bin = false);

int createHWctxt(int drv_fd, struct hwctxt &thisctxt, const char *pdi_path, const char *dpu_inst_path,
                 const unsigned int num_tiles);

int createHWctxt(int drv_fd, struct hwctxt &thisctxt, const char *pdi_path, const unsigned int num_tiles);

int createInstctxt(int drv_fd, struct instctxt &thisctxt, const char *dpu_inst_path, bool npu_is_bin = false);

int create_matmul_bo_context(int drv_fd, struct matmul_bo_ctxt &new_bos, size_t sizeA, size_t sizeB, size_t sizeC);

int allocate_and_load_resources(int drv_fd, uint64_t &pdi_vaddr, uint64_t &pdi_sram_vaddr, __u32 &pdi_handle,
                                uint64_t &dpu_0_vaddr, uint64_t &dpu_0_sram_vaddr, __u32 &dpu_0_handle,
                                __u32 &num_dpu_0_insts, const char *pdi_path, const char *dpu_inst_path);

int create_cmd_packet(int drv_fd, __u32 pdi_handle, uint64_t dpu_0_sram_vaddr, __u32 dpu_0_handle,
                      __u32 num_dpu_0_insts, uint64_t input_A, uint64_t input_B, uint64_t output_C,
                      __u32 input_A_handle, __u32 input_B_handle, __u32 output_C_handle,
                      struct amdxdna_drm_create_ctx &hw_ctx, struct amdxdna_drm_exec_cmd &cmd, uint32_t *bo_args,
                      uint32_t bo_args_count);

int execute_command_packet(int drv_fd, struct amdxdna_drm_create_ctx hw_ctx, __u32 pdi_handle, uint64_t dpu_sram_vaddr,
                           __u32 dpu_handle, __u32 num_dpu_0_insts, uint64_t input_A, uint64_t input_B,
                           uint64_t output_C, __u32 input_A_handle, __u32 input_B_handle, __u32 output_C_handle,
                           uint32_t *bo_args, uint32_t bo_args_count, double times[2]);

int create_cmd_packet_chain_bmm(int drv_fd, __u32 pdi_handle, uint64_t dpu_0_sram_vaddr, __u32 dpu_0_handle,
                                __u32 num_dpu_0_insts, uint64_t input_A, uint64_t input_B, uint64_t output_C,
                                __u32 input_A_handle, __u32 input_B_handle, __u32 output_C_handle,
                                struct amdxdna_drm_create_ctx &hw_ctx, struct amdxdna_drm_exec_cmd &cmd,
                                uint32_t *bo_args, uint32_t bo_args_count, uint32_t gemm_dims[4],
                                uint32_t input_data_bytes, uint32_t output_data_bytes);

void set_npu_debug_verbosity(int level);
extern int npu_debug_verbosity;
