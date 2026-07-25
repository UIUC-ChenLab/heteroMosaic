#pragma once

#include "amdxdna_accel.h"
#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

extern int npu_debug_verbosity;

// want to mmap the file
#include <sys/io.h>
#include <sys/mman.h>

// So we can flush the cache
#include <emmintrin.h>

#define QUEUE_SIZE 1024 // Queue buffer size in terms of bytes.
#define MAX_DEVICES 10
#define MAX_NUM_INSTRUCTIONS 16392 // Maximum number of dpu or pdi instructions.

// Dummy packet defines
#define AIR_ADDRESS_ABSOLUTE_RANGE 0x1L
#define AIR_PKT_TYPE_DEVICE_INITIALIZE 0x0010L


inline int get_driver_version(int fd, __u32 *major, __u32 *minor) {
  int ret;
  struct amdxdna_drm_query_aie_version version;

  struct amdxdna_drm_get_info info_params = {
      .param = DRM_AMDXDNA_QUERY_AIE_VERSION,
      .buffer_size = sizeof(version),
      .buffer = (__u64)&version,
  };

  ret = ioctl(fd, DRM_IOCTL_AMDXDNA_GET_INFO, &info_params);
  if (ret == 0) {
    *major = version.major;
    *minor = version.minor;
  }

  return ret;
}

/*
        Allocates a heap on the device by creating a BO of type dev heap
*/
static int alloc_heap(int fd, __u32 size, __u32 *handle) {
  int ret;
  void *heap_buf = NULL;
  const size_t alignment = 64 * 1024 * 1024;
  ret = posix_memalign(&heap_buf, alignment, size);
  if (ret != 0 || heap_buf == NULL) {
    if (npu_debug_verbosity >= 1) printf("[ERROR] Failed to allocate heap buffer of size %d\n", size);
  }

  void *dev_heap_parent = mmap(0, alignment * 2 - 1, PROT_READ | PROT_WRITE,
    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

  if (dev_heap_parent == MAP_FAILED) {
    dev_heap_parent = nullptr;
    return -1;
  }

  struct amdxdna_drm_create_bo create_bo_params = {
      .size = size,
      .type = AMDXDNA_BO_DEV_HEAP,
  };

  ret = ioctl(fd, DRM_IOCTL_AMDXDNA_CREATE_BO, &create_bo_params);
  if (ret == 0 && handle) {
    *handle = create_bo_params.handle;
  }

  struct amdxdna_drm_get_bo_info get_bo_info = {.handle = create_bo_params.handle};
  ret = ioctl(fd, DRM_IOCTL_AMDXDNA_GET_BO_INFO, &get_bo_info);
  if (ret != 0) {
    perror("Failed to get BO info");
    return -2;
  }

  // Need to free the heap buf but still use the address so we can 
  // ensure alignment
  free(heap_buf);
  heap_buf = (void  *)mmap(heap_buf, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, get_bo_info.map_offset);
  if (npu_debug_verbosity >= 1) printf("Heap buffer @:                  %p\n", heap_buf);

  return ret;
}

/*
        Creates a dev bo which is carved out of the heap bo.
*/
static int create_dev_bo(int fd, uint64_t *vaddr, uint64_t *sram_vaddr,
                         __u32 *handle, __u64 size_in_bytes) {

  struct amdxdna_drm_create_bo create_bo = {
      .size = size_in_bytes,
      .type = AMDXDNA_BO_DEV,
  };
  int ret = ioctl(fd, DRM_IOCTL_AMDXDNA_CREATE_BO, &create_bo);
  if (ret != 0) {
    perror("Failed to create BO");
    return -1;
  }

  struct amdxdna_drm_get_bo_info get_bo_info = {.handle = create_bo.handle};
  ret = ioctl(fd, DRM_IOCTL_AMDXDNA_GET_BO_INFO, &get_bo_info);
  if (ret != 0) {
    perror("Failed to get BO info");
    return -2;
  }

  *vaddr = get_bo_info.vaddr;
  *sram_vaddr = get_bo_info.xdna_addr;
  *handle = create_bo.handle;

  return 0;
}

/*
        Creates a shmem bo
*/
static int create_shmem_bo(int fd, uint64_t *vaddr, uint64_t *sram_vaddr,
                           __u32 *handle, __u64 size_in_bytes) {

  const size_t alignment = 64 * 1024 * 1024;
  void *shmem_create = NULL;
  uint64_t *shmem;
  int ret = posix_memalign(&shmem_create, alignment, size_in_bytes);
  if (ret != 0) {
    if (npu_debug_verbosity >= 1) printf("[ERROR] Failed to allocate shmem bo of size %lld\n", size_in_bytes);
  }

  // Touching buffer to map page
  *(uint32_t *)shmem_create = 0xDEADBEEF;

  if (npu_debug_verbosity >= 1) printf("Shmem BO @:                     %p\n", shmem_create);

  struct amdxdna_drm_create_bo create_bo = {
    .vaddr = (__u64)shmem_create,
    .size = size_in_bytes,
    .type = AMDXDNA_BO_SHARE,
  };
  ret = ioctl(fd, DRM_IOCTL_AMDXDNA_CREATE_BO, &create_bo);
  if (ret != 0) {
    perror("Failed to create BO");
    return -1;
  }

  struct amdxdna_drm_get_bo_info get_bo_info = {.handle = create_bo.handle};
  ret = ioctl(fd, DRM_IOCTL_AMDXDNA_GET_BO_INFO, &get_bo_info);
  if (ret != 0) {
    perror("Failed to get BO info");
    return -2;
  }

  *vaddr = (uint64_t)mmap(nullptr, size_in_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, get_bo_info.map_offset);
  if (vaddr == MAP_FAILED) {
    if (npu_debug_verbosity >= 1) printf("Failed to mmap() to allocate memory for BO\n");
    return -3;
  }
  *sram_vaddr = get_bo_info.xdna_addr;
  *handle = create_bo.handle;
  return 0;
}

// @brief: Flush the cachelines associated with the
// provided address, offset, and length
// @param: base(Input), base address to flush
// @param: offset(Input), offset of base address to flush
// @param: len(Input), length of buffer to flush
inline void FlushCpuCache(const void* base, size_t offset, size_t len) {
  static long cacheline_size = 0;

  if (!cacheline_size) {
    long sz = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    if (sz <= 0) return;
    cacheline_size = sz;
  }

  const char* cur = (const char*)base;
  cur += offset;
  uintptr_t lastline = (uintptr_t)(cur + len - 1) | (cacheline_size - 1);
  do {
    _mm_clflush((const void*)cur);
    cur += cacheline_size;
  } while (cur <= (const char*)lastline);
}

/*
  Create a BO_DEV and populate it with a PDI
*/

static int load_pdi(int fd, uint64_t *vaddr, uint64_t *sram_addr, __u32 *handle,
                    const char *path, uint64_t *pdi_size) {

  FILE *file = fopen(path, "r");
  if (file == NULL) {
    perror("Failed to open instructions file.");
    return -1;
  }

  fseek(file, 0L, SEEK_END);
  ssize_t file_size = ftell(file);
  fseek(file, 0L, SEEK_SET);

  if (npu_debug_verbosity >= 1) printf("Pdi file size: %ld\n", file_size);

  fclose(file);

  // Mmaping the file
  int pdi_fd = open(path, O_RDONLY);
  uint64_t *file_data =
      (uint64_t *)mmap(0, file_size, PROT_READ, MAP_PRIVATE, pdi_fd, 0);

  // Creating a BO_DEV bo to store the pdi file.
  int ret = create_dev_bo(fd, vaddr, sram_addr, handle, file_size);
  if (ret != 0) {
    perror("Failed to create pdi BO");
    return -1;
  }

  // copy the file into Bo dev
  uint64_t *bo = (uint64_t *)*vaddr;
  memcpy(bo, file_data, file_size);
  *pdi_size = (uint64_t)file_size;

  close(pdi_fd);
  return 0;
}

/*
  Create a BO DEV and populate it with instructions whose virtual address is
  passed to the driver via an HSA packet.
*/
static int load_instructions(int fd, uint64_t *vaddr, uint64_t *sram_addr,
                             __u32 *handle, const char *path, __u32 *num_inst, bool npu_is_bin) {

  void *inst_array = NULL;
  ssize_t file_size = 0;

  // Need to process the file differently if it is a binary vs a text file
  if (npu_is_bin) {
    // read dpu instructions into an array
    FILE *file = fopen(path, "r");
    if (file == NULL) {
      perror("Failed to open instructions file.");
      return -1;
    }

    fseek(file, 0L, SEEK_END);
    file_size = ftell(file);
    fseek(file, 0L, SEEK_SET);
    if (npu_debug_verbosity >= 1) printf("Instruction file size: %ld\n", file_size);
    fclose(file);

    // Map the instruction binary file as the memory in the instruction
    int inst_fd = open(path, O_RDONLY);
    inst_array = (uint64_t *)mmap(0, file_size, PROT_READ, MAP_PRIVATE, inst_fd, 0);

  }
  else {
    // read dpu instructions into an array
    FILE *file = fopen(path, "r");
    if (file == NULL) {
      perror("Failed to open instructions file.");
      return -1;
    }

    char *line = NULL;
    size_t len = 0;
    inst_array = malloc(sizeof(__u32) * MAX_NUM_INSTRUCTIONS);
    __u32 inst_counter = 0;
    while (getline(&line, &len, file) != -1) {
      ((__u32 *)inst_array)[inst_counter++] = strtoul(line, NULL, 16);
      if (inst_counter >= MAX_NUM_INSTRUCTIONS) {
        perror("Instruction array overflowed.");
        return -2;
      }
    }
    fclose(file);

    // Getting the size of the file so we can allocate an
    // appropriately sized DEV BO
    file_size = inst_counter * sizeof(__u32);
  }

  // Creating a BO_DEV bo to store the instruction.
  int ret =
      create_dev_bo(fd, vaddr, sram_addr, handle, file_size);
  if (ret != 0) {
    perror("Failed to create dpu BO");
    return -3;
  }

  // Getting the number of instructions in the sequence
  *num_inst = file_size / sizeof(__u32);

  memcpy((__u32 *)*vaddr, inst_array, file_size);
  return ret;
}

