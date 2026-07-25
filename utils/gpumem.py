import torch

if torch.cuda.is_available():
    free_mem, total_mem = torch.cuda.mem_get_info()
    print(f"Free memory: {free_mem / 1e9:.2f} GB")
    print(f"Total memory: {total_mem / 1e9:.2f} GB")
else:
    print("CUDA is not available")