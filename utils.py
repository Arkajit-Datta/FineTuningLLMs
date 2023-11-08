from huggingface_hub import login
import torch

hf_token = "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"

def login_huggingface():
    login(token = hf_token)

def check_cuda_available_and_assign_device(accelerator):
    if torch.cuda.is_available(accelerator):
        print("You have a GPU available! Setting `DEVICE=\"cuda\"`")
        return accelerator.device
    else:
        print("Cuda is unavailable! Setting `DEVICE=\"auto\"`")
        return "auto"

def clean_objects_and_empty_gpu_cache(arr: list, clear_cache: bool = True):
    """
    Use this function when you need to delete the objects, free their memory
    and also delete the cuda cache
    """
    for obj in arr:
        print(f"Deleting {obj}")
        del obj
    if clear_cache:
        torch.cuda.empty_cache()
        print("="*80)
        print("Cleared Cuda Cache")