import torch
import torch.nn as nn
import os
import time
from torchvision import transforms
from tqdm import tqdm
from piq import psnr, ssim as piq_ssim
import pandas as pd

# --- Import your custom modules ---
from model import WaterMamba
from dataset import create_dataloaders

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def print_model_size(model, label="", filepath=None):
    """Helper function to print model size."""
    if filepath:
        size_mb = os.path.getsize(filepath) / 1024**2
        print(f"Model size ({label}) from file: {size_mb:.3f} MB")
        return size_mb
    
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024**2
    print(f"Model size ({label}) in memory: {size_mb:.3f} MB")
    return size_mb

def evaluate_model(model, data_loader, device, model_label=""):
    """
    Runs a full evaluation loop on a given model.
    Returns:
        avg_psnr (float): Average Peak Signal-to-Noise Ratio
        avg_ssim (float): Average Structural Similarity Index
        avg_time (float): Average inference time per image (in ms)
    """
    model.eval()
    model.to(device)
    
    total_psnr, total_ssim, total_time = 0, 0, 0
    num_images = 0

    print(f"--- Evaluating {model_label} model on {device} ---")
    with torch.no_grad():
        for raw, ref in tqdm(data_loader, desc=f"Evaluating {model_label}"):
            raw, ref = raw.to(device), ref.to(device)
            
            batch_size = raw.size(0)
            
            # --- Benchmark Time ---
            # Warmup
            if num_images == 0:
                _ = model(raw) 
            
            start_time = time.perf_counter()
            output = model(raw)
            end_time = time.perf_counter()
            
            total_time += (end_time - start_time)
            
            # Clamp output for metrics
            output_clamped = output.clamp(0.0, 1.0)
            
            # --- Calculate Metrics ---
            total_psnr += psnr(output_clamped, ref, data_range=1.0).item() * batch_size
            total_ssim += piq_ssim(output_clamped, ref, data_range=1.0).item() * batch_size
            
            num_images += batch_size

    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images
    # Average time per *image*, not per batch
    avg_time_ms = (total_time / num_images) * 1000 
    
    return avg_psnr, avg_ssim, avg_time_ms

# ==============================================================================
# MAIN SCRIPT
# ==============================================================================
if __name__ == '__main__':
    # --- Configuration ---
    # We will test both CPU and CUDA (if available)
    CPU_DEVICE = torch.device("cpu")
    CUDA_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Testing on: {CPU_DEVICE} and {CUDA_DEVICE}")
    
    # Set threads for fair CPU comparison
    torch.set_num_threads(1)
    
    base_dir = os.getcwd()
    image_resolution = [256, 256]
    batch_size = 1 # Use Batch Size 1 for fair inference speed comparison
    
    # --- Paths ---
    SAVE_DIR = os.path.join(base_dir, 'UIEB_VSS_UNet_MSFFN_RandomMix_Results')
    CHECKPOINT_PATH = os.path.join(SAVE_DIR, "best_model_psnr.pth") # Use your best FP32 model
    QUANTIZED_MODEL_PATH = os.path.join(SAVE_DIR, "water_mamba_int8.pth")
    
    raw_data_path = os.path.join(base_dir, 'dataset', 'raw-890')
    ref_data_path = os.path.join(base_dir, 'dataset', 'reference-890')

    if not os.path.exists(CHECKPOINT_PATH) or not os.path.exists(QUANTIZED_MODEL_PATH):
        print(f"Error: Model file not found.")
        print(f"Ensure '{CHECKPOINT_PATH}' and '{QUANTIZED_MODEL_PATH}' exist.")
        print("Run quantize.py first!")
        exit()

    # --- 1. Load Data ---
    val_transform = transforms.Compose([
        transforms.Resize(image_resolution),
        transforms.ToTensor()
    ])
    _, val_loader = create_dataloaders(
        raw_data_path, ref_data_path,
        train_transform=val_transform, val_transform=val_transform,
        batch_size=batch_size, random_state=42
    )

    # --- 2. Load FP32 Model ---
    model_fp32 = WaterMamba(
        inp_channels=3, out_channels=3,
        resolution=image_resolution, dim=24
    )
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=CPU_DEVICE)
    model_fp32.load_state_dict(checkpoint['model_state_dict'])
    model_fp32.eval()
    
    # --- 3. Load INT8 Model ---
    # Note: The INT8 model can ONLY run on CPU
    model_int8 = torch.jit.load(QUANTIZED_MODEL_PATH, map_location=CPU_DEVICE)
    model_int8.eval()

    # --- 4. Run Benchmarks ---
    results = []

    # --- FP32 (CPU) ---
    size_fp32_cpu = print_model_size(model_fp32, "FP32 (CPU)")
    psnr_fp32_cpu, ssim_fp32_cpu, time_fp32_cpu = evaluate_model(
        model_fp32, val_loader, CPU_DEVICE, "FP32 (CPU)"
    )
    results.append({
        "Model": "FP32 (Original)", "Device": "CPU", "Size (MB)": size_fp32_cpu,
        "PSNR": psnr_fp32_cpu, "SSIM": ssim_fp32_cpu, "Inference Time (ms)": time_fp32_cpu
    })
    
    # --- INT8 (CPU) ---
    size_int8_cpu = print_model_size(model_int8, "INT8 (CPU)", filepath=QUANTIZED_MODEL_PATH)
    psnr_int8_cpu, ssim_int8_cpu, time_int8_cpu = evaluate_model(
        model_int8, val_loader, CPU_DEVICE, "INT8 (CPU)"
    )
    results.append({
        "Model": "INT8 (Quantized)", "Device": "CPU", "Size (MB)": size_int8_cpu,
        "PSNR": psnr_int8_cpu, "SSIM": ssim_int8_cpu, "Inference Time (ms)": time_int8_cpu
    })
    
    # --- FP32 (CUDA) ---
    if torch.cuda.is_available():
        model_fp32_cuda = WaterMamba(
            inp_channels=3, out_channels=3,
            resolution=image_resolution, dim=24
        )
        checkpoint_cuda = torch.load(CHECKPOINT_PATH, map_location=CUDA_DEVICE)
        model_fp32_cuda.load_state_dict(checkpoint_cuda['model_state_dict'])
        model_fp32_cuda.eval()
        model_fp32_cuda.to(CUDA_DEVICE)
        
        size_fp32_cuda = print_model_size(model_fp32_cuda, "FP32 (CUDA)")
        psnr_fp32_cuda, ssim_fp32_cuda, time_fp32_cuda = evaluate_model(
            model_fp32_cuda, val_loader, CUDA_DEVICE, "FP32 (CUDA)"
        )
        results.append({
            "Model": "FP32 (Original)", "Device": "CUDA", "Size (MB)": size_fp32_cuda,
            "PSNR": psnr_fp32_cuda, "SSIM": ssim_fp32_cuda, "Inference Time (ms)": time_fp32_cuda
        })
    
    # --- 5. Print Results ---
    df = pd.DataFrame(results)
    df = df.set_index(["Model", "Device"])
    
    print("\n" + "="*50)
    print("           BENCHMARK RESULTS (Batch Size = 1)")
    print("="*50)
    print(df.to_markdown(floatfmt=".3f"))
    print("\n" + "="*50)
    
    # --- Print Comparison ---
    print("\n           COMPARISON: INT8 vs. FP32 (on CPU)")
    print("="*50)
    try:
        size_reduction = 1 - (size_int8_cpu / size_fp32_cpu)
        speedup = (time_fp32_cpu / time_int8_cpu)
        psnr_drop = psnr_fp32_cpu - psnr_int8_cpu
        ssim_drop = ssim_fp32_cpu - ssim_int8_cpu
        
        print(f"Size Reduction: {size_reduction*100:.2f}%")
        print(f"Inference Speedup: {speedup:.2f}x")
        print(f"PSNR Drop: {psnr_drop:.3f} dB")
        print(f"SSIM Drop: {ssim_drop:.4f}")
    except Exception as e:
        print(f"Could not calculate comparison: {e}")
