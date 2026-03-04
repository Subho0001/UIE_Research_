import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from tqdm import tqdm
import os
from piq import psnr, ssim as piq_ssim
import matplotlib.pyplot as plt
from torchvision import transforms
import glob

# --- Custom Imports ---
# Assuming these modules exist in your environment:
from dataset import create_dataloaders
from model import VSS_UNet
from losses import SoftmaxWeightedLoss, FixedWeightedLoss
from without_ssm_model.model_1 import EdgeWaterUNet

# ==============================================================================
# VISUALIZATION HELPER
# ==============================================================================
# def visualize_results(model, teacher_model, data_loader, device, epoch_num, save_dir='.', num_images=3):

#     model.eval()
#     if teacher_model: teacher_model.eval()
    
#     images_shown = 0
#     with torch.no_grad():
#         for raw, ref in data_loader:
#             raw, ref = raw.to(device), ref.to(device)
#             student_out = model(raw)
#             teacher_out = teacher_model(raw) if teacher_model else None

#             for i in range(raw.size(0)):
#                 if images_shown >= num_images: break

#                 raw_img = raw[i].cpu().permute(1, 2, 0).numpy()
#                 ref_img = ref[i].cpu().permute(1, 2, 0).numpy()
#                 stud_img = student_out[i].clamp(0,1).cpu().permute(1, 2, 0).numpy()
#                 teach_img = teacher_out[i].clamp(0,1).cpu().permute(1, 2, 0).numpy() if teacher_out is not None else None

#                 cols = 4 if teach_img is not None else 3
#                 plt.figure(figsize=(20, 5))
                
#                 plt.subplot(1, cols, 1); plt.imshow(raw_img); plt.title("Input"); plt.axis('off')
#                 plt.subplot(1, cols, 2); plt.imshow(stud_img); plt.title("Student (Edge)"); plt.axis('off')
#                 if teach_img is not None:
#                     plt.subplot(1, cols, 3); plt.imshow(teach_img); plt.title("Teacher (VSS)"); plt.axis('off')
#                 plt.subplot(1, cols, cols); plt.imshow(ref_img); plt.title("Ground Truth"); plt.axis('off')
                
#                 plt.suptitle(f'KD Results - Epoch {epoch_num}', fontsize=16)
#                 vis_path = os.path.join(save_dir, f"vis_epoch_{epoch_num}_img_{images_shown+1}.png")
#                 plt.savefig(vis_path)
#                 plt.close()
#                 images_shown += 1
#             if images_shown >= num_images: break

def visualize_results(model, teacher_psnr, teacher_ssim, data_loader, device, epoch_num, save_dir='.', num_images=3):
    model.eval()
    if teacher_psnr: teacher_psnr.eval()
    if teacher_ssim: teacher_ssim.eval()
    
    images_shown = 0
    with torch.no_grad():
        for raw, ref in data_loader:
            raw, ref = raw.to(device), ref.to(device)
            student_out = model(raw)
            teach_p_out = teacher_psnr(raw) if teacher_psnr else None
            teach_s_out = teacher_ssim(raw) if teacher_ssim else None

            for i in range(raw.size(0)):
                if images_shown >= num_images: break

                raw_img = raw[i].cpu().permute(1, 2, 0).numpy()
                ref_img = ref[i].cpu().permute(1, 2, 0).numpy()
                stud_img = student_out[i].clamp(0,1).cpu().permute(1, 2, 0).numpy()
                tp_img = teach_p_out[i].clamp(0,1).cpu().permute(1, 2, 0).numpy() if teach_p_out is not None else None
                ts_img = teach_s_out[i].clamp(0,1).cpu().permute(1, 2, 0).numpy() if teach_s_out is not None else None

                # Calculate columns dynamically based on loaded teachers
                cols = 2  # Input + Student
                if tp_img is not None: cols += 1
                if ts_img is not None: cols += 1
                cols += 1 # Ground Truth

                plt.figure(figsize=(25, 5))
                idx = 1
                plt.subplot(1, cols, idx); plt.imshow(raw_img); plt.title("Input"); plt.axis('off'); idx+=1
                plt.subplot(1, cols, idx); plt.imshow(stud_img); plt.title("Student"); plt.axis('off'); idx+=1
                
                if tp_img is not None:
                    plt.subplot(1, cols, idx); plt.imshow(tp_img); plt.title("Teacher (PSNR)"); plt.axis('off'); idx+=1
                if ts_img is not None:
                    plt.subplot(1, cols, idx); plt.imshow(ts_img); plt.title("Teacher (SSIM)"); plt.axis('off'); idx+=1
                    
                plt.subplot(1, cols, idx); plt.imshow(ref_img); plt.title("Ground Truth"); plt.axis('off')
                
                plt.suptitle(f'Dual-Teacher KD Results - Epoch {epoch_num}', fontsize=16)
                vis_path = os.path.join(save_dir, f"vis_epoch_{epoch_num}_img_{images_shown+1}.png")
                plt.savefig(vis_path)
                plt.close()
                images_shown += 1
            if images_shown >= num_images: break

# ==============================================================================
# TEACHER LOADER
# ==============================================================================
def load_teacher_model(model_class, checkpoint_path, device):
    """Loads the teacher model and freezes it (Eval mode, No Grads)."""
    print(f"👨‍🏫 Loading Teacher Model from: {checkpoint_path}")
    teacher = model_class().to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Handle state dict keys if they were saved with 'module.' prefix or inside a dict
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    teacher.load_state_dict(state_dict)
    
    # FREEZE TEACHER
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    
    print("❄️ Teacher model loaded and frozen (Weights will not update).")
    return teacher

# ==============================================================================
# TRAINING LOOP (KD ENABLED)
# ==============================================================================
# def train_distillation(
#     student_model,
#     teacher_model,      # New Argument
#     criterion_gt,       # Loss vs Ground Truth
#     criterion_kd,       # Loss vs Teacher (Distillation)
#     distill_weight,     # Alpha: Weight for KD Loss
#     train_loader,
#     val_loader,
#     optimizer,
#     scheduler,
#     device,
#     num_epochs,
#     save_dir,
#     start_epoch=0,
#     history=None
# ):
#     os.makedirs(save_dir, exist_ok=True)
#     student_model.to(device)
#     if teacher_model: teacher_model.to(device)

#     if history is None:
#         history = {'train_loss': [], 'val_loss': [], 'val_psnr': [], 'val_ssim': [], 'lr': []}
#         best_psnr = 0.0
#     else:
#         best_psnr = max(history['val_psnr']) if history['val_psnr'] else 0.0

#     print(f"🚀 Starting Knowledge Distillation. Lambda (KD Weight): {distill_weight}")

#     for epoch in range(start_epoch, num_epochs):
#         student_model.train()
#         if teacher_model: teacher_model.eval() # Ensure teacher stays in eval
        
#         total_loss = 0
#         total_gt_loss = 0
#         total_kd_loss = 0
        
#         pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Distilling]")
        
#         for raw, ref in pbar:
#             raw, ref = raw.to(device), ref.to(device)
            
#             optimizer.zero_grad()
            
#             # 1. Student Forward
#             student_out = student_model(raw)
            
#             # 2. Teacher Forward (No Grad)
#             with torch.no_grad():
#                 teacher_out = teacher_model(raw)
            
#             # 3. Calculate Losses
#             loss_gt = criterion_gt(student_out, ref)
#             loss_kd = criterion_kd(student_out, teacher_out) # Minimize distance between Student & Teacher
            
#             # 4. Combine
#             loss = 0.5 * loss_gt + (distill_weight * loss_kd)
            
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()
#             total_gt_loss += loss_gt.item()
#             total_kd_loss += loss_kd.item()
            
#             pbar.set_postfix({
#                 "T_Loss": f"{loss.item():.4f}", 
#                 "GT": f"{loss_gt.item():.4f}", 
#                 "KD": f"{loss_kd.item():.4f}"
#             })

#         avg_loss = total_loss / len(train_loader)
#         history['train_loss'].append(avg_loss)
#         print(f"\nEpoch {epoch+1} Avg Loss: {avg_loss:.4f} (GT: {total_gt_loss/len(train_loader):.4f}, KD: {total_kd_loss/len(train_loader):.4f})")

#         # --- Validation (Same as before) ---
#         student_model.eval()
#         total_val_loss, psnr_v, ssim_v = 0, 0, 0
        
#         with torch.no_grad():
#             for raw, ref in tqdm(val_loader, desc="Validation"):
#                 raw, ref = raw.to(device), ref.to(device)
#                 # 1. Student Forward
#                 output = student_model(raw)
                
#                 # 2. Teacher Forward (No Grad)
#                 with torch.no_grad():
#                     teacher_out = teacher_model(raw)
                
#                 # 3. Calculate Losses
#                 val_loss_gt = criterion_gt(output, ref)
#                 val_loss_kd = criterion_kd(output, teacher_out) # Minimize distance between Student & Teacher
                
#                 # 4. Combine
#                 val_loss = val_loss_gt + (distill_weight * val_loss_kd)
#                 #val_loss = criterion_gt(output, ref) # Validate against GT only
                
#                 total_val_loss += val_loss.item()
#                 psnr_v += psnr(output.clamp(0,1), ref.clamp(0,1), data_range=1.0).item()
#                 ssim_v += piq_ssim(output.clamp(0,1), ref.clamp(0,1), data_range=1.0).item()

#         avg_val_loss = total_val_loss / len(val_loader)
#         avg_psnr = psnr_v / len(val_loader)
#         avg_ssim = ssim_v / len(val_loader)
        
#         # Check if scheduler is CosineAnnealingWarmRestarts or ReduceLROnPlateau
#         if isinstance(scheduler, ReduceLROnPlateau):
#              scheduler.step(avg_val_loss)
#         elif hasattr(scheduler, 'step'):
#              scheduler.step()
#         # For CosineAnnealingWarmRestarts, step() is typically called per batch or per epoch without a metric,
#         # but the original code calls it with a metric, which is characteristic of ReduceLROnPlateau.
#         # Sticking to the original logic:
#         # scheduler.step(avg_val_loss)

#         print(f"Val -> Loss: {avg_val_loss:.4f}, PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
#         history['val_loss'].append(avg_val_loss)
#         history['val_psnr'].append(avg_psnr)
#         history['val_ssim'].append(avg_ssim)

#         # --- Checkpointing ---
#         checkpoint_data = {
#             'epoch': epoch + 1,
#             'model_state_dict': student_model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'scheduler_state_dict': scheduler.state_dict(),
#             'history': history,
#         }

#         if avg_psnr > best_psnr:
#             best_psnr = avg_psnr
#             torch.save(checkpoint_data, os.path.join(save_dir, "best_student_model.pth"))
#             print(f"✅ Best Student Saved: {best_psnr:.4f}")

#         if (epoch + 1) % 30 == 0:
#             torch.save(checkpoint_data, os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth"))
#             visualize_results(student_model, teacher_model, val_loader, device, epoch + 1, save_dir)

def train_distillation(
    student_model,
    teacher_psnr,           # Teacher 1
    teacher_ssim,           # Teacher 2
    criterion_gt,       
    criterion_kd,       
    distill_weight,     
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    num_epochs,
    save_dir,
    start_epoch=0,
    history=None
):
    os.makedirs(save_dir, exist_ok=True)
    student_model.to(device)
    if teacher_psnr: teacher_psnr.to(device)
    if teacher_ssim: teacher_ssim.to(device)

    if history is None:
        history = {'train_loss': [], 'val_loss': [], 'val_psnr': [], 'val_ssim': [], 'lr': []}
        best_psnr = 0.0
    else:
        best_psnr = max(history['val_psnr']) if history['val_psnr'] else 0.0

    print(f"🚀 Starting Dual-Teacher Distillation. Lambda: {distill_weight}")

    for epoch in range(start_epoch, num_epochs):
        student_model.train()
        # Ensure both teachers are in eval mode
        if teacher_psnr: teacher_psnr.eval()
        if teacher_ssim: teacher_ssim.eval()
        
        total_loss = 0
        total_gt_loss = 0
        total_kd_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Distilling]")
        
        for raw, ref in pbar:
            raw, ref = raw.to(device), ref.to(device)
            
            optimizer.zero_grad()
            
            # 1. Student Forward
            student_out = student_model(raw)
            
            # 2. Teachers Forward (No Grad)
            with torch.no_grad():
                out_psnr = teacher_psnr(raw)
                out_ssim = teacher_ssim(raw)
            
            # 3. Calculate Losses
            loss_gt = criterion_gt(student_out, ref)
            
            # KD Loss: Average distance to both teachers
            loss_kd_p = criterion_kd(student_out, out_psnr)
            loss_kd_s = criterion_kd(student_out, out_ssim)
            
            # Combine KD losses (Equal contribution: 0.5 each)
            loss_kd = 0.5 * loss_kd_p + 0.5 * loss_kd_s
            
            # 4. Total Loss
            # Formula: 0.5 * GT_Loss + Alpha * (0.5 * TeachP + 0.5 * TeachS)
            loss = 0.5 * loss_gt + (distill_weight * loss_kd_p)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_gt_loss += loss_gt.item()
            total_kd_loss += loss_kd.item()
            
            pbar.set_postfix({
                "T_Loss": f"{loss.item():.4f}", 
                "GT": f"{loss_gt.item():.4f}", 
                "KD": f"{loss_kd.item():.4f}"
            })

        avg_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        print(f"\nEpoch {epoch+1} Avg Loss: {avg_loss:.4f}")

        # --- Validation ---
        student_model.eval()
        total_val_loss, psnr_v, ssim_v = 0, 0, 0
        
        with torch.no_grad():
            for raw, ref in tqdm(val_loader, desc="Validation"):
                raw, ref = raw.to(device), ref.to(device)
                
                output = student_model(raw)
                
                # Get teacher outputs for validation loss calculation
                out_psnr = teacher_psnr(raw)
                out_ssim = teacher_ssim(raw)
                
                val_loss_gt = criterion_gt(output, ref)
                #val_loss_kd = 0.5 * criterion_kd(output, out_psnr) + 0.5 * criterion_kd(output, out_ssim)
                val_loss_kd = criterion_kd(output, out_psnr)
                val_loss = val_loss_gt + (distill_weight * val_loss_kd)
                
                total_val_loss += val_loss.item()
                psnr_v += psnr(output.clamp(0,1), ref.clamp(0,1), data_range=1.0).item()
                ssim_v += piq_ssim(output.clamp(0,1), ref.clamp(0,1), data_range=1.0).item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_psnr = psnr_v / len(val_loader)
        avg_ssim = ssim_v / len(val_loader)
        
        if isinstance(scheduler, ReduceLROnPlateau):
             scheduler.step(avg_val_loss)
        elif hasattr(scheduler, 'step'):
             scheduler.step()

        print(f"Val -> Loss: {avg_val_loss:.4f}, PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
        history['val_loss'].append(avg_val_loss)
        history['val_psnr'].append(avg_psnr)
        history['val_ssim'].append(avg_ssim)

        # --- Checkpointing ---
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': student_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
        }

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(checkpoint_data, os.path.join(save_dir, "best_student_model.pth"))
            print(f"✅ Best Student Saved: {best_psnr:.4f}")

        if (epoch + 1) % 30 == 0:
            torch.save(checkpoint_data, os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth"))
            # 2. Generate Plots
            epochs_range = range(1, epoch + 2)
            plt.figure(figsize=(18, 5))
            
            # Subplot 1: Losses
            plt.subplot(1, 3, 1)
            plt.plot(epochs_range, history['train_loss'], label='Train Loss')
            plt.plot(epochs_range, history['val_loss'], label='Val Loss')
            plt.title('Loss vs Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # Subplot 2: PSNR
            plt.subplot(1, 3, 2)
            plt.plot(epochs_range, history['val_psnr'], label='Val PSNR', color='green')
            plt.title('PSNR vs Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('PSNR (dB)')
            plt.legend()
            plt.grid(True)
            
            # Subplot 3: SSIM
            plt.subplot(1, 3, 3)
            plt.plot(epochs_range, history['val_ssim'], label='Val SSIM', color='red')
            plt.title('SSIM vs Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('SSIM')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            
            # Save the plot
            plot_path = os.path.join(save_dir, f"metrics_epoch_{epoch+1}.png")
            plt.savefig(plot_path)
            plt.close() # Close to free memory
            
            print(f"📈 Metrics plot saved to {plot_path}")
            visualize_results(student_model, teacher_psnr, teacher_ssim, val_loader, device, epoch + 1, save_dir)

# ==============================================================================
# MAIN
# ==============================================================================
# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # --- CONFIGURATION ---
#     NUM_EPOCHS = 750
#     BATCH_SIZE = 2
#     LR = 1e-4
#     DISTILL_WEIGHT = 2.5 
    
#     # Paths
#     base_dir = os.getcwd()
#     SAVE_DIR = os.path.join(base_dir, 'Student_MoreTowardsTeacher_EdgeWaterNet_Results')
    
#     # Dataset Paths
#     raw_data_path = os.path.join(base_dir, 'dataset', 'raw-890')
#     ref_data_path = os.path.join(base_dir, 'dataset', 'reference-890')
    
#     # Teacher Checkpoint
#     teacher_ckpt_path = os.path.join(base_dir, 'UIEB_VSS_UNet_MSFFN_RandomMix_Results', 'best_model_psnr.pth')

#     # --- 1. Data ---
#     # NOTE: create_dataloaders is a custom import and must be defined/available
    
#     # Dummy placeholder for transforms (assuming standard PyTorch imports are resolved)
#     train_transform = transforms.Compose([
#         transforms.Resize((256,256)),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#         transforms.ToTensor()
#     ])

#     val_transform = transforms.Compose([
#         transforms.Resize((256,256)),
#         transforms.ToTensor()
#     ])
    
#     train_loader, val_loader = create_dataloaders(
#         raw_data_path,
#         ref_data_path,
#         train_transform=train_transform,
#         val_transform=val_transform,
#         random_state=42
#     )

#     # --- 2. Load Models ---
#     # A. Teacher
#     teacher_model = load_teacher_model(VSS_UNet, teacher_ckpt_path, device)
    
#     # B. Student
#     student_model = EdgeWaterUNet().to(device)

#     # --- 3. Losses ---
#     criterion_gt = FixedWeightedLoss(device=device) 
#     criterion_kd = FixedWeightedLoss(device=device)                   

#     # --- 4. Optimizer & Scheduler ---
#     optimizer = optim.AdamW(student_model.parameters(), lr=LR, betas=(0.9, 0.999))
#     scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=3, eta_min=1e-6)

#     # --- 5. CHECKPOINT RESUME LOGIC ---
#     start_epoch = 0
#     history = None

#     # Find all checkpoint files
#     checkpoint_files = glob.glob(os.path.join(SAVE_DIR, "checkpoint_epoch_*.pth"))

#     if checkpoint_files:
#         # Sort files by epoch number (extracted from filename) to find the latest
#         latest_ckpt_path = max(checkpoint_files, key=lambda x: int(x.split('_epoch_')[-1].split('.pth')[0]))
        
#         print(f"🔄 Found checkpoints! Resuming from: {latest_ckpt_path}")
#         checkpoint = torch.load(latest_ckpt_path, map_location=device)

#         # Restore states
#         student_model.load_state_dict(checkpoint['model_state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
#         # Restore history and epoch
#         history = checkpoint.get('history', None)
#         start_epoch = checkpoint['epoch']
        
#         print(f"✅ Successfully restored state. Resuming from Epoch {start_epoch + 1}")
#     else:
#         print("🆕 No previous checkpoints found. Starting training from scratch.")

#     # --- 6. Run Training ---
#     train_distillation(
#         student_model=student_model,
#         teacher_model=teacher_model,
#         criterion_gt=criterion_gt,
#         criterion_kd=criterion_kd,
#         distill_weight=DISTILL_WEIGHT,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         optimizer=optimizer,
#         scheduler=scheduler,
#         device=device,
#         num_epochs=NUM_EPOCHS,
#         save_dir=SAVE_DIR,
#         start_epoch=start_epoch, # Pass the restored epoch
#         history=history          # Pass the restored history
#     )

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- CONFIGURATION ---
    NUM_EPOCHS = 750
    BATCH_SIZE = 2
    LR = 1e-4
    DISTILL_WEIGHT = 2.5 
    
    # Paths
    base_dir = os.getcwd()
    SAVE_DIR = os.path.join(base_dir, 'Student_DualTeacher_EdgeWaterNet_Results')
    
    # Dataset Paths
    raw_data_path = os.path.join(base_dir, 'dataset', 'raw-890')
    ref_data_path = os.path.join(base_dir, 'dataset', 'reference-890')
    
    # --- TEACHER PATHS ---
    teacher_base = os.path.join(base_dir, 'UIEB_VSS_UNet_MSFFN_RandomMix_Results')
    path_psnr = os.path.join(teacher_base, 'best_model_psnr.pth')
    path_ssim = os.path.join(teacher_base, 'best_model_ssim.pth')

    # --- 1. Data ---
    train_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])
    
    train_loader, val_loader = create_dataloaders(
        raw_data_path,
        ref_data_path,
        train_transform=train_transform,
        val_transform=val_transform,
        random_state=42
    )

    # --- 2. Load Models ---
    print("--- Loading Teacher Models ---")
    # Load PSNR Teacher
    teacher_psnr = load_teacher_model(VSS_UNet, path_psnr, device)
    # Load SSIM Teacher
    teacher_ssim = load_teacher_model(VSS_UNet, path_ssim, device)
    
    # Student
    student_model = EdgeWaterUNet().to(device)

    # --- 3. Losses ---
    criterion_gt = FixedWeightedLoss(device=device) 
    criterion_kd = FixedWeightedLoss(device=device)

    # criterion_gt = nn.L1Loss()
    # criterion_kd = nn.L1Loss()                   

    # --- 4. Optimizer & Scheduler ---
    optimizer = optim.AdamW(student_model.parameters(), lr=LR, betas=(0.9, 0.999))
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=3, eta_min=1e-6)

    # --- 5. CHECKPOINT RESUME LOGIC ---
    start_epoch = 0
    history = None

    checkpoint_files = glob.glob(os.path.join(SAVE_DIR, "checkpoint_epoch_*.pth"))

    if checkpoint_files:
        latest_ckpt_path = max(checkpoint_files, key=lambda x: int(x.split('_epoch_')[-1].split('.pth')[0]))
        
        print(f"🔄 Found checkpoints! Resuming from: {latest_ckpt_path}")
        checkpoint = torch.load(latest_ckpt_path, map_location=device)

        student_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        history = checkpoint.get('history', None)
        start_epoch = checkpoint['epoch']
        
        print(f"✅ Successfully restored state. Resuming from Epoch {start_epoch + 1}")
    else:
        print("🆕 No previous checkpoints found. Starting training from scratch.")

    # --- 6. Run Training ---
    train_distillation(
        student_model=student_model,
        teacher_psnr=teacher_psnr,    # Pass Teacher 1
        teacher_ssim=teacher_ssim,    # Pass Teacher 2
        criterion_gt=criterion_gt,
        criterion_kd=criterion_kd,
        distill_weight=DISTILL_WEIGHT,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=NUM_EPOCHS,
        save_dir=SAVE_DIR,
        start_epoch=start_epoch,
        history=history
    )