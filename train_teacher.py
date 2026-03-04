import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
from piq import psnr, ssim as piq_ssim
import matplotlib.pyplot as plt
from torchvision import transforms
from dataset import create_dataloaders
from model import VSS_UNet
from losses import SoftmaxWeightedLoss,FixedWeightedLoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from without_ssm_model.model_1 import EdgeWaterUNet

#from utils import visualize_enhancement_with_ref

# ==============================================================================
# VISUALIZATION HELPER (IMPROVED)
# ==============================================================================
def visualize_results(model, data_loader, device, epoch_num, save_dir='.', num_images=5):
    """
    Visualizes and saves a specified number of image comparisons from the data_loader.
    """
    model.eval()
    images_shown = 0
    with torch.no_grad():
        # Loop through the dataloader to get enough images
        for raw, ref in data_loader:
            raw, ref = raw.to(device), ref.to(device)
            output = model(raw)

            # Loop through the images in the current batch
            for i in range(raw.size(0)):
                if images_shown >= num_images:
                    break

                # Get the specific image from the batch
                raw_img = raw[i].cpu().permute(1, 2, 0).numpy()
                ref_img = ref[i].cpu().permute(1, 2, 0).numpy()
                output_img = output[i].clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()

                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1); plt.imshow(raw_img); plt.title("Input"); plt.axis('off')
                plt.subplot(1, 3, 2); plt.imshow(output_img); plt.title("Model Output"); plt.axis('off')
                plt.subplot(1, 3, 3); plt.imshow(ref_img); plt.title("Ground Truth"); plt.axis('off')
                plt.suptitle(f'Visual Results - Epoch {epoch_num}', fontsize=16)

                # Save each visualization with a unique name
                vis_path = os.path.join(save_dir, f"visualization_epoch_{epoch_num}_img_{images_shown+1}.png")
                plt.savefig(vis_path)
                plt.show()

                images_shown += 1
            if images_shown >= num_images:
                break


# ==============================================================================
# TRAINING LOOP (IMPROVED)
# ==============================================================================
# ==============================================================================
# TRAINING LOOP (IMPROVED)
# ==============================================================================
def train_vss_unet(
    model,
    criterion,
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
    model.to(device)

    if history is None:
        # --- NEW: Start a fresh training session ---
        print("ℹ️ Initializing new history and best-value trackers.")
        history = {'train_loss': [], 'val_loss': [], 'val_psnr': [], 'val_ssim': [], 'lr': []}
        best_val_loss = float('inf')
        best_psnr = 0.00
        best_ssim = 0.00
    else:
        # --- NEW: Resume from loaded history ---
        print("ℹ️ Initializing best-value trackers from loaded history.")
        # Check for empty lists, though they should be populated from a checkpoint
        best_val_loss = min(history['val_loss']) if history['val_loss'] else float('inf')
        best_psnr = max(history['val_psnr']) if history['val_psnr'] else 0.00
        best_ssim = max(history['val_ssim']) if history['val_ssim'] else 0.00
        print(f"▶️ Resuming with best values: Loss={best_val_loss:.4f}, PSNR={best_psnr:.4f}, SSIM={best_ssim:.4f}")
        
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")
        for raw, ref in pbar:
            raw, ref = raw.to(device), ref.to(device)
            optimizer.zero_grad()
            output = model(raw)
            loss = criterion(output, ref)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{scheduler.get_last_lr()[0]:.1e}"})

        avg_train_loss = total_train_loss / len(train_loader)

        history['train_loss'].append(avg_train_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        print(f"\nEpoch {epoch+1} Avg Training Loss: {avg_train_loss:.4f}")

        model.eval()
        total_val_loss, psnr_v, ssim_v = 0, 0, 0
        pbar_val = tqdm(val_loader, desc="Validation")
        with torch.no_grad():
            for raw, ref in pbar_val:
                raw, ref = raw.to(device), ref.to(device)
                output = model(raw)
                val_loss = criterion(output, ref)
                total_val_loss += val_loss.item()
                psnr_v += psnr(output.clamp(0,1), ref.clamp(0,1), data_range=1.0).item()
                ssim_v += piq_ssim(output.clamp(0,1), ref.clamp(0,1), data_range=1.0).item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_psnr = psnr_v / len(val_loader)
        avg_ssim = ssim_v / len(val_loader)
        scheduler.step(avg_val_loss)
        # --- NEW: STORE AND PRINT LOSS WEIGHTS ---
        # Access the logits from the loss function
        #current_logits = criterion.loss_logits
        # Apply softmax to get the weights that sum to 1
        #current_weights = torch.nn.functional.softmax(current_logits, dim=0).detach().cpu().numpy()
        # Store the weights
        #history['loss_weights'].append(current_weights)
        # Print the current weights
        #print(f"Loss Weights -> MSE: {current_weights[0]:.3f}, SSIM: {current_weights[1]:.3f}, Perceptual: {current_weights[2]:.3f}")


        print(f"Validation -> Loss: {avg_val_loss:.4f}, PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
        history['val_loss'].append(avg_val_loss)
        history['val_psnr'].append(avg_psnr)
        history['val_ssim'].append(avg_ssim)

        # --- IMPROVED: Robust Best Model Saving ---
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
        }

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint_data, os.path.join(save_dir, "best_model_loss.pth"))
            print(f"✅ New best model saved (by Loss): {best_val_loss:.4f}")

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(checkpoint_data, os.path.join(save_dir, "best_model_psnr.pth"))
            print(f"✅ New best model saved (by PSNR): {best_psnr:.4f}")

        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
            torch.save(checkpoint_data, os.path.join(save_dir, "best_model_ssim.pth"))
            print(f"✅ New best model saved (by SSIM): {best_ssim:.4f}")

        # --- IMPROVED: Plot and save checkpoints at a smarter interval ---
        if (epoch + 1) % 30 == 0 or (epoch + 1) == num_epochs:
            torch.save(checkpoint_data, os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth"))

            # --- FIXED: Plotting Logic ---
            epochs = range(1, epoch + 2) # Create an x-axis for the plot
            plt.figure(figsize=(18, 5))

            plt.subplot(1, 3, 1)
            plt.plot(epochs, history['train_loss'], 'o-', label='Train Loss')
            plt.plot(epochs, history['val_loss'], 'o-', label='Val Loss')
            plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss vs. Epochs'); plt.grid(True)

            plt.subplot(1, 3, 2)
            plt.plot(epochs, history['val_psnr'], 'o-', label='PSNR', color='green')
            plt.xlabel('Epoch'); plt.ylabel('PSNR (dB)'); plt.legend(); plt.title('PSNR vs. Epochs'); plt.grid(True)

            plt.subplot(1, 3, 3)
            plt.plot(epochs, history['val_ssim'], 'o-', label='SSIM', color='red')
            plt.xlabel('Epoch'); plt.ylabel('SSIM'); plt.legend(); plt.title('SSIM vs. Epochs'); plt.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"training_metrics_epoch_{epoch+1}.png"))
            plt.show()

            # Call the visualization function with 5 images
            visualize_results(model, val_loader, device, epoch + 1, save_dir, num_images=5)

    print(f"🎉 Training finished! Model and metrics saved in {save_dir}")

if __name__ == '__main__':
    # --- 1. Set up your training parameters and configurations here ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_EPOCHS = 750
    batch_size=2
    
    base_dir = os.getcwd()
    print(f"Project base directory: {base_dir}")
    
    
    
    SAVE_DIR = os.path.join(base_dir, 'UIEB_EdgeWaterUnet_Results')
    raw_data_path = os.path.join(base_dir, 'dataset', 'raw-890')
    ref_data_path = os.path.join(base_dir, 'dataset', 'reference-890')
    
    checkpoint_path = os.path.join(SAVE_DIR, "checkpoint_epoch_450.pth")
    
    # --- 2. Data Loading ---
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
        batch_size=batch_size,
        random_state=42 # Using a fixed state for reproducibility
    )

    # --- 3. Model, Loss, and Optimizer ---
    #model = VSS_UNet(in_channels=3, out_channels=3).to(device)
    model = EdgeWaterUNet()
    #my_initial_weights = [0.85, 0.125, 0.025]
    learning_rate = 1e-4
    criterion = FixedWeightedLoss(device=device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=3, eta_min=1e-6, verbose=True)
    
    start_epoch = 0
    history = None
    
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"✅ Resuming training from checkpoint: {checkpoint_path}")
        # Load the checkpoint dictionary
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Restore the model, optimizer, and scheduler states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Restore the epoch number and history
        # We start from the epoch *after* the one that was saved
        start_epoch = checkpoint['epoch']
        history = checkpoint['history']

        print(f"▶️ Model, optimizer, and scheduler states loaded. Starting from epoch {start_epoch + 1}.")
    else:
        print("ℹ️ No checkpoint found. Starting a new training session from scratch.")


    # --- 4. Start Training ---
    train_vss_unet(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=NUM_EPOCHS,
        start_epoch=start_epoch,
        save_dir=SAVE_DIR,
        history=history
    )
