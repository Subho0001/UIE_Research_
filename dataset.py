import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split # Import train_test_split

class UIEBDataset(Dataset):
    def __init__(self, raw_dir, ref_dir, filenames, transform=None):
        super().__init__()
        self.raw_dir = raw_dir
        self.ref_dir = ref_dir
        self.transform = transform if transform else transforms.ToTensor()
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        raw_path = os.path.join(self.raw_dir, filename)
        ref_path = os.path.join(self.ref_dir, filename)
        raw_img = Image.open(raw_path).convert('RGB')
        ref_img = Image.open(ref_path).convert('RGB')
        return self.transform(raw_img), self.transform(ref_img)

def create_dataloaders(raw_dir, ref_dir, train_transform=None, val_transform=None, val_split=0.1, batch_size=2, random_state=42):
    all_files = sorted([f for f in os.listdir(raw_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))])
    assert set(all_files) == set(os.listdir(ref_dir)), "Mismatch between raw and reference filenames"

    # --- THE FIX IS HERE ---
    # Use train_test_split for a reproducible random split
    train_files, val_files = train_test_split(
        all_files,
        test_size=val_split,
        random_state=random_state # This ensures the split is the same every time
    )
    # --- END OF FIX ---

    train_dataset = UIEBDataset(raw_dir, ref_dir, train_files, train_transform)
    val_dataset = UIEBDataset(raw_dir, ref_dir, val_files, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
