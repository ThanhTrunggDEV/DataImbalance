import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

class KneeDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (string): Directory with all the images. Example: './data/train'
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Usually datasets like this have train/val/test folders
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # The severity levels usually span 0 to 4
        if os.path.exists(self.root_dir):
            self.classes = sorted(os.listdir(self.root_dir))
            for label, cls in enumerate(self.classes):
                cls_dir = os.path.join(self.root_dir, cls)
                if os.path.isdir(cls_dir):
                    for img_name in os.listdir(cls_dir):
                        if img_name.endswith(('.png', '.jpg', '.jpeg')):
                            self.image_paths.append(os.path.join(cls_dir, img_name))
                            self.labels.append(label)
        else:
            print(f"Warning: Directory {self.root_dir} not found.")
            
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
        
    def get_class_weights(self):
        """
        Calculate class weights to handle severe imbalance.
        """
        class_counts = np.bincount(self.labels)
        # Handle zero division if a class has 0 samples
        class_weights = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
        weights = class_weights[self.labels]
        return weights, class_weights

def get_dataloaders(data_dir, batch_size=16):
    # EfficientNet-B5 default input size is 456x456
    img_size = 456
    
    # Data Augmentation to help with minority class representation
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = KneeDataset(data_dir, split="train", transform=train_transform)
    val_dataset = KneeDataset(data_dir, split="val", transform=val_test_transform)
    test_dataset = KneeDataset(data_dir, split="test", transform=val_test_transform)
    
    # ---------------------------------------------------------
    # MODERN IMBALANCE HANDLING: WeightedRandomSampler
    # This ensures each batch has a balanced representation of classes
    # ---------------------------------------------------------
    sample_weights, raw_class_weights = train_dataset.get_class_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, raw_class_weights