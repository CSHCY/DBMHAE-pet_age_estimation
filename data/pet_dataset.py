import os  
import numpy as np  
import torch  
from torch.utils.data import Dataset, DataLoader  
import albumentations as A
from albumentations.pytorch import ToTensorV2  
from PIL import Image  

class PetAgeDataset(Dataset):  
    def __init__(self,   
                 metadata_path,   
                 image_dir,   
                 mode='train',  
                 image_size=384,  
                 transform=None):  
        """  
        Pet Age Dataset Loader for txt-based metadata  
        
        Args:  
        - metadata_path: Path to txt file with image metadata  
        - image_dir: Directory containing images  
        - mode: 'train' or 'val'  
        - image_size: Input image size  
        - transform: Optional custom transform  
        """  
        self.image_dir = image_dir  
        self.mode = mode  
        
        # Read metadata  
        with open(metadata_path, 'r') as f:  
            self.metadata = [line.strip().split('\t') for line in f]  
        
        # Convert to DataFrame-like structure  
        self.filenames = [item[0] for item in self.metadata]  
        self.ages = [int(item[1]) for item in self.metadata]  
        
        # Create two transform pipelines for different sizes
        if transform is None:
            self.transform_vit = self._get_transforms(336)  # ViT size
            self.transform_convnext = self._get_transforms(384)  # ConvNeXt size
        else:
            self.transform_vit = transform
            self.transform_convnext = transform
    
    def _get_transforms(self, image_size):  
        """  
        Create augmentation transforms  
        """  
        if self.mode == 'train':  
            return A.Compose([  
                A.RandomResizedCrop(  
                    height=image_size,   
                    width=image_size,   
                    scale=(0.08, 1.0),  
                    ratio=(3./4., 4./3.)  
                ),  
                A.HorizontalFlip(p=0.5),  
                A.Rotate(limit=15, p=0.5),  
                A.ColorJitter(  
                    brightness=0.2,   
                    contrast=0.2,   
                    saturation=0.2,   
                    hue=0.1,   
                    p=0.5  
                ),  
                A.Normalize(  
                    mean=[0.485, 0.456, 0.406],  
                    std=[0.229, 0.224, 0.225]  
                ),  
                ToTensorV2()  
            ])  
        else:  
            return A.Compose([  
                A.Resize(height=image_size, width=image_size),  
                A.Normalize(  
                    mean=[0.485, 0.456, 0.406],  
                    std=[0.229, 0.224, 0.225]  
                ),  
                ToTensorV2()  
            ])  
    
    def __len__(self):  
        return len(self.metadata)  
    
    def __getitem__(self, idx):  
        # Load image
        filename = self.filenames[idx]
        image_path = os.path.join(self.image_dir, filename)
        image = np.array(Image.open(image_path).convert('RGB'))
        
        # Apply transformations
        transformed_vit = self.transform_vit(image=image)
        transformed_convnext = self.transform_convnext(image=image)
        
        # Get age
        age = torch.tensor(self.ages[idx], dtype=torch.float32)
        
        return {
            'image_vit': transformed_vit['image'],
            'image_convnext': transformed_convnext['image']
        }, age

def create_dataloader(  
    metadata_path,   
    image_dir,   
    mode='train',   
    batch_size=256,   
    num_workers=4,  
    shuffle=None,
    image_size=384
):  
    """  
    Create DataLoader with smart defaults  
    
    Args:  
    - metadata_path: Path to metadata txt file  
    - image_dir: Directory with images  
    - mode: 'train' or 'val'  
    - batch_size: Batch size for loading  
    - num_workers: Parallel data loading workers  
    - shuffle: Optional shuffle override  
    
    Returns:  
    - PyTorch DataLoader  
    """  
    # Default shuffle based on mode  
    if shuffle is None:  
        shuffle = (mode == 'train')  
    
    dataset = PetAgeDataset(  
        metadata_path=metadata_path,   
        image_dir=image_dir,   
        mode=mode,
        image_size=image_size
    )  
    
    dataloader = DataLoader(  
        dataset,  
        batch_size=batch_size,  
        shuffle=shuffle,  
        num_workers=num_workers,  
        pin_memory=True,  
        drop_last=(mode == 'train')  
    )  
    
    return dataloader  

# Example usage  
def get_pet_age_dataloaders(  
    base_dir='pet_age_estimation/dataset/',  
    batch_size=256,  
    num_workers=8,
    image_size=384
):  
    """  
    Convenience function to create train and val dataloaders  
    """  
    train_loader = create_dataloader(  
        metadata_path=os.path.join(base_dir, 'annotations/train.txt'),  
        image_dir=os.path.join(base_dir, 'preprocessed_trainset_sam_point_guided'),  
        mode='train',  
        batch_size=batch_size,  
        num_workers=num_workers,
        image_size=image_size
    )  
    
    val_loader = create_dataloader(  
        metadata_path=os.path.join(base_dir, 'annotations/val.txt'),  
        image_dir=os.path.join(base_dir, 'preprocessed_valset_sam_point_guided'),  
        mode='val',  
        batch_size=batch_size,  
        num_workers=num_workers,
        image_size=image_size
    )  
    
    return train_loader, val_loader