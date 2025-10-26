"""
Data loading and augmentation for CIFAR-10
"""
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms


def get_transforms(config, train=True):
    """
    Create data transforms for training or testing.
    
    Args:
        config: Configuration dictionary
        train: If True, return training transforms with augmentation
    
    Returns:
        transforms.Compose object
    """
    image_size = config['data']['image_size']
    aug_config = config['augmentation']
    
    if train:
        transform_list = [
            transforms.Resize((image_size, image_size)),
        ]
        
        # Add augmentations if enabled
        if aug_config.get('random_flip', True):
            transform_list.append(transforms.RandomHorizontalFlip())
        
        if aug_config.get('random_rotation', 0) > 0:
            transform_list.append(transforms.RandomRotation(degrees=aug_config['random_rotation']))
        
        if aug_config.get('random_crop_scale'):
            transform_list.append(
                transforms.RandomResizedCrop(
                    image_size, 
                    scale=tuple(aug_config['random_crop_scale'])
                )
            )
        
        if aug_config.get('color_jitter', False):
            transform_list.append(
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            )
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=aug_config['normalize_mean'],
                std=aug_config['normalize_std']
            )
        ])
    else:
        # Test transforms (no augmentation)
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=aug_config['normalize_mean'],
                std=aug_config['normalize_std']
            )
        ]
    
    return transforms.Compose(transform_list)


def get_dataloaders(config):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        train_loader, val_loader, test_loader
    """
    data_dir = config['data']['data_dir']
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']
    val_split = config['data']['val_split']
    
    # Get transforms
    train_transform = get_transforms(config, train=True)
    test_transform = get_transforms(config, train=False)
    
    # Load datasets
    print("Loading CIFAR-10 dataset...")
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=train_transform
    )
    
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=test_transform
    )
    
    # Split training into train and validation
    train_size = int((1 - val_split) * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = random_split(
        trainset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config['seed'])
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        valset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        testset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Training samples: {len(trainset)}")
    print(f"Validation samples: {len(valset)}")
    print(f"Test samples: {len(testset)}")
    
    return train_loader, val_loader, test_loader


def get_class_names():
    """Return CIFAR-10 class names."""
    return [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]


def denormalize(tensor, mean, std):
    """
    Denormalize a tensor image.
    
    Args:
        tensor: (C, H, W) or (B, C, H, W) tensor
        mean: normalization mean
        std: normalization std
    
    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    return tensor * std + mean


if __name__ == "__main__":
    # Test data loading
    import yaml
    import matplotlib.pyplot as plt
    
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    train_loader, val_loader, test_loader = get_dataloaders(config)
    
    # Visualize some samples
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Plot some images
    class_names = get_class_names()
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(images):
            # Denormalize
            img = denormalize(
                images[idx], 
                config['augmentation']['normalize_mean'],
                config['augmentation']['normalize_std']
            )
            img = torch.clamp(img, 0, 1)
            
            ax.imshow(img.permute(1, 2, 0))
            ax.set_title(class_names[labels[idx]])
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('./results/plots/sample_images.png', dpi=150, bbox_inches='tight')
    print("Sample images saved to results/plots/sample_images.png")