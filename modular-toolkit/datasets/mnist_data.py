import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import os

def get_mnist_transforms():
    """Defines the standard transformations for MNIST."""
    # MNIST images are 28x28 grayscale.
    # Normalization values are standard for MNIST.
    transform = transforms.Compose([
        transforms.ToTensor(), # Converts image to PyTorch Tensor (C x H x W) and scales pixels to [0.0, 1.0]
        transforms.Normalize((0.1307,), (0.3081,)) # Mean and Std Dev for MNIST
    ])
    return transform

def load_mnist_datasets(data_path='./data', train_fraction=0.8):
    """Loads the MNIST training and testing datasets."""
    transform = get_mnist_transforms()
    
    # Create data directory if it doesn't exist
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Download and load the full training dataset
    full_train_dataset = torchvision.datasets.MNIST(
        root=data_path,
        train=True,
        download=True,
        transform=transform
    )

    # Download and load the test dataset
    test_dataset = torchvision.datasets.MNIST(
        root=data_path,
        train=False,
        download=True,
        transform=transform
    )

    # Split the full training dataset into training and validation sets
    num_train = len(full_train_dataset)
    indices = list(range(num_train))
    split = int(train_fraction * num_train)
    
    # Use fixed random seed for reproducibility if desired
    # torch.manual_seed(42) 
    torch.Generator().manual_seed(42) # Use a generator for shuffling
    shuffled_indices = torch.randperm(num_train, generator=torch.Generator().manual_seed(42)).tolist()

    train_indices = shuffled_indices[:split]
    valid_indices = shuffled_indices[split:]

    train_subset = Subset(full_train_dataset, train_indices)
    valid_subset = Subset(full_train_dataset, valid_indices)

    print(f"MNIST datasets loaded:")
    print(f"  Full training size: {num_train}")
    print(f"  Training subset size: {len(train_subset)}")
    print(f"  Validation subset size: {len(valid_subset)}")
    print(f"  Test set size: {len(test_dataset)}")
    
    return train_subset, valid_subset, test_dataset


def create_mnist_dataloaders(batch_size=64, data_path='./data', train_fraction=0.8, num_workers=0):
    """Creates DataLoaders for MNIST train, validation, and test sets."""
    
    train_dataset, valid_dataset, test_dataset = load_mnist_datasets(data_path, train_fraction)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False, # No need to shuffle validation data
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False, # No need to shuffle test data
        num_workers=num_workers
    )

    return train_loader, valid_loader, test_loader

if __name__ == '__main__':
    # Example usage:
    print("Creating MNIST DataLoaders...")
    train_loader, valid_loader, test_loader = create_mnist_dataloaders(batch_size=128, train_fraction=0.9)
    
    print("\nChecking one batch from train_loader:")
    # Get one batch of training data
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    
    # Print statistics
    print(f"  Images batch shape: {images.shape}") # Should be [batch_size, 1, 28, 28]
    print(f"  Labels batch shape: {labels.shape}") # Should be [batch_size]
    print(f"  First image min/max: {torch.min(images[0])}, {torch.max(images[0])}")
    print(f"  First 10 labels: {labels[:10]}")

    print("\nMNIST data loading setup complete.")