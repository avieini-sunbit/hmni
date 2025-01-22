import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, Dict, Any, Iterator
import pandas as pd

from .preprocess_torch import VocabularyProcessor

class InputHelper:
    """Helper class for data processing that matches the original implementation."""
    
    def batch_iter(self, data: List[Any], batch_size: int, num_epochs: int, shuffle: bool = True) -> Iterator[List[Any]]:
        """Generate batches of data.
        
        Args:
            data: List of data points
            batch_size: Size of each batch
            num_epochs: Number of epochs to generate
            shuffle: Whether to shuffle data at each epoch
            
        Yields:
            Batches of data
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int(len(data) / batch_size) + 1
        
        for epoch in range(num_epochs):
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
                
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]
    
    def get_datasets(self, 
                    X_train: pd.DataFrame,
                    y_train: np.ndarray,
                    max_document_length: int,
                    percent_dev: int,
                    batch_size: int) -> Tuple[Tuple, Tuple, VocabularyProcessor, int]:
        """Prepare training and development datasets.
        
        Args:
            X_train: Training data DataFrame with name pairs
            y_train: Training labels
            max_document_length: Maximum sequence length
            percent_dev: Percentage of data to use for validation
            batch_size: Batch size
            
        Returns:
            Tuple of (train_set, dev_set, vocab_processor, num_batches)
        """
        # Convert names to lowercase
        x1_text = np.asarray(X_train.iloc[:, 0].str.lower())
        x2_text = np.asarray(X_train.iloc[:, 1].str.lower())
        y = np.asarray(y_train)
        
        # Build vocabulary
        vocab_processor = VocabularyProcessor(max_document_length, min_frequency=0)
        vocab_processor.fit(np.concatenate((x2_text, x1_text), axis=0))
        
        # Transform text to sequences
        x1 = np.asarray(list(vocab_processor.transform(x1_text)))
        x2 = np.asarray(list(vocab_processor.transform(x2_text)))
        
        # Shuffle data
        np.random.seed(131)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x1_shuffled = x1[shuffle_indices]
        x2_shuffled = x2[shuffle_indices]
        y_shuffled = y[shuffle_indices]
        
        # Split into train and dev
        dev_idx = -1 * len(y_shuffled) * percent_dev // 100
        x1_train, x1_dev = x1_shuffled[:dev_idx], x1_shuffled[dev_idx:]
        x2_train, x2_dev = x2_shuffled[:dev_idx], x2_shuffled[dev_idx:]
        y_train, y_dev = y_shuffled[:dev_idx], y_shuffled[dev_idx:]
        
        # Calculate number of batches
        sum_no_of_batches = len(y_train) // batch_size
        
        return (x1_train, x2_train, y_train), (x1_dev, x2_dev, y_dev), vocab_processor, sum_no_of_batches
    
    def get_test_dataset(self,
                        X_test: pd.DataFrame,
                        y_test: np.ndarray,
                        vocab_processor: VocabularyProcessor,
                        max_document_length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare test dataset.
        
        Args:
            X_test: Test data DataFrame with name pairs
            y_test: Test labels
            vocab_processor: Trained vocabulary processor
            max_document_length: Maximum sequence length
            
        Returns:
            Tuple of (x1, x2, y) arrays
        """
        x1_text = np.asarray(X_test.iloc[:, 0].str.lower())
        x2_text = np.asarray(X_test.iloc[:, 1].str.lower())
        y = np.asarray(y_test)
        
        x1 = np.asarray(list(vocab_processor.transform(x1_text)))
        x2 = np.asarray(list(vocab_processor.transform(x2_text)))
        
        return x1, x2, y

class SiameseDataset(Dataset):
    """PyTorch Dataset for Siamese network that uses the InputHelper preprocessing."""
    
    def __init__(self, 
                 x1: np.ndarray,
                 x2: np.ndarray,
                 labels: Optional[np.ndarray] = None):
        """Initialize dataset.
        
        Args:
            x1: First sequence array
            x2: Second sequence array
            labels: Optional labels array
        """
        self.x1 = torch.from_numpy(x1).long()
        self.x2 = torch.from_numpy(x2).long()
        self.labels = torch.from_numpy(labels).float() if labels is not None else None
    
    def __len__(self) -> int:
        return len(self.x1)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.labels is not None:
            return self.x1[idx], self.x2[idx], self.labels[idx]
        return self.x1[idx], self.x2[idx]

def create_dataloaders(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    max_document_length: int = 15,
    batch_size: int = 64,
    percent_dev: int = 10,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader, VocabularyProcessor]:
    """Create DataLoaders using the original preprocessing logic.
    
    Args:
        X_train: Training data DataFrame
        y_train: Training labels
        X_test: Test data DataFrame
        y_test: Test labels
        max_document_length: Maximum sequence length
        batch_size: Batch size
        percent_dev: Percentage of training data to use for validation
        num_workers: Number of workers for DataLoader
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, vocab_processor)
    """
    input_helper = InputHelper()
    
    # Get train and dev sets
    train_set, dev_set, vocab_processor, _ = input_helper.get_datasets(
        X_train, y_train, max_document_length, percent_dev, batch_size
    )
    
    # Get test set
    x1_test, x2_test, y_test = input_helper.get_test_dataset(
        X_test, y_test, vocab_processor, max_document_length
    )
    
    # Create datasets
    train_dataset = SiameseDataset(*train_set)
    val_dataset = SiameseDataset(*dev_set)
    test_dataset = SiameseDataset(x1_test, x2_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, vocab_processor 