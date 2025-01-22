import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from hmni.input_helpers_torch import create_dataloaders
from hmni.siamese_network_torch import SiameseLSTM, ContrastiveLoss

def load_data():
    """Load the prepared data splits."""
    print("Loading data splits...")
    X_train = pd.read_csv('X_train.csv')
    X_test = pd.read_csv('X_test.csv')
    y_train = pd.read_csv('y_train.csv').values.ravel()
    y_test = pd.read_csv('y_test.csv').values.ravel()
    return X_train, X_test, y_train, y_test

def train_siamese_model(num_epochs=20, patience=3):
    """Train and evaluate the Siamese Network model."""
    print("\n=== Phase 3: Training Neural Network Model (Siamese Network) ===")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Extract name columns for Siamese network
    train_names = X_train[['name_a', 'name_b']]
    test_names = X_test[['name_a', 'name_b']]
    
    # Set device and batch size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32 if device.type == 'cpu' else 64
    
    # Create dataloaders
    print("\nPreparing data loaders...")
    train_loader, val_loader, test_loader, vocab_processor = create_dataloaders(
        train_names, y_train,
        test_names, y_test,
        max_document_length=15,
        batch_size=batch_size,
        percent_dev=10,
        num_workers=0 if device.type == 'cpu' else 4
    )

    # Save vocabulary immediately after creation
    print(f"\nSaving vocabulary (size: {len(vocab_processor.vocabulary_)})...")
    vocab_processor.save('vocab_siamese.pkl')

    # Initialize model and training components
    model = SiameseLSTM(
        vocab_size=len(vocab_processor.vocabulary_),
        embedding_size=300,
        hidden_units=50,
        n_layers=2,
        dropout=0.2
    ).to(device)
    
    criterion = ContrastiveLoss(margin=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    print(f"\nUsing device: {device}")
    print(f"Batch size: {batch_size}")
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for x1, x2, labels in progress_bar:
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            
            optimizer.zero_grad()
            distance = model(x1, x2)
            loss = criterion(distance, labels)
            
            # Add L1 regularization
            l1_lambda = 1e-6
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            predictions = (distance < 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{train_loss/(progress_bar.n+1):.4f}",
                'acc': f"{100.*train_correct/train_total:.2f}%"
            })
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x1, x2, labels in val_loader:
                x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
                
                distance = model(x1, x2)
                loss = criterion(distance, labels)
                val_loss += loss.item()
                
                predictions = (distance < 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= len(val_loader)
        val_accuracy = 100. * val_correct / val_total
        
        print(f"\nValidation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save checkpoint with all necessary information
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'vocab_size': len(vocab_processor.vocabulary_),
                'training_args': {
                    'embedding_size': 300,
                    'hidden_units': 50,
                    'n_layers': 2,
                    'dropout': 0.2,
                    'max_document_length': 15
                }
            }
            print(f"Saving best model checkpoint (val_loss: {best_val_loss:.4f})...")
            torch.save(checkpoint, '../hmni/models/latin/siamese_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
    
    # Final evaluation on test set
    model.eval()
    test_correct = 0
    test_total = 0
    
    print("\nEvaluating on test set...")
    with torch.no_grad():
        for x1, x2, labels in test_loader:
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            
            distance = model(x1, x2)
            predictions = (distance < 0.5).float()
            test_correct += (predictions == labels).sum().item()
            test_total += labels.size(0)
    
    test_accuracy = 100. * test_correct / test_total
    print(f"\nFinal Test Accuracy: {test_accuracy:.2f}%")
    
    # Save final model
    final_checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_test_accuracy': test_accuracy,
        'best_val_loss': best_val_loss,
        'vocab_size': len(vocab_processor.vocabulary_),
        'training_args': {
            'embedding_size': 300,
            'hidden_units': 50,
            'n_layers': 2,
            'dropout': 0.2,
            'max_document_length': 15
        }
    }
    print("\nSaving final model checkpoint...")
    torch.save(final_checkpoint, '../hmni/models/latin/siamese_model_final.pt')

    return model, vocab_processor

if __name__ == "__main__":
    train_siamese_model() 