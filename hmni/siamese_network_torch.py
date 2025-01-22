import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseLSTM(nn.Module):
    """Siamese LSTM network for name matching."""
    
    def __init__(self, vocab_size, embedding_size=300, hidden_units=50, n_layers=3, dropout=0.2):
        """Initialize the model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_size: Dimension of character embeddings
            hidden_units: Number of LSTM hidden units
            n_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(SiameseLSTM, self).__init__()
        
        # Character embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_units,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_units * 2, hidden_units)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights for better training."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward_once(self, x):
        """Process one input sequence.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            
        Returns:
            Encoded representation of the input
        """
        # Embed characters
        embedded = self.embedding(x)  # (batch_size, sequence_length, embedding_size)
        
        # Pass through LSTM
        output, (hidden, _) = self.lstm(embedded)
        
        # Use the last hidden state from both directions
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # Project to output space
        out = self.fc(hidden)
        return out
    
    def forward(self, input1, input2):
        """Process a pair of inputs.
        
        Args:
            input1: First sequence
            input2: Second sequence
            
        Returns:
            Normalized distance between the encoded representations
        """
        # Get embeddings for both inputs
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        
        # Calculate normalized Euclidean distance
        # 1. Calculate squared difference
        squared_diff = torch.square(torch.subtract(output1, output2))
        # 2. Sum along feature dimension
        squared_dist = torch.sum(squared_diff, dim=1, keepdim=True)
        # 3. Take square root to get Euclidean distance
        distance = torch.sqrt(squared_dist)
        
        # 4. Calculate magnitudes of both vectors
        magnitude1 = torch.sqrt(torch.sum(torch.square(output1), dim=1, keepdim=True))
        magnitude2 = torch.sqrt(torch.sum(torch.square(output2), dim=1, keepdim=True))
        
        # 5. Normalize distance by sum of magnitudes
        normalized_distance = distance / (magnitude1 + magnitude2)
        
        # 6. Reshape to match original implementation
        normalized_distance = normalized_distance.reshape(-1)
        
        return normalized_distance

class ContrastiveLoss(nn.Module):
    """Contrastive loss function for Siamese network."""
    
    def __init__(self, margin=2.0):
        """Initialize loss function.
        
        Args:
            margin: Margin for contrastive loss
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, distance, target):
        """Calculate loss.
        
        Args:
            distance: Distance between pairs
            target: Target labels (1 for similar, 0 for dissimilar)
            
        Returns:
            Calculated loss value
        """
        # Loss for similar pairs: want distance to be small
        loss_similar = target * torch.pow(distance, 2)
        
        # Loss for dissimilar pairs: want distance to be larger than margin
        loss_dissimilar = (1 - target) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        
        # Combine losses
        loss = torch.mean(loss_similar + loss_dissimilar)
        return loss 