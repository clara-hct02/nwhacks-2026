import torch
import torch.nn as nn


class SpamClassifier(nn.Module):
    """
    LSTM-based neural network for spam message classification.
    
    Architecture:
        - Embedding layer: converts word indices to dense vectors
        - LSTM: captures sequential patterns in text
        - Fully connected layers: classification head
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Args:
            vocab_size: Size of the vocabulary (number of unique tokens)
            embedding_dim: Dimension of word embeddings
            hidden_dim: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout probability for regularization
            bidirectional: Whether to use bidirectional LSTM
        """
        super(SpamClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Account for bidirectional
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Binary classification: spam or not
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length) containing token indices
            
        Returns:
            Logits tensor of shape (batch_size, 1)
        """
        # x: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out: (batch_size, seq_length, hidden_dim * num_directions)
        
        # Use the last hidden state from both directions
        if self.lstm.bidirectional:
            # Concatenate the final forward and backward hidden states
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        # hidden: (batch_size, hidden_dim * num_directions)
        logits = self.classifier(hidden)
        
        return logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predictions (0 or 1) from input.
        
        Args:
            x: Input tensor of token indices
            
        Returns:
            Binary predictions (0 = not spam, 1 = spam)
        """
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.sigmoid(logits) > 0.5
            return predictions.long().squeeze()


def get_loss_fn():
    """Returns the loss function for training."""
    return nn.BCEWithLogitsLoss()


def get_optimizer(model: nn.Module, lr: float = 1e-3):
    """Returns the optimizer for training."""
    return torch.optim.Adam(model.parameters(), lr=lr)


# Example usage
if __name__ == "__main__":
    # Hyperparameters
    VOCAB_SIZE = 10000  # Adjust based on your tokenizer
    BATCH_SIZE = 32
    SEQ_LENGTH = 100    # Max message length
    
    # Create model
    model = SpamClassifier(vocab_size=VOCAB_SIZE)
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass with dummy data
    dummy_input = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LENGTH))
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

