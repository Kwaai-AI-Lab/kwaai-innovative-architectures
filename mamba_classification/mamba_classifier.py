import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class MambaClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_classes, num_layers=2, d_state=16, d_conv=4, expand=2):
        """
        A Mamba-based classification model.

        Args:
            vocab_size (int): Size of the vocabulary for the input embedding.
            d_model (int): The dimension of the input features.
            num_classes (int): The number of output classes for classification.
            num_layers (int): The number of Mamba layers.
            d_state (int): SSM state expansion factor.
            d_conv (int): Convolution width.
            expand (int): Expansion factor for the Mamba block.
        """
        super().__init__()
        self.d_model = d_model

        # Define the input embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Define the Mamba layers
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(num_layers)
        ])

        # Define the classification head
        self.classifer = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        """
        Forward pass for classification.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        
        Returns:
            torch.Tensor: Logits for each class.
        """

        # Convert input into embeddings
        x = self.embedding(x)
        
        for layer in self.mamba_layers:
            x = layer(x)
        
        x = x.mean(dim=1)  # Global average pooling

        logits = self.classifer(x)
        return logits