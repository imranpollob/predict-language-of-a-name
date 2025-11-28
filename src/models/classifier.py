"""
Character-level RNN Classifier for name -> language classification.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class CharRNNClassifier(nn.Module):
    """
    Character-level RNN for classifying names by language.

    Architecture:
        - LSTM/GRU layer(s) for processing character sequences
        - Linear layer for classification
        - LogSoftmax activation

    Args:
        input_size: Size of input (number of characters)
        hidden_size: Size of hidden state
        output_size: Number of classes (languages)
        num_layers: Number of RNN layers
        dropout: Dropout probability
        rnn_type: "LSTM" or "GRU"
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        rnn_type: str = "LSTM",
    ):
        super(CharRNNClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type

        # RNN layer
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=False,
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size,
                hidden_size,
                num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=False,
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")

        # Dropout after RNN
        self.dropout = nn.Dropout(dropout)

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

        # Log softmax for classification
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(
        self,
        input: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            input: Input tensor [seq_len, batch, input_size]
            hidden: Hidden state (optional)

        Returns:
            output: Class probabilities [batch, output_size]
            hidden: Final hidden state
        """
        # Pass through RNN
        rnn_out, hidden = self.rnn(input, hidden)

        # Take the last output
        if self.rnn_type == "LSTM":
            last_hidden = hidden[0][-1]  # Last layer's hidden state
        else:
            last_hidden = hidden[-1]

        # Apply dropout
        last_hidden = self.dropout(last_hidden)

        # Classification layer
        output = self.fc(last_hidden)

        # Log softmax
        output = self.log_softmax(output)

        return output, hidden

    def init_hidden(
        self, batch_size: int = 1, device: torch.device = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state with zeros.

        Args:
            batch_size: Batch size
            device: Device to create tensor on

        Returns:
            Initial hidden state
        """
        if device is None:
            device = next(self.parameters()).device

        if self.rnn_type == "LSTM":
            return (
                torch.zeros(
                    self.num_layers, batch_size, self.hidden_size, device=device
                ),
                torch.zeros(
                    self.num_layers, batch_size, self.hidden_size, device=device
                ),
            )
        else:
            return torch.zeros(
                self.num_layers, batch_size, self.hidden_size, device=device
            )

    def predict(
        self, input: torch.Tensor, top_k: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make prediction with top-k results.

        Args:
            input: Input tensor [seq_len, 1, input_size]
            top_k: Number of top predictions to return

        Returns:
            values: Top-k log probabilities
            indices: Top-k class indices
        """
        with torch.no_grad():
            output, _ = self.forward(input)
            values, indices = output.topk(top_k)
        return values, indices

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Simple RNN version (from original notebook)
class SimpleRNN(nn.Module):
    """
    Simple RNN classifier (original version from notebook).
    Kept for compatibility and comparison.

    Args:
        input_size: Size of input
        hidden_size: Size of hidden state
        output_size: Number of output classes
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(SimpleRNN, self).__init__()

        self.hidden_size = hidden_size

        # Input to hidden
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)

        # Input to output
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

        # Log softmax
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(
        self, input: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            input: Input tensor [batch, input_size]
            hidden: Hidden state [batch, hidden_size]

        Returns:
            output: Class probabilities [batch, output_size]
            hidden: New hidden state [batch, hidden_size]
        """
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self, device: torch.device = None) -> torch.Tensor:
        """Initialize hidden state."""
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(1, self.hidden_size, device=device)
