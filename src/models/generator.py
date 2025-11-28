"""
Character-level RNN Generator for language -> name generation.
"""

import torch
import torch.nn as nn
from typing import Optional


class CharRNNGenerator(nn.Module):
    """
    Character-level RNN for generating names conditioned on language.

    Architecture:
        - Category conditioning at each timestep
        - LSTM/GRU for sequence generation
        - Character prediction at each step

    Args:
        input_size: Size of input (number of characters)
        category_size: Number of categories (languages)
        hidden_size: Size of hidden state
        output_size: Output size (number of characters + EOS)
        num_layers: Number of RNN layers
        dropout: Dropout probability
        rnn_type: "LSTM" or "GRU"
    """

    def __init__(
        self,
        input_size: int,
        category_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        rnn_type: str = "LSTM",
    ):
        super(CharRNNGenerator, self).__init__()

        self.input_size = input_size
        self.category_size = category_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type

        # Combine category and input
        combined_input_size = category_size + input_size

        # RNN layer
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                combined_input_size,
                hidden_size,
                num_layers,
                dropout=dropout if num_layers > 1 else 0,
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                combined_input_size,
                hidden_size,
                num_layers,
                dropout=dropout if num_layers > 1 else 0,
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

        # Log softmax for character prediction
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(
        self,
        category: torch.Tensor,
        input: torch.Tensor,
        hidden: Optional[tuple] = None,
    ) -> tuple:
        """
        Forward pass.

        Args:
            category: Category tensor [1, category_size]
            input: Input character tensor [1, input_size]
            hidden: Hidden state (optional)

        Returns:
            output: Character probabilities [1, output_size]
            hidden: New hidden state
        """
        # Combine category and input
        combined = torch.cat((category, input), 1)
        combined = combined.unsqueeze(0)  # Add sequence dimension

        # Pass through RNN
        rnn_out, hidden = self.rnn(combined, hidden)

        # Output layer
        output = self.fc(rnn_out.squeeze(0))
        output = self.log_softmax(output)

        return output, hidden

    def initHidden(self, device: torch.device = None):
        """
        Initialize hidden state with zeros.

        Args:
            device: Device to create tensor on

        Returns:
            Initial hidden state
        """
        if device is None:
            device = next(self.parameters()).device

        if self.rnn_type == "LSTM":
            return (
                torch.zeros(self.num_layers, 1, self.hidden_size, device=device),
                torch.zeros(self.num_layers, 1, self.hidden_size, device=device),
            )
        else:
            return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)

    def sample(
        self,
        category: torch.Tensor,
        max_length: int = 20,
        temperature: float = 0.8,
        start_letter: str = None,
        all_letters: str = None,
        EOS_token: int = None,
    ) -> str:
        """
        Generate a name by sampling from the model.

        Args:
            category: Category tensor [1, category_size]
            max_length: Maximum name length
            temperature: Sampling temperature (higher = more random)
            start_letter: Optional starting letter
            all_letters: String of all valid characters
            EOS_token: End of sequence token index

        Returns:
            Generated name string
        """
        with torch.no_grad():
            device = next(self.parameters()).device
            category = category.to(device)

            hidden = self.initHidden(device)
            output_name = ""

            # Start with a letter or random
            if start_letter:
                input = torch.zeros(1, self.input_size, device=device)
                input[0][all_letters.find(start_letter)] = 1
            else:
                input = torch.zeros(1, self.input_size, device=device)
                input[0][0] = 1  # Start with first letter

            for i in range(max_length):
                output, hidden = self.forward(category, input, hidden)

                # Apply temperature
                output_dist = output.data.div(temperature).exp()

                # Sample from distribution
                top_i = torch.multinomial(output_dist, 1)[0]

                # Check for EOS
                if EOS_token is not None and top_i == EOS_token:
                    break

                # Get letter
                if all_letters and top_i < len(all_letters):
                    letter = all_letters[top_i]
                    output_name += letter

                    # Prepare next input
                    input = torch.zeros(1, self.input_size, device=device)
                    input[0][top_i] = 1
                else:
                    break

            return output_name

    def generate_topk(
        self,
        category: torch.Tensor,
        max_length: int = 20,
        k: int = 3,
        all_letters: str = None,
        EOS_token: int = None,
    ) -> str:
        """
        Generate a name by selecting top-k at each step (less random).

        Args:
            category: Category tensor [1, category_size]
            max_length: Maximum name length
            k: Top-k for sampling
            all_letters: String of all valid characters
            EOS_token: End of sequence token index

        Returns:
            Generated name string
        """
        with torch.no_grad():
            device = next(self.parameters()).device
            category = category.to(device)

            hidden = self.initHidden(device)
            output_name = ""

            input = torch.zeros(1, self.input_size, device=device)
            input[0][0] = 1  # Start token

            for i in range(max_length):
                output, hidden = self.forward(category, input, hidden)

                # Get top-k
                topv, topi = output.topk(k)

                # Sample from top-k
                topi = topi[0]
                choice_i = topi[torch.randint(0, k, (1,))].item()

                # Check for EOS
                if EOS_token is not None and choice_i == EOS_token:
                    break

                # Get letter
                if all_letters and choice_i < len(all_letters):
                    letter = all_letters[choice_i]
                    output_name += letter

                    # Prepare next input
                    input = torch.zeros(1, self.input_size, device=device)
                    input[0][choice_i] = 1
                else:
                    break

            return output_name

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
