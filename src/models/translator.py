"""
Sequence-to-Sequence model with Attention for translation.
Implements Encoder-Decoder architecture with Bahdanau attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import random


class EncoderRNN(nn.Module):
    """
    Encoder for sequence-to-sequence model.
    Uses bidirectional GRU to process input sequence.

    Args:
        input_size: Size of input vocabulary
        embedding_size: Size of embeddings
        hidden_size: Size of hidden state
        num_layers: Number of GRU layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_size: int,
        embedding_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(
            embedding_size,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

    def forward(
        self, input: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            input: Input tensor [seq_len, batch_size]
            hidden: Initial hidden state (optional)

        Returns:
            outputs: Encoder outputs [seq_len, batch, hidden_size * 2]
            hidden: Final hidden state [num_layers * 2, batch, hidden_size]
        """
        embedded = self.embedding(input)
        outputs, hidden = self.gru(embedded, hidden)

        # Sum bidirectional outputs
        outputs = outputs[:, :, : self.hidden_size] + outputs[:, :, self.hidden_size :]

        return outputs, hidden


class BahdanauAttention(nn.Module):
    """
    Bahdanau (additive) attention mechanism.

    Args:
        hidden_size: Size of hidden state
    """

    def __init__(self, hidden_size: int):
        super(BahdanauAttention, self).__init__()

        self.hidden_size = hidden_size

        # Attention layers
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self, hidden: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate attention weights and context vector.

        Args:
            hidden: Decoder hidden state [batch, hidden_size]
            encoder_outputs: Encoder outputs [seq_len, batch, hidden_size]

        Returns:
            context: Context vector [batch, hidden_size]
            attn_weights: Attention weights [batch, seq_len]
        """
        seq_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)

        # Repeat hidden state seq_len times
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # [batch, seq_len, hidden]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # [batch, seq_len, hidden]

        # Calculate attention energies
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), 2)))
        attention = self.v(energy).squeeze(2)  # [batch, seq_len]

        # Softmax to get attention weights
        attn_weights = F.softmax(attention, dim=1)

        # Calculate context vector
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)  # [batch, hidden_size]

        return context, attn_weights


class AttnDecoderRNN(nn.Module):
    """
    Decoder with attention mechanism.

    Args:
        output_size: Size of output vocabulary
        embedding_size: Size of embeddings
        hidden_size: Size of hidden state
        num_layers: Number of GRU layers
        dropout: Dropout probability
        attention_type: Type of attention ("bahdanau" or "luong")
    """

    def __init__(
        self,
        output_size: int,
        embedding_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        attention_type: str = "bahdanau",
    ):
        super(AttnDecoderRNN, self).__init__()

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention_type = attention_type

        self.embedding = nn.Embedding(output_size, embedding_size)
        self.dropout = nn.Dropout(dropout)

        # Attention mechanism
        self.attention = BahdanauAttention(hidden_size)

        # GRU input size is embedding + context
        self.gru = nn.GRU(
            embedding_size + hidden_size,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output layer
        self.out = nn.Linear(hidden_size, output_size)

    def forward(
        self, input: torch.Tensor, hidden: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for one time step.

        Args:
            input: Input token [batch_size]
            hidden: Hidden state [num_layers, batch, hidden_size]
            encoder_outputs: Encoder outputs [seq_len, batch, hidden_size]

        Returns:
            output: Output predictions [batch, output_size]
            hidden: New hidden state
            attn_weights: Attention weights [batch, seq_len]
        """
        # Embedding
        embedded = self.embedding(input).unsqueeze(0)  # [1, batch, embedding]
        embedded = self.dropout(embedded)

        # Calculate attention
        context, attn_weights = self.attention(hidden[-1], encoder_outputs)

        # Combine embedded input and context
        rnn_input = torch.cat((embedded, context.unsqueeze(0)), 2)

        # Pass through GRU
        output, hidden = self.gru(rnn_input, hidden)

        # Output layer
        output = self.out(output.squeeze(0))

        return output, hidden, attn_weights

    def initHidden(
        self, batch_size: int = 1, device: torch.device = None
    ) -> torch.Tensor:
        """Initialize hidden state."""
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)


class Seq2SeqWithAttention(nn.Module):
    """
    Complete Sequence-to-Sequence model with attention.
    Combines encoder and decoder.

    Args:
        encoder: EncoderRNN instance
        decoder: AttnDecoderRNN instance
        device: Device to run on
    """

    def __init__(
        self, encoder: EncoderRNN, decoder: AttnDecoderRNN, device: torch.device
    ):
        super(Seq2SeqWithAttention, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(
        self,
        input_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
        max_length: int = 50,
    ) -> Tuple[torch.Tensor, list]:
        """
        Forward pass with teacher forcing.

        Args:
            input_tensor: Input sequence [input_len, batch_size]
            target_tensor: Target sequence [target_len, batch_size]
            teacher_forcing_ratio: Probability of using teacher forcing
            max_length: Maximum output length

        Returns:
            outputs: Decoder outputs [target_len, batch, vocab_size]
            attentions: List of attention weights
        """
        batch_size = input_tensor.size(1)
        target_len = target_tensor.size(0)

        # Encode
        encoder_outputs, encoder_hidden = self.encoder(input_tensor)

        # Initialize decoder
        decoder_hidden = encoder_hidden[: self.decoder.num_layers]
        decoder_input = torch.tensor(
            [[0]] * batch_size, device=self.device
        )  # SOS tokens

        # Store outputs and attentions
        decoder_outputs = []
        attentions = []

        # Decode
        use_teacher_forcing = random.random() < teacher_forcing_ratio

        for di in range(target_len):
            decoder_output, decoder_hidden, attn_weights = self.decoder(
                decoder_input.squeeze(0), decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if use_teacher_forcing:
                # Teacher forcing: use target as next input
                decoder_input = target_tensor[di].unsqueeze(0)
            else:
                # Use own prediction
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(1).detach().unsqueeze(0)

        # Stack outputs
        decoder_outputs = torch.stack(decoder_outputs)

        return decoder_outputs, attentions

    def translate(
        self, input_tensor: torch.Tensor, max_length: int = 50, EOS_token: int = 1
    ) -> Tuple[list, torch.Tensor]:
        """
        Translate a sentence (greedy decoding).

        Args:
            input_tensor: Input sequence [input_len]
            max_length: Maximum output length
            EOS_token: End of sequence token

        Returns:
            decoded_words: List of decoded word indices
            attention_weights: Attention weights [output_len, input_len]
        """
        with torch.no_grad():
            input_tensor = input_tensor.unsqueeze(1)  # Add batch dimension

            # Encode
            encoder_outputs, encoder_hidden = self.encoder(input_tensor)

            # Initialize decoder
            decoder_hidden = encoder_hidden[: self.decoder.num_layers]
            decoder_input = torch.tensor([[0]], device=self.device)  # SOS token

            decoded_words = []
            decoder_attentions = []

            # Decode
            for di in range(max_length):
                decoder_output, decoder_hidden, attn_weights = self.decoder(
                    decoder_input.squeeze(0), decoder_hidden, encoder_outputs
                )

                decoder_attentions.append(attn_weights.cpu())

                # Get top prediction
                _, topi = decoder_output.topk(1)
                decoded_idx = topi.item()

                if decoded_idx == EOS_token:
                    break

                decoded_words.append(decoded_idx)
                decoder_input = topi.squeeze(1)

            # Stack attention weights
            if decoder_attentions:
                attention_weights = torch.cat(decoder_attentions, dim=0)
            else:
                attention_weights = None

            return decoded_words, attention_weights

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
