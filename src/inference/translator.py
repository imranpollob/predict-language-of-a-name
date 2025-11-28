"""
Translation module for inference.
"""

import torch
from typing import List, Tuple, Optional
from ..models.translator import Seq2SeqWithAttention, EncoderRNN, AttnDecoderRNN
from ..data.preprocessing import Lang


class Translator:
    """
    Translator for French -> English (or vice versa).

    Args:
        model: Trained Seq2SeqWithAttention model
        input_lang: Input language vocabulary
        output_lang: Output language vocabulary
        device: Device to run inference on
    """

    def __init__(
        self,
        model: Seq2SeqWithAttention,
        input_lang: Lang,
        output_lang: Lang,
        device: torch.device = None,
    ):
        self.model = model
        self.input_lang = input_lang
        self.output_lang = output_lang

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        input_lang: Lang,
        output_lang: Lang,
        device: torch.device = None,
    ):
        """
        Load translator from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            input_lang: Input language vocabulary
            output_lang: Output language vocabulary
            device: Device to run inference on

        Returns:
            Translator instance
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Create model (assuming standard config)
        encoder = EncoderRNN(
            input_size=input_lang.n_words,
            embedding_size=256,
            hidden_size=512,
            num_layers=2,
            dropout=0.1,
        )

        decoder = AttnDecoderRNN(
            output_size=output_lang.n_words,
            embedding_size=256,
            hidden_size=512,
            num_layers=2,
            dropout=0.1,
        )

        model = Seq2SeqWithAttention(encoder, decoder, device)
        model.load_state_dict(checkpoint["model_state_dict"])

        return cls(model, input_lang, output_lang, device)

    def _sentence_to_tensor(self, sentence: str, lang: Lang) -> torch.Tensor:
        """Convert sentence to tensor of indices."""
        indices = lang.sentence_to_indices(sentence)
        indices.append(lang.word2index["EOS"])
        return torch.LongTensor(indices).to(self.device)

    def translate(
        self, sentence: str, max_length: int = 50, return_attention: bool = False
    ) -> Tuple[str, Optional[torch.Tensor]]:
        """
        Translate a sentence.

        Args:
            sentence: Input sentence
            max_length: Maximum output length
            return_attention: Whether to return attention weights

        Returns:
            translated_sentence: Translated sentence
            attention_weights: Attention weights (if return_attention=True)
        """
        with torch.no_grad():
            # Convert to tensor
            input_tensor = self._sentence_to_tensor(sentence, self.input_lang)

            # Translate
            decoded_indices, attention_weights = self.model.translate(
                input_tensor,
                max_length=max_length,
                EOS_token=self.output_lang.word2index["EOS"],
            )

            # Convert indices to words
            decoded_words = []
            for idx in decoded_indices:
                if idx < self.output_lang.n_words:
                    word = self.output_lang.index2word[idx]
                    decoded_words.append(word)

            translated_sentence = " ".join(decoded_words)

            if return_attention:
                return translated_sentence, attention_weights
            else:
                return translated_sentence, None

    def batch_translate(self, sentences: List[str], max_length: int = 50) -> List[str]:
        """
        Translate multiple sentences.

        Args:
            sentences: List of input sentences
            max_length: Maximum output length

        Returns:
            List of translated sentences
        """
        translations = []
        for sentence in sentences:
            translated, _ = self.translate(sentence, max_length)
            translations.append(translated)
        return translations

    def evaluate_bleu(
        self, test_pairs: List[Tuple[str, str]], max_length: int = 50
    ) -> float:
        """
        Evaluate model using BLEU score.

        Args:
            test_pairs: List of (source, target) sentence pairs
            max_length: Maximum output length

        Returns:
            Average BLEU score
        """
        try:
            from nltk.translate.bleu_score import sentence_bleu
        except ImportError:
            print("NLTK not installed. Install with: pip install nltk")
            return 0.0

        bleu_scores = []

        for source, target in test_pairs:
            translated, _ = self.translate(source, max_length)

            # Tokenize
            reference = [target.split()]
            candidate = translated.split()

            # Calculate BLEU
            score = sentence_bleu(reference, candidate)
            bleu_scores.append(score)

        return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

    def translate_with_visualization(
        self, sentence: str, max_length: int = 50
    ) -> Tuple[str, List[str], List[str], torch.Tensor]:
        """
        Translate and return data for visualization.

        Args:
            sentence: Input sentence
            max_length: Maximum output length

        Returns:
            translated_sentence: Translated sentence
            input_words: List of input words
            output_words: List of output words
            attention_weights: Attention weights matrix
        """
        translated, attention = self.translate(
            sentence, max_length=max_length, return_attention=True
        )

        input_words = sentence.split()
        output_words = translated.split()

        return translated, input_words, output_words, attention
