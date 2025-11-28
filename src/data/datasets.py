"""
PyTorch Dataset classes for all three tasks.
"""

import torch
from torch.utils.data import Dataset
import random
from typing import List, Tuple, Dict, Optional
from .preprocessing import (
    load_language_files,
    unicodeToAscii,
    nameToTensor,
    letterToIndex,
    normalize_string,
    Lang,
    ALL_LETTERS,
    EOS_TOKEN,
)


class NameClassificationDataset(Dataset):
    """
    Dataset for name -> language classification.

    Args:
        data_dir: Directory containing language text files
        all_letters: Valid character set
        split: Train or validation split (handled externally)
    """

    def __init__(self, data_dir: str, all_letters: str = ALL_LETTERS, transform=None):
        self.all_letters = all_letters
        self.n_letters = len(all_letters)
        self.transform = transform

        # Load all language files
        self.language_names, self.all_languages = load_language_files(
            data_dir, all_letters
        )
        self.n_languages = len(self.all_languages)

        # Flatten into (name, language, language_idx) tuples
        self.data = []
        for language in self.all_languages:
            language_idx = self.all_languages.index(language)
            for name in self.language_names[language]:
                self.data.append((name, language, language_idx))

        print(f"Loaded {len(self.data)} names from {self.n_languages} languages")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        """
        Returns:
            name_tensor: Tensor of shape [seq_len, 1, n_letters]
            language_tensor: Tensor with language index
            language: Language name string
            name: Name string
        """
        name, language, language_idx = self.data[idx]

        name_tensor = nameToTensor(name, self.all_letters)
        language_tensor = torch.tensor([language_idx], dtype=torch.long)

        if self.transform:
            name_tensor = self.transform(name_tensor)

        return name_tensor, language_tensor, language, name

    def get_language_list(self) -> List[str]:
        """Get list of all languages."""
        return self.all_languages

    def get_random_example(self) -> Tuple[str, str]:
        """Get a random (name, language) pair."""
        name, language, _ = random.choice(self.data)
        return name, language


class NameGenerationDataset(Dataset):
    """
    Dataset for language -> name generation.
    Returns category tensor, input tensor, and target tensor for training.

    Args:
        data_dir: Directory containing language text files
        all_letters: Valid character set
    """

    def __init__(self, data_dir: str, all_letters: str = ALL_LETTERS):
        self.all_letters = all_letters
        self.n_letters = len(all_letters)

        # Load all language files
        self.language_names, self.all_languages = load_language_files(
            data_dir, all_letters
        )
        self.n_categories = len(self.all_languages)

        # Flatten into (name, language, language_idx) tuples
        self.data = []
        for language in self.all_languages:
            language_idx = self.all_languages.index(language)
            for name in self.language_names[language]:
                self.data.append((name, language, language_idx))

        print(
            f"Loaded {len(self.data)} names for generation from {self.n_categories} languages"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            category_tensor: One-hot category [1, n_categories]
            input_tensor: Input sequence [seq_len, 1, n_letters]
            target_tensor: Target indices [seq_len] (includes EOS)
        """
        name, language, language_idx = self.data[idx]

        # Category tensor (one-hot)
        category_tensor = torch.zeros(1, self.n_categories)
        category_tensor[0][language_idx] = 1

        # Input tensor: all letters except last
        input_tensor = torch.zeros(len(name), 1, self.n_letters)
        for i, letter in enumerate(name):
            input_tensor[i][0][letterToIndex(letter, self.all_letters)] = 1

        # Target tensor: second letter to end + EOS
        target_list = []
        for i in range(1, len(name)):
            target_list.append(letterToIndex(name[i], self.all_letters))
        target_list.append(EOS_TOKEN)  # Add EOS token
        target_tensor = torch.LongTensor(target_list)

        return category_tensor, input_tensor, target_tensor

    def random_training_example(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a random training example."""
        idx = random.randint(0, len(self.data) - 1)
        return self[idx]

    def get_category_list(self) -> List[str]:
        """Get list of all categories (languages)."""
        return self.all_languages


class TranslationDataset(Dataset):
    """
    Dataset for French -> English translation.

    Args:
        data_file: Path to translation data file
        reverse: If True, translate English -> French
        max_length: Maximum sentence length
    """

    def __init__(self, data_file: str, reverse: bool = False, max_length: int = 10):
        self.max_length = max_length
        self.reverse = reverse

        # Read and prepare data
        self.input_lang, self.output_lang, self.pairs = self._prepare_data(
            data_file, reverse
        )

        print(f"Loaded {len(self.pairs)} sentence pairs")
        print(
            f"Input language: {self.input_lang.name} ({self.input_lang.n_words} words)"
        )
        print(
            f"Output language: {self.output_lang.name} ({self.output_lang.n_words} words)"
        )

    def _prepare_data(
        self, data_file: str, reverse: bool
    ) -> Tuple[Lang, Lang, List[Tuple[str, str]]]:
        """Read and prepare translation data."""
        # Read file
        with open(data_file, encoding="utf-8") as f:
            lines = f.read().strip().split("\n")

        # Split into pairs and normalize
        pairs = []
        for line in lines:
            parts = line.split("\t")
            if len(parts) >= 2:
                pairs.append([normalize_string(parts[0]), normalize_string(parts[1])])

        # Reverse pairs if needed
        if reverse:
            pairs = [list(reversed(p)) for p in pairs]
            input_lang = Lang("eng")
            output_lang = Lang("fra")
        else:
            input_lang = Lang("fra")
            output_lang = Lang("eng")

        # Filter by length
        pairs = self._filter_pairs(pairs)

        # Build vocabularies
        for pair in pairs:
            input_lang.addSentence(pair[0])
            output_lang.addSentence(pair[1])

        return input_lang, output_lang, pairs

    def _filter_pairs(self, pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Filter pairs by length and prefix."""
        eng_prefixes = (
            "i am ",
            "i m ",
            "he is",
            "he s ",
            "she is",
            "she s ",
            "you are",
            "you re ",
            "we are",
            "we re ",
            "they are",
            "they re ",
        )

        filtered = []
        for pair in pairs:
            if (
                len(pair[0].split(" ")) < self.max_length
                and len(pair[1].split(" ")) < self.max_length
            ):
                # Check if starts with common prefix (for English)
                if self.reverse:
                    if pair[0].startswith(eng_prefixes):
                        filtered.append(pair)
                else:
                    if pair[1].startswith(eng_prefixes):
                        filtered.append(pair)

        return filtered

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        """
        Returns:
            input_tensor: Input sentence indices [input_len]
            target_tensor: Target sentence indices [target_len]
            input_text: Input sentence string
            target_text: Target sentence string
        """
        input_text, target_text = self.pairs[idx]

        # Convert to indices
        input_indices = self.input_lang.sentence_to_indices(input_text)
        target_indices = self.output_lang.sentence_to_indices(target_text)

        # Add EOS token
        input_indices.append(self.input_lang.word2index["EOS"])
        target_indices.append(self.output_lang.word2index["EOS"])

        input_tensor = torch.LongTensor(input_indices)
        target_tensor = torch.LongTensor(target_indices)

        return input_tensor, target_tensor, input_text, target_text

    def random_pair(self) -> Tuple[str, str]:
        """Get a random pair of sentences."""
        return random.choice(self.pairs)


# Collate functions for DataLoader


def classification_collate_fn(batch):
    """
    Custom collate function for classification (handles variable-length sequences).
    """
    # Batch is a list of (name_tensor, language_tensor, language, name)
    # We'll return them as-is since we process one at a time in simple RNN
    return batch


def generation_collate_fn(batch):
    """
    Custom collate function for generation.
    """
    return batch


def translation_collate_fn(batch):
    """
    Custom collate function for translation (pads sequences).
    """
    # Sort by input length (descending) for pack_padded_sequence
    batch.sort(key=lambda x: len(x[0]), reverse=True)

    input_tensors, target_tensors, input_texts, target_texts = zip(*batch)

    # Pad sequences
    input_lengths = [len(x) for x in input_tensors]
    target_lengths = [len(x) for x in target_tensors]

    # Pad with 0 (or use nn.utils.rnn.pad_sequence)
    max_input_len = max(input_lengths)
    max_target_len = max(target_lengths)

    padded_inputs = torch.zeros(len(batch), max_input_len, dtype=torch.long)
    padded_targets = torch.zeros(len(batch), max_target_len, dtype=torch.long)

    for i, (inp, tgt) in enumerate(zip(input_tensors, target_tensors)):
        padded_inputs[i, : len(inp)] = inp
        padded_targets[i, : len(tgt)] = tgt

    return (
        padded_inputs,
        padded_targets,
        input_lengths,
        target_lengths,
        input_texts,
        target_texts,
    )
