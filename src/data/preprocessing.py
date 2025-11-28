"""
Preprocessing utilities for character-level text processing.
Extracted from original notebooks and unified.
"""

import unicodedata
import string
import glob
import os
from typing import Dict, List, Tuple
import torch


# Default character set (can be overridden from config)
ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)

# Special tokens
EOS_TOKEN = N_LETTERS  # End of sequence for generation


def unicodeToAscii(s: str, all_letters: str = ALL_LETTERS) -> str:
    """
    Turn a Unicode string to plain ASCII.
    Removes accents and normalizes characters.

    Args:
        s: Input Unicode string
        all_letters: Valid character set

    Returns:
        Normalized ASCII string

    Example:
        >>> unicodeToAscii('Émile')
        'Emile'
    """
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in all_letters
    )


def letterToIndex(letter: str, all_letters: str = ALL_LETTERS) -> int:
    """
    Convert a letter to its index in the character set.

    Args:
        letter: Single character
        all_letters: Valid character set

    Returns:
        Index of the letter
    """
    return all_letters.find(letter)


def letterToTensor(letter: str, all_letters: str = ALL_LETTERS) -> torch.Tensor:
    """
    Convert a letter to a one-hot tensor.

    Args:
        letter: Single character
        all_letters: Valid character set

    Returns:
        One-hot tensor of shape [1, n_letters]
    """
    tensor = torch.zeros(1, len(all_letters))
    tensor[0][letterToIndex(letter, all_letters)] = 1
    return tensor


def nameToTensor(name: str, all_letters: str = ALL_LETTERS) -> torch.Tensor:
    """
    Convert a name/word to a tensor.

    Args:
        name: Input string
        all_letters: Valid character set

    Returns:
        Tensor of shape [len(name), 1, n_letters]
    """
    tensor = torch.zeros(len(name), 1, len(all_letters))
    for i, letter in enumerate(name):
        tensor[i][0][letterToIndex(letter, all_letters)] = 1
    return tensor


def lineToTensor(line: str, all_letters: str = ALL_LETTERS) -> torch.Tensor:
    """
    Alias for nameToTensor for compatibility.
    """
    return nameToTensor(line, all_letters)


def load_language_files(
    data_dir: str, all_letters: str = ALL_LETTERS
) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Load language data from text files.

    Args:
        data_dir: Directory containing language txt files
        all_letters: Valid character set

    Returns:
        Tuple of:
            - Dictionary mapping language -> list of names
            - List of all language names
    """
    language_names = {}
    all_languages = []

    # Find all .txt files
    text_files = glob.glob(os.path.join(data_dir, "*.txt"))

    if not text_files:
        raise ValueError(f"No .txt files found in {data_dir}")

    for filename in text_files:
        # Extract language name from filename
        language = os.path.splitext(os.path.basename(filename))[0]
        all_languages.append(language)

        # Read and process names
        with open(filename, encoding="utf-8") as f:
            lines = f.read().strip().split("\n")

        # Convert to ASCII and filter valid names
        names = [unicodeToAscii(line, all_letters) for line in lines]
        names = [name for name in names if name]  # Remove empty strings

        language_names[language] = names

    # Sort languages for consistency
    all_languages.sort()

    return language_names, all_languages


def normalize_string(s: str) -> str:
    """
    Normalize a string for translation tasks.
    Lowercase, trim, and remove non-letter characters.

    Args:
        s: Input string

    Returns:
        Normalized string
    """
    import re

    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()


def build_vocabulary(pairs: List[Tuple[str, str]]) -> Tuple["Lang", "Lang"]:
    """
    Build vocabulary from sentence pairs (for translation).

    Args:
        pairs: List of (source, target) sentence pairs

    Returns:
        Tuple of (input_lang, output_lang) vocabulary objects
    """
    # This will be used by the translation dataset
    # Placeholder for now, will be implemented in datasets.py
    pass


class Lang:
    """
    Vocabulary class for translation tasks.
    Tracks words and their indices.
    """

    def __init__(self, name: str):
        self.name = name
        self.word2index = {"SOS": 0, "EOS": 1}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence: str):
        """Add all words in a sentence to vocabulary."""
        for word in sentence.split(" "):
            self.addWord(word)

    def addWord(self, word: str):
        """Add a word to vocabulary."""
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def sentence_to_indices(self, sentence: str) -> List[int]:
        """Convert sentence to list of indices."""
        return [self.word2index[word] for word in sentence.split(" ")]


# Utility functions
def time_since(start_time: float) -> str:
    """
    Calculate elapsed time in human-readable format.

    Args:
        start_time: Start time from time.time()

    Returns:
        Formatted time string
    """
    import time
    import math

    now = time.time()
    elapsed = now - start_time
    minutes = math.floor(elapsed / 60)
    seconds = elapsed - minutes * 60
    return f"{minutes}m {seconds:.0f}s"


def random_choice(items: List) -> any:
    """
    Get random item from list.

    Args:
        items: List of items

    Returns:
        Random item
    """
    import random

    return items[random.randint(0, len(items) - 1)]
