"""
Name generation module for inference.
"""

import torch
from typing import List
from ..models.generator import CharRNNGenerator
from ..data.preprocessing import ALL_LETTERS, EOS_TOKEN


class NameGenerator:
    """
    Generator for creating names from languages.

    Args:
        model: Trained CharRNNGenerator
        all_categories: List of category names (languages)
        all_letters: String of valid characters
        device: Device to run inference on
    """

    def __init__(
        self,
        model: CharRNNGenerator,
        all_categories: List[str],
        all_letters: str = ALL_LETTERS,
        device: torch.device = None,
    ):
        self.model = model
        self.all_categories = all_categories
        self.all_letters = all_letters
        self.n_letters = len(all_letters)

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
        all_categories: List[str],
        all_letters: str = ALL_LETTERS,
        device: torch.device = None,
    ):
        """
        Load generator from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            all_categories: List of category names
            all_letters: String of valid characters
            device: Device to run inference on

        Returns:
            NameGenerator instance
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Create model (assuming standard config)
        model = CharRNNGenerator(
            input_size=len(all_letters),
            category_size=len(all_categories),
            hidden_size=256,
            output_size=len(all_letters) + 1,  # +1 for EOS
            num_layers=2,
            dropout=0.2,
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        return cls(model, all_categories, all_letters, device)

    def _category_to_tensor(self, category: str) -> torch.Tensor:
        """Convert category name to one-hot tensor."""
        category_idx = self.all_categories.index(category)
        tensor = torch.zeros(1, len(self.all_categories), device=self.device)
        tensor[0][category_idx] = 1
        return tensor

    def generate(
        self,
        category: str,
        max_length: int = 20,
        temperature: float = 0.8,
        num_samples: int = 1,
    ) -> List[str]:
        """
        Generate name(s) for a given category.

        Args:
            category: Category/language name
            max_length: Maximum name length
            temperature: Sampling temperature (higher = more random)
            num_samples: Number of names to generate

        Returns:
            List of generated names
        """
        if category not in self.all_categories:
            raise ValueError(f"Unknown category: {category}")

        category_tensor = self._category_to_tensor(category)

        names = []
        for _ in range(num_samples):
            name = self.model.sample(
                category_tensor,
                max_length=max_length,
                temperature=temperature,
                all_letters=self.all_letters,
                EOS_token=EOS_TOKEN,
            )
            names.append(name)

        return names

    def generate_topk(
        self, category: str, max_length: int = 20, k: int = 3, num_samples: int = 1
    ) -> List[str]:
        """
        Generate name(s) using top-k sampling (less random).

        Args:
            category: Category/language name
            max_length: Maximum name length
            k: Top-k for sampling
            num_samples: Number of names to generate

        Returns:
            List of generated names
        """
        if category not in self.all_categories:
            raise ValueError(f"Unknown category: {category}")

        category_tensor = self._category_to_tensor(category)

        names = []
        for _ in range(num_samples):
            name = self.model.generate_topk(
                category_tensor,
                max_length=max_length,
                k=k,
                all_letters=self.all_letters,
                EOS_token=EOS_TOKEN,
            )
            names.append(name)

        return names

    def generate_all_categories(
        self,
        max_length: int = 20,
        temperature: float = 0.8,
        samples_per_category: int = 3,
    ) -> dict:
        """
        Generate samples for all categories.

        Args:
            max_length: Maximum name length
            temperature: Sampling temperature
            samples_per_category: Number of samples per category

        Returns:
            Dictionary mapping category -> list of generated names
        """
        results = {}

        for category in self.all_categories:
            names = self.generate(
                category,
                max_length=max_length,
                temperature=temperature,
                num_samples=samples_per_category,
            )
            results[category] = names

        return results

    def get_diversity_score(
        self,
        category: str,
        num_samples: int = 100,
        max_length: int = 20,
        temperature: float = 0.8,
    ) -> float:
        """
        Calculate diversity score (ratio of unique names).

        Args:
            category: Category/language name
            num_samples: Number of samples to generate
            max_length: Maximum name length
            temperature: Sampling temperature

        Returns:
            Diversity score (0-1, higher is more diverse)
        """
        names = self.generate(
            category,
            max_length=max_length,
            temperature=temperature,
            num_samples=num_samples,
        )

        unique_names = set(names)
        diversity = len(unique_names) / len(names)

        return diversity
