"""
Classifier prediction module for inference.
"""

import torch
from typing import List, Tuple
from ..models.classifier import CharRNNClassifier
from ..data.preprocessing import nameToTensor, ALL_LETTERS


class ClassifierPredictor:
    """
    Predictor for name classification.

    Args:
        model: Trained CharRNNClassifier
        all_languages: List of language names
        all_letters: String of valid characters
        device: Device to run inference on
    """

    def __init__(
        self,
        model: CharRNNClassifier,
        all_languages: List[str],
        all_letters: str = ALL_LETTERS,
        device: torch.device = None,
    ):
        self.model = model
        self.all_languages = all_languages
        self.all_letters = all_letters

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
        all_languages: List[str],
        all_letters: str = ALL_LETTERS,
        device: torch.device = None,
    ):
        """
        Load predictor from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            all_languages: List of language names
            all_letters: String of valid characters
            device: Device to run inference on

        Returns:
            ClassifierPredictor instance
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Create model (assuming standard config)
        model = CharRNNClassifier(
            input_size=len(all_letters),
            hidden_size=256,
            output_size=len(all_languages),
            num_layers=2,
            dropout=0.3,
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        return cls(model, all_languages, all_letters, device)

    def predict(self, name: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Predict language(s) for a name.

        Args:
            name: Input name
            top_k: Number of top predictions to return

        Returns:
            List of (language, probability) tuples
        """
        with torch.no_grad():
            # Convert to tensor
            name_tensor = nameToTensor(name, self.all_letters).to(self.device)

            # Get predictions
            output, _ = self.model(name_tensor)

            # Convert log probabilities to probabilities
            probabilities = torch.exp(output)

            # Get top-k
            values, indices = probabilities.topk(top_k)

            # Format results
            results = []
            for i in range(top_k):
                language = self.all_languages[indices[0][i].item()]
                prob = values[0][i].item()
                results.append((language, prob))

            return results

    def batch_predict(
        self, names: List[str], top_k: int = 1
    ) -> List[List[Tuple[str, float]]]:
        """
        Predict languages for multiple names.

        Args:
            names: List of input names
            top_k: Number of top predictions per name

        Returns:
            List of prediction lists
        """
        results = []
        for name in names:
            results.append(self.predict(name, top_k))
        return results

    def evaluate(self, test_data: List[Tuple[str, str]]) -> Tuple[float, dict]:
        """
        Evaluate model on test data.

        Args:
            test_data: List of (name, true_language) tuples

        Returns:
            accuracy: Overall accuracy
            per_language_acc: Dictionary of per-language accuracy
        """
        correct = 0
        total = 0
        per_language_correct = {lang: 0 for lang in self.all_languages}
        per_language_total = {lang: 0 for lang in self.all_languages}

        for name, true_language in test_data:
            predictions = self.predict(name, top_k=1)
            predicted_language = predictions[0][0]

            total += 1
            per_language_total[true_language] += 1

            if predicted_language == true_language:
                correct += 1
                per_language_correct[true_language] += 1

        accuracy = correct / total if total > 0 else 0

        per_language_acc = {
            lang: (
                per_language_correct[lang] / per_language_total[lang]
                if per_language_total[lang] > 0
                else 0
            )
            for lang in self.all_languages
        }

        return accuracy, per_language_acc
