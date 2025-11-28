"""
Training utilities and trainer classes for all three tasks.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from typing import Optional, Dict, Any, Tuple
from tqdm import tqdm
import time
import os


class Trainer:
    """
    Base trainer class with common functionality.

    Args:
        model: PyTorch model
        device: Device to train on
        config: Configuration dictionary
    """

    def __init__(self, model: nn.Module, device: torch.device, config: Dict[str, Any]):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.losses = []
        self.best_loss = float("inf")

    def save_checkpoint(self, filepath: str, **kwargs):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "losses": self.losses,
            "best_loss": self.best_loss,
            **kwargs,
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.losses = checkpoint.get("losses", [])
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        print(f"Checkpoint loaded from {filepath}")
        return checkpoint


class ClassifierTrainer(Trainer):
    """
    Trainer for name classification model.

    Args:
        model: CharRNNClassifier model
        dataset: NameClassificationDataset
        device: Device to train on
        config: Configuration dictionary
    """

    def __init__(
        self, model: nn.Module, dataset, device: torch.device, config: Dict[str, Any]
    ):
        super().__init__(model, device, config)

        # Split dataset
        train_size = int(len(dataset) * config["train_split"])
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(config["seed"]),
        )

        print(
            f"Train size: {len(self.train_dataset)}, Val size: {len(self.val_dataset)}"
        )

        # Loss and optimizer
        self.criterion = nn.NLLLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["classifier"]["learning_rate"],
            weight_decay=config["classifier"].get("weight_decay", 0),
        )

        self.all_languages = dataset.all_languages

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        # Shuffle and iterate
        indices = torch.randperm(len(self.train_dataset))

        pbar = tqdm(indices, desc=f"Epoch {epoch+1}")
        for idx in pbar:
            name_tensor, language_tensor, language, name = self.train_dataset[idx]

            # Move to device
            name_tensor = name_tensor.to(self.device)
            language_tensor = language_tensor.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            output, hidden = self.model(name_tensor)

            # Calculate loss
            loss = self.criterion(output, language_tensor)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if "gradient_clip" in self.config["classifier"]:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["classifier"]["gradient_clip"]
                )

            # Update weights
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += 1
            correct += (predicted == language_tensor).sum().item()

            # Update progress bar
            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{100. * correct / total:.2f}%"}
            )

        avg_loss = total_loss / len(self.train_dataset)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for idx in range(len(self.val_dataset)):
                name_tensor, language_tensor, language, name = self.val_dataset[idx]

                # Move to device
                name_tensor = name_tensor.to(self.device)
                language_tensor = language_tensor.to(self.device)

                # Forward pass
                output, hidden = self.model(name_tensor)

                # Calculate loss
                loss = self.criterion(output, language_tensor)
                total_loss += loss.item()

                # Track accuracy
                _, predicted = output.max(1)
                total += 1
                correct += (predicted == language_tensor).sum().item()

        avg_loss = total_loss / len(self.val_dataset)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def train(self, num_epochs: int, checkpoint_dir: str = "models"):
        """
        Train the classifier.

        Args:
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        print(f"\nTraining Classifier for {num_epochs} epochs...")
        print(f"Model parameters: {self.model.count_parameters():,}")

        patience = self.config["classifier"].get("early_stopping_patience", 5)
        patience_counter = 0

        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)

            # Validate
            val_loss, val_acc = self.validate()

            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Save losses
            self.losses.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )

            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(
                    os.path.join(checkpoint_dir, "classifier_best.pth"),
                    epoch=epoch,
                    val_acc=val_acc,
                )
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

        print("\nTraining complete!")
        print(f"Best validation loss: {self.best_loss:.4f}")


class GeneratorTrainer(Trainer):
    """
    Trainer for name generation model.

    Args:
        model: CharRNNGenerator model
        dataset: NameGenerationDataset
        device: Device to train on
        config: Configuration dictionary
    """

    def __init__(
        self, model: nn.Module, dataset, device: torch.device, config: Dict[str, Any]
    ):
        super().__init__(model, device, config)

        self.dataset = dataset

        # Loss and optimizer
        self.criterion = nn.NLLLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config["generator"]["learning_rate"]
        )

    def train_step(self, category_tensor, input_tensor, target_tensor):
        """Single training step."""
        # Move to device
        category_tensor = category_tensor.to(self.device)
        input_tensor = input_tensor.to(self.device)
        target_tensor = target_tensor.to(self.device)

        # Zero gradients
        self.optimizer.zero_grad()

        # Initialize hidden
        hidden = self.model.initHidden(self.device)

        total_loss = 0

        # Process each character
        for i in range(input_tensor.size(0)):
            output, hidden = self.model(category_tensor, input_tensor[i], hidden)
            loss = self.criterion(output, target_tensor[i].unsqueeze(0))
            total_loss += loss

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        if "gradient_clip" in self.config["generator"]:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config["generator"]["gradient_clip"]
            )

        # Update weights
        self.optimizer.step()

        return total_loss.item() / input_tensor.size(0)

    def train(self, num_iterations: int, checkpoint_dir: str = "models"):
        """
        Train the generator.

        Args:
            num_iterations: Number of training iterations
            checkpoint_dir: Directory to save checkpoints
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        print(f"\nTraining Generator for {num_iterations} iterations...")
        print(f"Model parameters: {self.model.count_parameters():,}")

        print_every = self.config["generator"].get("print_every", 5000)
        all_losses = []
        current_loss = 0
        start_time = time.time()

        self.model.train()

        for iteration in range(1, num_iterations + 1):
            # Get random training example
            category_tensor, input_tensor, target_tensor = (
                self.dataset.random_training_example()
            )

            # Train step
            loss = self.train_step(category_tensor, input_tensor, target_tensor)
            current_loss += loss

            # Print progress
            if iteration % print_every == 0:
                avg_loss = current_loss / print_every
                elapsed = time.time() - start_time
                print(
                    f"[{iteration}/{num_iterations}] {elapsed:.0f}s - Loss: {avg_loss:.4f}"
                )
                all_losses.append(avg_loss)
                current_loss = 0

                # Save checkpoint
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    self.save_checkpoint(
                        os.path.join(checkpoint_dir, "generator_best.pth"),
                        iteration=iteration,
                    )

        self.losses = all_losses
        print("\nTraining complete!")


class TranslatorTrainer(Trainer):
    """
    Trainer for translation model.

    Args:
        model: Seq2SeqWithAttention model
        dataset: TranslationDataset
        device: Device to train on
        config: Configuration dictionary
    """

    def __init__(
        self, model: nn.Module, dataset, device: torch.device, config: Dict[str, Any]
    ):
        super().__init__(model, device, config)

        # Split dataset
        train_size = int(len(dataset) * config["train_split"])
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(config["seed"]),
        )

        print(
            f"Train size: {len(self.train_dataset)}, Val size: {len(self.val_dataset)}"
        )

        # Loss and optimizer
        self.criterion = nn.NLLLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config["translator"]["learning_rate"]
        )

        self.teacher_forcing_ratio = config["translator"].get(
            "teacher_forcing_ratio", 0.5
        )

    def train_step(self, input_tensor, target_tensor):
        """Single training step."""
        # Move to device
        input_tensor = input_tensor.unsqueeze(1).to(self.device)  # [seq_len, 1]
        target_tensor = target_tensor.unsqueeze(1).to(self.device)  # [seq_len, 1]

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass
        decoder_outputs, attentions = self.model(
            input_tensor, target_tensor, self.teacher_forcing_ratio
        )

        # Calculate loss
        loss = 0
        for di in range(decoder_outputs.size(0)):
            loss += self.criterion(decoder_outputs[di], target_tensor[di].squeeze(0))

        # Backward pass
        loss.backward()

        # Gradient clipping
        if "gradient_clip" in self.config["translator"]:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config["translator"]["gradient_clip"]
            )

        # Update weights
        self.optimizer.step()

        return loss.item() / decoder_outputs.size(0)

    def train(self, num_epochs: int, checkpoint_dir: str = "models"):
        """
        Train the translator.

        Args:
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        print(f"\nTraining Translator for {num_epochs} epochs...")
        print(f"Model parameters: {self.model.count_parameters():,}")

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0

            # Shuffle indices
            indices = torch.randperm(len(self.train_dataset))

            pbar = tqdm(indices, desc=f"Epoch {epoch+1}")
            for idx in pbar:
                input_tensor, target_tensor, input_text, target_text = (
                    self.train_dataset[idx]
                )

                # Train step
                loss = self.train_step(input_tensor, target_tensor)
                total_loss += loss

                pbar.set_postfix({"loss": f"{loss:.4f}"})

            avg_loss = total_loss / len(self.train_dataset)
            self.losses.append(avg_loss)

            print(f"\nEpoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

            # Save best model
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_checkpoint(
                    os.path.join(checkpoint_dir, "translator_best.pth"), epoch=epoch
                )

        print("\nTraining complete!")
        print(f"Best loss: {self.best_loss:.4f}")
