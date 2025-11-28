"""
Template for creating training scripts.
Copy and modify this for each task (classifier, generator, translator).
"""

# Example 1: Train Classifier
# Save as: train_classifier.py

"""
#!/usr/bin/env python3
import torch
from src.utils.config import load_config, get_device, set_seed
from src.data.datasets import NameClassificationDataset
from src.models.classifier import CharRNNClassifier
from src.training.trainer import ClassifierTrainer

def main():
    # Load configuration
    config = load_config('config.yaml')
    set_seed(config['seed'])
    device = get_device(config['device'])
    
    print("Loading dataset...")
    dataset = NameClassificationDataset(config['data_dir'])
    print(f"Loaded {len(dataset)} names from {dataset.n_languages} languages")
    
    print("\nCreating model...")
    model = CharRNNClassifier(
        input_size=config['classifier']['input_size'],
        hidden_size=config['classifier']['hidden_size'],
        output_size=dataset.n_languages,
        num_layers=config['classifier']['num_layers'],
        dropout=config['classifier']['dropout'],
        rnn_type=config['classifier']['model_type']
    )
    print(f"Model has {model.count_parameters():,} parameters")
    
    print("\nCreating trainer...")
    trainer = ClassifierTrainer(model, dataset, device, config)
    
    print("\nStarting training...")
    trainer.train(
        num_epochs=config['classifier']['epochs'],
        checkpoint_dir=config['checkpoint_dir']
    )
    
    print("\nTraining complete!")
    print(f"Best model saved to: {config['checkpoint_dir']}/classifier_best.pth")

if __name__ == '__main__':
    main()
"""


# Example 2: Train Generator
# Save as: train_generator.py

"""
#!/usr/bin/env python3
import torch
from src.utils.config import load_config, get_device, set_seed
from src.data.datasets import NameGenerationDataset
from src.models.generator import CharRNNGenerator
from src.training.trainer import GeneratorTrainer

def main():
    # Load configuration
    config = load_config('config.yaml')
    set_seed(config['seed'])
    device = get_device(config['device'])
    
    print("Loading dataset...")
    dataset = NameGenerationDataset(config['data_dir'])
    print(f"Loaded {len(dataset)} names from {dataset.n_categories} categories")
    
    print("\nCreating model...")
    model = CharRNNGenerator(
        input_size=config['generator']['input_size'],
        category_size=dataset.n_categories,
        hidden_size=config['generator']['hidden_size'],
        output_size=config['generator']['output_size'],
        num_layers=config['generator']['num_layers'],
        dropout=config['generator']['dropout'],
        rnn_type=config['generator']['model_type']
    )
    print(f"Model has {model.count_parameters():,} parameters")
    
    print("\nCreating trainer...")
    trainer = GeneratorTrainer(model, dataset, device, config)
    
    print("\nStarting training...")
    trainer.train(
        num_iterations=config['generator']['epochs'],
        checkpoint_dir=config['checkpoint_dir']
    )
    
    print("\nTraining complete!")
    print(f"Best model saved to: {config['checkpoint_dir']}/generator_best.pth")

if __name__ == '__main__':
    main()
"""


# Example 3: Train Translator
# Save as: train_translator.py

"""
#!/usr/bin/env python3
import torch
from src.utils.config import load_config, get_device, set_seed
from src.data.datasets import TranslationDataset
from src.models.translator import EncoderRNN, AttnDecoderRNN, Seq2SeqWithAttention
from src.training.trainer import TranslatorTrainer

def main():
    # Load configuration
    config = load_config('config.yaml')
    set_seed(config['seed'])
    device = get_device(config['device'])
    
    print("Loading dataset...")
    # Note: You'll need to download the translation data first
    dataset = TranslationDataset(
        'data/fra-eng.txt',  # Update path as needed
        reverse=True,  # True for eng->fra, False for fra->eng
        max_length=config['translator']['max_length']
    )
    print(f"Loaded {len(dataset)} sentence pairs")
    
    print("\nCreating model...")
    encoder = EncoderRNN(
        input_size=dataset.input_lang.n_words,
        embedding_size=config['translator']['encoder']['embedding_size'],
        hidden_size=config['translator']['encoder']['hidden_size'],
        num_layers=config['translator']['encoder']['num_layers'],
        dropout=config['translator']['encoder']['dropout']
    )
    
    decoder = AttnDecoderRNN(
        output_size=dataset.output_lang.n_words,
        embedding_size=config['translator']['decoder']['embedding_size'],
        hidden_size=config['translator']['decoder']['hidden_size'],
        num_layers=config['translator']['decoder']['num_layers'],
        dropout=config['translator']['decoder']['dropout'],
        attention_type=config['translator']['decoder']['attention_type']
    )
    
    model = Seq2SeqWithAttention(encoder, decoder, device)
    print(f"Model has {model.count_parameters():,} parameters")
    
    print("\nCreating trainer...")
    trainer = TranslatorTrainer(model, dataset, device, config)
    
    print("\nStarting training...")
    trainer.train(
        num_epochs=config['translator']['epochs'],
        checkpoint_dir=config['checkpoint_dir']
    )
    
    print("\nTraining complete!")
    print(f"Best model saved to: {config['checkpoint_dir']}/translator_best.pth")

if __name__ == '__main__':
    main()
"""


# Example 4: Test All Models
# Save as: test_inference.py

"""
#!/usr/bin/env python3
import torch
from src.utils.config import load_config, get_device
from src.data.datasets import NameClassificationDataset, NameGenerationDataset
from src.inference.classifier_predictor import ClassifierPredictor
from src.inference.name_generator import NameGenerator

def test_classifier():
    print("\\n" + "="*50)
    print("Testing Classifier")
    print("="*50)
    
    config = load_config()
    dataset = NameClassificationDataset(config['data_dir'])
    
    predictor = ClassifierPredictor.from_checkpoint(
        'models/classifier_best.pth',
        dataset.all_languages
    )
    
    test_names = ['Yamamoto', 'Schmidt', 'Dubois', 'O\'Brien', 'Gonzalez']
    
    for name in test_names:
        predictions = predictor.predict(name, top_k=3)
        print(f"\\n{name}:")
        for lang, prob in predictions:
            print(f"  {lang:15s} {prob:6.2%}")

def test_generator():
    print("\\n" + "="*50)
    print("Testing Generator")
    print("="*50)
    
    config = load_config()
    dataset = NameGenerationDataset(config['data_dir'])
    
    generator = NameGenerator.from_checkpoint(
        'models/generator_best.pth',
        dataset.all_categories
    )
    
    test_languages = ['Russian', 'Japanese', 'Italian', 'Arabic']
    
    for language in test_languages:
        names = generator.generate(language, num_samples=5)
        print(f"\\n{language}:")
        for name in names:
            print(f"  {name}")

def main():
    test_classifier()
    test_generator()
    # Add translator test when ready

if __name__ == '__main__':
    main()
"""
