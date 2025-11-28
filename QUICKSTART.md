# Quick Start Guide - NLP Character RNN Suite

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Phase 1-4 Complete! ✅

The following modules are now ready:

### ✅ Phase 1: Foundation
- Project structure created
- Configuration system (`config.yaml`)
- Utility modules (config, visualization)
- Preprocessing utilities

### ✅ Phase 2: Data Layer
- `NameClassificationDataset` - For name → language classification
- `NameGenerationDataset` - For language → name generation  
- `TranslationDataset` - For French ↔ English translation

### ✅ Phase 3: Models
- `CharRNNClassifier` - LSTM-based name classifier
- `CharRNNGenerator` - Conditional name generator
- `Seq2SeqWithAttention` - Encoder-decoder with Bahdanau attention

### ✅ Phase 4: Training & Inference
- `ClassifierTrainer` - Training loop for classification
- `GeneratorTrainer` - Training loop for generation
- `TranslatorTrainer` - Training loop for translation
- Inference modules for all three tasks

---

## Training Scripts (Next Step)

You can now create training scripts to train all three models.

### Example: Train Classifier

```python
import torch
from src.utils.config import load_config, get_device, set_seed
from src.data.datasets import NameClassificationDataset
from src.models.classifier import CharRNNClassifier
from src.training.trainer import ClassifierTrainer

# Load config
config = load_config('config.yaml')
set_seed(config['seed'])
device = get_device(config['device'])

# Create dataset
dataset = NameClassificationDataset(config['data_dir'])

# Create model
model = CharRNNClassifier(
    input_size=config['classifier']['input_size'],
    hidden_size=config['classifier']['hidden_size'],
    output_size=config['classifier']['output_size'],
    num_layers=config['classifier']['num_layers'],
    dropout=config['classifier']['dropout']
)

# Create trainer
trainer = ClassifierTrainer(model, dataset, device, config)

# Train
trainer.train(
    num_epochs=config['classifier']['epochs'],
    checkpoint_dir=config['checkpoint_dir']
)
```

### Example: Train Generator

```python
from src.data.datasets import NameGenerationDataset
from src.models.generator import CharRNNGenerator
from src.training.trainer import GeneratorTrainer

# Create dataset
dataset = NameGenerationDataset(config['data_dir'])

# Create model
model = CharRNNGenerator(
    input_size=config['generator']['input_size'],
    category_size=config['generator']['category_size'],
    hidden_size=config['generator']['hidden_size'],
    output_size=config['generator']['output_size'],
    num_layers=config['generator']['num_layers'],
    dropout=config['generator']['dropout']
)

# Train
trainer = GeneratorTrainer(model, dataset, device, config)
trainer.train(
    num_iterations=config['generator']['epochs'],
    checkpoint_dir=config['checkpoint_dir']
)
```

### Example: Inference

```python
from src.inference.classifier_predictor import ClassifierPredictor

# Load trained model
predictor = ClassifierPredictor.from_checkpoint(
    'models/classifier_best.pth',
    all_languages=dataset.all_languages
)

# Make prediction
predictions = predictor.predict('Yamamoto', top_k=3)
for language, prob in predictions:
    print(f"{language}: {prob:.2%}")
```

---

## File Structure

```
src/
├── data/
│   ├── preprocessing.py      # ✅ Unicode handling, tensor conversion
│   └── datasets.py           # ✅ PyTorch Dataset classes
├── models/
│   ├── classifier.py         # ✅ CharRNNClassifier
│   ├── generator.py          # ✅ CharRNNGenerator
│   └── translator.py         # ✅ Seq2SeqWithAttention
├── training/
│   └── trainer.py            # ✅ Training loops for all tasks
├── inference/
│   ├── classifier_predictor.py  # ✅ Classification inference
│   ├── name_generator.py        # ✅ Generation inference
│   └── translator.py            # ✅ Translation inference
└── utils/
    ├── config.py             # ✅ Configuration management
    └── visualization.py      # ✅ Plotting utilities
```

---

## Configuration

All hyperparameters are in `config.yaml`:

```yaml
# Classifier settings
classifier:
  hidden_size: 256
  num_layers: 2
  dropout: 0.3
  epochs: 20
  learning_rate: 0.001

# Generator settings
generator:
  hidden_size: 256
  num_layers: 2
  dropout: 0.2
  epochs: 100000  # iterations
  learning_rate: 0.0005

# Translator settings
translator:
  encoder:
    hidden_size: 512
  decoder:
    hidden_size: 512
  epochs: 100
  learning_rate: 0.001
```

---

## Next Steps

1. **Create training scripts** (one for each task)
2. **Train models** on your data
3. **Test inference** with trained checkpoints
4. **Build Streamlit app** for interactive demo
5. **Create visualizations** and documentation

---

## Key Features Implemented

### 1. **Modular Architecture**
- Clean separation of concerns
- Reusable components
- Easy to extend

### 2. **Configuration-Driven**
- All hyperparameters in YAML
- No hardcoded values
- Easy experimentation

### 3. **Production-Ready**
- Proper error handling
- Checkpoint management
- Progress tracking
- Early stopping

### 4. **Three Complete Pipelines**
- Classification (name → language)
- Generation (language → name)
- Translation (French ↔ English)

---

## Troubleshooting

### Import Errors
If you see import errors for torch/numpy/etc:
```bash
pip install -r requirements.txt
```

### Data Not Found
Ensure your datasets are in the correct location:
```
datasets/
├── Arabic.txt
├── Chinese.txt
├── Czech.txt
└── ... (all 18 language files)
```

### Training Issues
- Check `config.yaml` for correct paths
- Verify dataset loading with small test
- Monitor GPU memory if using CUDA

---

## What's Ready

✅ **All Phase 1-4 objectives complete!**

You now have:
- Complete project structure
- Three dataset loaders
- Three model architectures
- Training infrastructure
- Inference modules
- Utility functions

**Ready for training!** 🚀
