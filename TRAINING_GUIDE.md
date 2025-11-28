# Training on GPU-Enabled PC

## Quick Setup on New Machine

### 1. Clone Repository
```bash
git clone https://github.com/imranpollob/predict-language-of-a-name.git
cd predict-language-of-a-name
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify CUDA (Optional but Recommended)
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

---

## Training Scripts

### Create Training Scripts from Templates

Copy the code from `TRAINING_TEMPLATES.py` to create:

#### 1. `train_classifier.py`
```python
#!/usr/bin/env python3
import torch
from src.utils.config import load_config, get_device, set_seed
from src.data.datasets import NameClassificationDataset
from src.models.classifier import CharRNNClassifier
from src.training.trainer import ClassifierTrainer

def main():
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
```

#### 2. `train_generator.py`
```python
#!/usr/bin/env python3
import torch
from src.utils.config import load_config, get_device, set_seed
from src.data.datasets import NameGenerationDataset
from src.models.generator import CharRNNGenerator
from src.training.trainer import GeneratorTrainer

def main():
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
```

---

## Run Training

### Train Classifier (Fastest - ~10-15 min on GPU)
```bash
python train_classifier.py
```

Expected output:
- Training progress with tqdm progress bars
- Validation metrics after each epoch
- Best model saved to `models/classifier_best.pth`
- Target accuracy: >85%

### Train Generator (Medium - ~30-45 min on GPU)
```bash
python train_generator.py
```

Expected output:
- Progress every 5000 iterations
- Loss tracking
- Best model saved to `models/generator_best.pth`
- Generates diverse, valid names

### Optional: Train Translator (Longest - requires translation data)
Note: You'll need to download French-English translation data first.

---

## Quick Test After Training

### Test Classifier
```python
from src.inference.classifier_predictor import ClassifierPredictor
from src.data.datasets import NameClassificationDataset

dataset = NameClassificationDataset('datasets')
predictor = ClassifierPredictor.from_checkpoint(
    'models/classifier_best.pth',
    dataset.all_languages
)

# Test some names
test_names = ['Yamamoto', 'Schmidt', 'Dubois', 'Garcia', 'O\'Brien']
for name in test_names:
    predictions = predictor.predict(name, top_k=3)
    print(f"\n{name}:")
    for lang, prob in predictions:
        print(f"  {lang:15s} {prob:6.2%}")
```

### Test Generator
```python
from src.inference.name_generator import NameGenerator
from src.data.datasets import NameGenerationDataset

dataset = NameGenerationDataset('datasets')
generator = NameGenerator.from_checkpoint(
    'models/generator_best.pth',
    dataset.all_categories
)

# Generate names for different languages
for language in ['Russian', 'Japanese', 'Italian', 'Arabic', 'Irish']:
    names = generator.generate(language, num_samples=5)
    print(f"\n{language}: {', '.join(names)}")
```

---

## Troubleshooting

### CUDA Out of Memory
If you get OOM errors, reduce batch size in `config.yaml`:
```yaml
batch_size: 32  # Reduce from 64
```

### Import Errors
Make sure you're in the project root directory:
```bash
cd predict-language-of-a-name
python train_classifier.py
```

### Dataset Not Found
Verify the `datasets/` folder contains all 18 .txt files:
```bash
ls datasets/*.txt
```

---

## Expected Training Times (on GPU)

| Model      | Epochs/Iterations | Time (GPU) | Time (CPU) |
| ---------- | ----------------- | ---------- | ---------- |
| Classifier | 20 epochs         | 10-15 min  | 1-2 hours  |
| Generator  | 100K iterations   | 30-45 min  | 3-4 hours  |
| Translator | 100 epochs        | 1-2 hours  | 8-10 hours |

---

## After Training

### 1. Commit Trained Models
```bash
git add models/*.pth
git commit -m "Add trained model checkpoints"
git push
```

### 2. Document Results
Update `PLAN.md` with:
- ✅ Classification accuracy achieved
- ✅ Sample generated names
- ✅ Training times

### 3. Next Steps
- Build Streamlit app (Phase 5)
- Create demo GIF/video
- Write portfolio README

---

## Configuration Tips

### For Faster Training (Lower Accuracy)
```yaml
classifier:
  epochs: 10
  hidden_size: 128

generator:
  epochs: 50000
  hidden_size: 128
```

### For Better Accuracy (Slower Training)
```yaml
classifier:
  epochs: 30
  hidden_size: 512
  num_layers: 3

generator:
  epochs: 200000
  hidden_size: 512
```

---

## Good Luck! 🚀

The infrastructure is solid. Training should be straightforward. If you encounter any issues, check:
1. ✅ Dependencies installed
2. ✅ CUDA available
3. ✅ In correct directory
4. ✅ Dataset files present

All the hard work is done - now just let the GPU do its magic! 💪
