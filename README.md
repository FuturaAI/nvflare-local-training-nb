# NVFlare Local Training Notebooks

These notebooks provide an example implementation of a binary image classification system using PyTorch. The code demonstrates how to structure a custom dataset loader and a CNN architecture, using chest X-ray classification (pneumonia/normal) as an example case.

## Available Notebooks
- `local_training.ipynb`: Local development and training notebook
- `inference.ipynb`: Model inference and testing notebook
- `wsi-torch-tutorial.ipynb`: Example tutorial from pytorch on using WSI ( this is a standalone example just for reference)

## Environment Setup

### Virtual Environment
Using virtualenvwrapper:
```bash
# Create new environment
mkvirtualenv your_env_name

# Activate environment
workon your_env_name

# Deactivate environment when done
deactivate
```

## Example Model Architecture

The sample CNN implementation includes:
- 3 convolutional blocks with increasing feature maps (32 → 64 → 128)
- Batch normalization and dropout for regularization
- Adaptive pooling and fully connected layers for classification
- Kaiming initialization for weights

This architecture serves as a starting point and should be modified based on your specific use case.

## Dataset Structure Example

The example dataset loader works with a directory structure like:
```
images/
    ├── normal/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── pneumonia/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

Note: This is just an example structure - adapt the dataset loader for your specific data organization.

## Example Usage

### Dataset Setup Example
```python
transform = transforms.Compose([
    transforms.ToTensor(),
    # Modify transforms based on your needs
])

dataset = CustomDataset(
    data_folder="images",
    transform=transform
)
```

### Model Creation Example
```python
model = CustomModel(dropout_rate=0.5)  # Adjust parameters as needed
```

### Basic Training Setup Example
```python
import torch
import torch.optim as optim

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model initialization
model = CustomModel().to(device)

# Example loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

## Requirements
- requirements.txt

## Notes
- This is a template implementation that should be modified for your specific use case
- The model architecture, hyperparameters, and transforms should be adjusted based on your data and requirements
- The example uses binary classification but can be adapted for multiclass problems
- For NVIDIA FLARE deployment, this code must be adapted to the `pt/` directory structure following the NVIDIA FLARE requirements:
  ```
  pt/
  ├── learners/      # Training logic implementation
  ├── networks/      # Model architecture
  └── utils/         # Dataset management
  ```
- I provided the `pt/` directory as an example for this specific use case