# Clothing-Identifier
A simple neural network implementation for classifying FashionMNIST images using PyTorch.
## Project Overview
This project implements a feedforward neural network to classify images from the FashionMNIST dataset. The model consists of two linear layers with ReLU activation and achieves reasonable accuracy on this fashion product classification task.
## Features
- Data loading and preprocessing pipeline
- Neural network model implementation
- Training loop with accuracy/loss tracking
- Model evaluation functionality
- Prediction with top-3 class probabilities
## Installation
Clone this repository:
```bash
git clone <repo-url>
cd Clothing-Identifier
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```
## Usage
The project contains several functions that can be used independently:

### Loading Data
```python
train_loader = get_data_loader(training=True)
test_loader = get_data_loader(training=False)
```
### Building and Training the Model
```python
model = build_model()
criterion = nn.CrossEntropyLoss()
train_model(model, train_loader, criterion, T=5)  # Train for 5 epochs
```
### Evaluating the Model
```python
evaluate_model(model, test_loader, criterion, show_loss=True)
```
### Making Predictions
```python
# Assuming test_images contains your test data
predict_label(model, test_images, index=0)  # Predict label for first test image
```
### Running the Full Pipeline
Simply execute the Python script to run the complete training and evaluation process:

```bash
python fashion_mnist_classifier.py
```
