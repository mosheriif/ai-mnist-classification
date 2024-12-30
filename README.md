# MNIST Classification and Neural Network Modelling

This repository contains a project developed for the (CSE351) Introduction to Artificial Intelligence course. It contains a Python notebook implementing classification models for the MNIST dataset using PyTorch. It includes Softmax Regression and a Feedforward Neural Network. The goal is to classify handwritten digits (0-9) and evaluate the performance of these models.

## Table of Contents

1. **Data Preprocessing**
   - MNIST dataset loading and preprocessing (normalization, splitting).
2. **Softmax Regression**
   - Implementation, training, and evaluation.
   - Parameter tuning: Learning rates, batch sizes, and L2 regularization.
3. **Feedforward Neural Network**
   - Model implementation, training, and evaluation.
4. **Performance Analysis**
   - Metrics for comparison include accuracy, training time, and confusion matrix visualization.

## Results

| Model                        | Final Test Accuracy | Training Time (seconds) |
|------------------------------|---------------------|--------------------------|
| Softmax Regression           | 0.85               | 50                       |
| Feedforward Neural Network   | 0.92               | 200                      |


## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/mosheriif/ai-mnist-classification.git
   cd ai-mnist-classification
   ```

2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib scikit-learn torch torchvision idx2numpy
   ```

3. Run the notebook:
   Use any Jupyter notebook environment to execute the notebook.

## Visualizations

- Training and validation loss/accuracy plots.
- Confusion matrix heatmap for model predictions.

## License

This project is licensed under the MIT License.
