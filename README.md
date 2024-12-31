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

### Softmax Regression Results

| Batch Size | Learning Rate | Regularization | Training Time (s) | Validation Time (s) | Validation Accuracy | Test Accuracy |
|------------|---------------|----------------|--------------------|---------------------|---------------------|---------------|
| 32         | 0.001         | No             | 11.970790         | 2.118804           | 0.866083            | 0.861250      |
| 32         | 0.010         | No             | 11.016639         | 2.006012           | 0.906917            | 0.905250      |
| 32         | 0.100         | No             | 11.527780         | 2.266402           | 0.919583            | 0.920167      |
| 32         | 1.000         | No             | 10.744712         | 1.910052           | 0.880750            | 0.875583      |
| 64         | 0.001         | No             | 6.347309          | 1.404014           | 0.847167            | 0.838917      |
| 64         | 0.010         | No             | 5.929550          | 1.248466           | 0.898000            | 0.895500      |
| 64         | 0.100         | No             | 6.354043          | 1.443549           | 0.919667            | 0.918583      |
| 64         | 1.000         | No             | 5.677582          | 1.226526           | 0.896833            | 0.894417      |
| 128        | 0.001         | No             | 4.174688          | 1.017976           | 0.825333            | 0.814750      |
| 128        | 0.010         | No             | 3.859220          | 0.948523           | 0.885917            | 0.882917      |
| 128        | 0.100         | No             | 3.666445          | 0.937195           | 0.914917            | 0.914667      |
| 128        | 1.000         | No             | 4.147740          | 1.049768           | 0.913333            | 0.914083      |
| 256        | 0.001         | No             | 3.108118          | 0.776611           | 0.780250            | 0.770667      |
| 256        | 0.010         | No             | 3.092919          | 0.822761           | 0.871583            | 0.865667      |
| 256        | 0.100         | No             | 2.981443          | 0.897654           | 0.909333            | 0.908000      |
| 256        | 1.000         | No             | 3.117429          | 0.816181           | 0.922167            | 0.920000      |
| 32         | 0.001         | Yes            | 11.584298         | 2.063241           | 0.863833            | 0.859667      |
| 32         | 0.010         | Yes            | 11.194145         | 1.970026           | 0.901000            | 0.898500      |
| 32         | 0.100         | Yes            | 10.909430         | 1.937202           | 0.899333            | 0.896167      |
| 32         | 1.000         | Yes            | 10.083084         | 1.838706           | 0.734667            | 0.735417      |
| 64         | 0.001         | Yes            | 5.791924          | 1.240275           | 0.846333            | 0.837917      |
| 64         | 0.010         | Yes            | 5.761280          | 1.237508           | 0.893917            | 0.892833      |
| 64         | 0.100         | Yes            | 5.755915          | 1.201788           | 0.900167            | 0.898500      |
| 64         | 1.000         | Yes            | 5.776903          | 1.229390           | 0.872667            | 0.872833      |
| 128        | 0.001         | Yes            | 3.802248          | 0.938596           | 0.826167            | 0.818667      |
| 128        | 0.010         | Yes            | 4.166002          | 1.018829           | 0.885833            | 0.881417      |
| 128        | 0.100         | Yes            | 3.810080          | 0.927748           | 0.900167            | 0.898000      |
| 128        | 1.000         | Yes            | 3.731156          | 0.896294           | 0.811917            | 0.805083      |
| 256        | 0.001         | Yes            | 2.996189          | 0.769596           | 0.784833            | 0.782417      |
| 256        | 0.010         | Yes            | 2.885518          | 0.862792           | 0.870333            | 0.867583      |
| 256        | 0.100         | Yes            | 3.040345          | 0.782626           | 0.900000            | 0.899167      |
| 256        | 1.000         | Yes            | 3.043919          | 0.792799           | 0.882583            | 0.879583      |


### Neural Networks Results

| Batch Size | Learning Rate | Regularization | Training Time (s) | Validation Time (s) | Validation Accuracy | Test Accuracy |
|------------|---------------|----------------|--------------------|---------------------|---------------------|---------------|
| 32         | 0.001         | No             | 20.481364         | 1.765152           | 0.972917            | 0.974583      |
| 32         | 0.010         | No             | 28.303874         | 1.882041           | 0.964417            | 0.963083      |
| 32         | 0.100         | No             | 34.812692         | 1.944909           | 0.359333            | 0.363583      |
| 32         | 1.000         | No             | 37.369679         | 1.713209           | 0.097917            | 0.098083      |
| 64         | 0.001         | No             | 11.090237         | 1.349956           | 0.973000            | 0.971500      |
| 64         | 0.010         | No             | 14.394027         | 1.443001           | 0.965667            | 0.962833      |
| 64         | 0.100         | No             | 22.107260         | 1.614276           | 0.661583            | 0.668583      |
| 64         | 1.000         | No             | 25.994314         | 1.650558           | 0.106833            | 0.102833      |
| 128        | 0.001         | No             | 7.222749          | 1.098831           | 0.970500            | 0.969750      |
| 128        | 0.010         | No             | 8.503206          | 1.188966           | 0.968500            | 0.966917      |
| 128        | 0.100         | No             | 10.723768         | 1.080098           | 0.731417            | 0.720583      |
| 128        | 1.000         | No             | 12.403085         | 1.122255           | 0.097917            | 0.098083      |
| 256        | 0.001         | No             | 4.843583          | 0.988000           | 0.962833            | 0.965000      |
| 256        | 0.010         | No             | 5.152539          | 0.901508           | 0.968917            | 0.971417      |
| 256        | 0.100         | No             | 5.894685          | 1.016321           | 0.902833            | 0.901667      |
| 256        | 1.000         | No             | 6.413746          | 0.874479           | 0.102167            | 0.108667      |
| 32         | 0.001         | Yes            | 41.174098         | 6.158346           | 0.933833            | 0.930000      |
| 32         | 0.010         | Yes            | 34.151327         | 5.007726           | 0.870333            | 0.871917      |
| 32         | 0.100         | Yes            | 28.899045         | 3.924938           | 0.650417            | 0.651333      |
| 32         | 1.000         | Yes            | 34.329283         | 5.245781           | 0.148750            | 0.148667      |
| 64         | 0.001         | Yes            | 24.399306         | 4.446432           | 0.935083            | 0.933333      |
| 64         | 0.010         | Yes            | 22.127577         | 3.899560           | 0.912667            | 0.913750      |
| 64         | 0.100         | Yes            | 17.864102         | 2.498750           | 0.686417            | 0.683583      |
| 64         | 1.000         | Yes            | 22.590973         | 3.913483           | 0.289667            | 0.291250      |
| 128        | 0.001         | Yes            | 13.439845         | 2.627017           | 0.934417            | 0.933167      |
| 128        | 0.010         | Yes            | 13.530710         | 2.781530           | 0.912000            | 0.915833      |
| 128        | 0.100         | Yes            | 9.117067          | 1.581441           | 0.672167            | 0.682250      |
| 128        | 1.000         | Yes            | 12.313985         | 2.458157           | 0.275500            | 0.274250      |
| 256        | 0.001         | Yes            | 5.216988          | 0.920500           | 0.933083            | 0.931750      |
| 256        | 0.010         | Yes            | 5.025460          | 0.907063           | 0.926000            | 0.927750      |
| 256        | 0.100         | Yes            | 5.063877          | 0.989302           | 0.846667            | 0.847000      |
| 256        | 1.000         | Yes            | 5.531376          | 0.878088           | 0.105417            | 0.104667      |



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
