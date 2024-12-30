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

| Batch Size | Learning Rate | Regularization | Training Time (s) | Validation Time (s) | Test Accuracy |
|------------|---------------|----------------|--------------------|---------------------|---------------|
| 32         | 0.001         | No             | 10.618336         | 1.922368           | 0.860083      |
| 32         | 0.010         | No             | 9.972970          | 1.838935           | 0.905667      |
| 32         | 0.100         | No             | 9.594610          | 1.772511           | 0.921417      |
| 32         | 1.000         | No             | 9.496401          | 1.689946           | 0.906417      |
| 64         | 0.001         | No             | 5.742435          | 1.251234           | 0.840083      |
| 64         | 0.010         | No             | 5.750075          | 1.238782           | 0.895417      |
| 64         | 0.100         | No             | 6.178684          | 1.269684           | 0.917917      |
| 64         | 1.000         | No             | 6.359141          | 1.321672           | 0.901000      |
| 128        | 0.001         | No             | 4.920842          | 1.055449           | 0.815750      |
| 128        | 0.010         | No             | 3.910599          | 0.900768           | 0.883250      |
| 128        | 0.100         | No             | 3.966460          | 0.955608           | 0.915250      |
| 128        | 1.000         | No             | 3.849463          | 0.941083           | 0.917583      |
| 256        | 0.001         | No             | 3.191471          | 0.802991           | 0.772750      |
| 256        | 0.010         | No             | 3.174954          | 0.814402           | 0.866750      |
| 256        | 0.100         | No             | 3.113983          | 0.925978           | 0.907833      |
| 256        | 1.000         | No             | 3.137086          | 0.817539           | 0.911083      |
| 32         | 0.001         | Yes            | 10.070731         | 1.817888           | 0.862167      |
| 32         | 0.010         | Yes            | 10.015956         | 1.813548           | 0.897750      |
| 32         | 0.100         | Yes            | 9.463851          | 1.667337           | 0.897167      |
| 32         | 1.000         | Yes            | 9.840020          | 1.767056           | 0.809417      |
| 64         | 0.001         | Yes            | 5.815382          | 1.268403           | 0.840083      |
| 64         | 0.010         | Yes            | 6.166116          | 1.301984           | 0.891417      |
| 64         | 0.100         | Yes            | 6.651792          | 1.342169           | 0.901667      |
| 64         | 1.000         | Yes            | 6.544258          | 1.396584           | 0.736583      |
| 128        | 0.001         | Yes            | 4.070819          | 0.945871           | 0.817833      |
| 128        | 0.010         | Yes            | 4.118722          | 1.000741           | 0.880500      |
| 128        | 0.100         | Yes            | 3.902983          | 0.953600           | 0.898583      |
| 128        | 1.000         | Yes            | 3.917112          | 0.927950           | 0.834333      |
| 256        | 0.001         | Yes            | 3.146292          | 0.786392           | 0.778833      |
| 256        | 0.010         | Yes            | 3.036348          | 0.907333           | 0.867083      |
| 256        | 0.100         | Yes            | 3.109457          | 0.836555           | 0.899000      |
| 256        | 1.000         | Yes            | 3.111300          | 0.846374           | 0.840417      |

### Neural Networks Results

| Batch Size | Learning Rate | Regularization | Training Time (s) | Validation Time (s) | Test Accuracy |
|------------|---------------|----------------|--------------------|---------------------|---------------|
| 32         | 0.001         | No             | 22.366846         | 1.870744           | 0.969083      |
| 32         | 0.010         | No             | 29.285980         | 2.093653           | 0.961750      |
| 32         | 0.100         | No             | 32.714286         | 2.087725           | 0.529250      |
| 32         | 1.000         | No             | 37.424560         | 1.931948           | 0.098000      |
| 64         | 0.001         | No             | 11.753811         | 1.434222           | 0.971500      |
| 64         | 0.010         | No             | 13.973129         | 1.448734           | 0.957583      |
| 64         | 0.100         | No             | 18.428567         | 1.410095           | 0.718583      |
| 64         | 1.000         | No             | 21.892138         | 1.417789           | 0.097833      |
| 128        | 0.001         | No             | 7.051995          | 1.087158           | 0.967917      |
| 128        | 0.010         | No             | 7.832273          | 1.129741           | 0.969917      |
| 128        | 0.100         | No             | 10.442930         | 1.099069           | 0.845833      |
| 128        | 1.000         | No             | 12.025454         | 1.089711           | 0.096667      |
| 256        | 0.001         | No             | 4.725859          | 0.980815           | 0.966333      |
| 256        | 0.010         | No             | 4.901381          | 0.858181           | 0.967583      |
| 256        | 0.100         | No             | 6.243006          | 0.922942           | 0.903417      |
| 256        | 1.000         | No             | 6.412262          | 0.895592           | 0.154583      |
| 32         | 0.001         | Yes            | 39.848149         | 6.240499           | 0.932917      |
| 32         | 0.010         | Yes            | 33.586327         | 4.817635           | 0.896250      |
| 32         | 0.100         | Yes            | 29.473652         | 3.846096           | 0.622667      |
| 32         | 1.000         | Yes            | 36.194433         | 5.557230           | 0.219000      |
| 64         | 0.001         | Yes            | 25.141231         | 4.667979           | 0.933250      |
| 64         | 0.010         | Yes            | 21.683671         | 3.937542           | 0.910333      |
| 64         | 0.100         | Yes            | 14.739898         | 2.339227           | 0.590250      |
| 64         | 1.000         | Yes            | 20.594732         | 3.723285           | 0.436083      |
| 128        | 0.001         | Yes            | 13.013378         | 2.699625           | 0.930833      |
| 128        | 0.010         | Yes            | 12.963231         | 2.650190           | 0.913583      |
| 128        | 0.100         | Yes            | 8.852022          | 1.500548           | 0.660333      |
| 128        | 1.000         | Yes            | 11.971239         | 2.518043           | 0.304917      |
| 256        | 0.001         | Yes            | 4.993166          | 0.915272           | 0.932583      |
| 256        | 0.010         | Yes            | 4.948658          | 0.875647           | 0.926000      |
| 256        | 0.100         | Yes            | 4.996486          | 0.984093           | 0.883583      |
| 256        | 1.000         | Yes            | 5.488192          | 0.890394           | 0.480000      |


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
