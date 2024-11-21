# Neural Network for Handwritten Digit Recognition

This project implements a **Feedforward Neural Network (FNN)** to recognize handwritten digits (0-9) from the MNIST dataset. The model is trained using **Stochastic Gradient Descent (SGD)** and uses customizable architecture parameters such as activation functions, number of layers, and hidden layer width.

---

## Features

- **Customizable Neural Network Architecture**:
  - Select the number of layers and width of hidden layers.
  - Choose activation functions (ReLU or Sigmoid).
  
- **Cross-Entropy with Softmax**:
  - Implements a cross-entropy loss function for multi-class classification.

- **Validation and Test Evaluation**:
  - Tracks training and validation performance at every epoch.
  - Generates predictions for test data and saves the results to a CSV file.

- **Interactive Visualizations**:
  - Plots training and validation loss/accuracy over time.

---

## Technologies Used

- **Python**: Main programming language.
- **NumPy**: Efficient numerical computation.
- **Matplotlib**: Visualization of loss and accuracy trends.

---

## Files

- `mnist_small_train.csv`: Training data (features and labels).
- `mnist_small_val.csv`: Validation data (features and labels).
- `mnist_small_test.csv`: Test data (features only).
- `test_predicted.csv`: Output file with predictions for test data.

---

## How to Use

1. **Dataset Setup**:
   - Ensure the MNIST dataset files (`mnist_small_train.csv`, `mnist_small_val.csv`, `mnist_small_test.csv`) are in the same directory.

2. **Run the Program**:
   - Execute the script: `python neural_network.py`.

3. **Inspect Training Progress**:
   - Monitor training and validation loss/accuracy printed to the console.
   - View plots of loss and accuracy trends after training.

4. **Test Predictions**:
   - The script generates `test_predicted.csv` with two columns:
     - `id`: Index of the test sample.
     - `digit`: Predicted digit (0-9).

---

## Neural Network Architecture

- **Input Layer**: 784 features (28x28 grayscale image).
- **Hidden Layers**: Configurable number and width.
- **Output Layer**: 10 neurons (one for each digit class).
- **Activation Functions**: ReLU (default) or Sigmoid.

---

## Training Parameters

- **Batch Size**: 200
- **Step Size (Learning Rate)**: 0.01
- **Maximum Epochs**: 300
- **Loss Function**: Cross-Entropy with Softmax.

---

## Key Functions

### 1. **Main Training Loop**
   - Shuffles the data and divides it into mini-batches.
   - Performs forward and backward passes for each batch.
   - Updates weights using gradient descent.
   - Evaluates validation performance at the end of each epoch.

### 2. **FeedForwardNeuralNetwork Class**
   - Constructs the neural network architecture.
   - Implements forward, backward passes, and gradient descent steps.

### 3. **CrossEntropySoftmax Class**
   - Computes cross-entropy loss and softmax probabilities.
   - Calculates gradients for backpropagation.

### 4. **LinearLayer Class**
   - Implements a fully connected layer (dense layer).
   - Supports gradient-based updates for weights and biases.

---

## Visualization

At the end of training, the program generates a plot displaying:

1. Training and validation loss over iterations.
2. Training and validation accuracy over iterations.

---

## Example Output

After training, the program outputs the following for each epoch:
  [Epoch 1] Loss: 0.4289 Train Acc: 87.54% Val Acc: 88.22%
  [Epoch 2] Loss: 0.2857 Train Acc: 91.23% Val Acc: 90.85%
  [Epoch 3] Loss: 0.2159 Train Acc: 93.21% Val Acc: 91.88%
  .
  .
  .
  [Epoch n] Loss: x Train Acc: x Val Acc: x

## Author

- **Cole Seifert**
- **Date**: November 21, 2024

---

## License

This project is licensed under the **MIT License**.
