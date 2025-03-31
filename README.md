### PyTorch Simple Linear Regression

This is my first PyTorch project—a basic linear regression model built from scratch. As part of my journey to learn PyTorch, I implemented this to understand the fundamentals of machine learning frameworks. The model predicts a continuous output based on a single input by fitting a straight line to the data.


---

## How It Works
This project implements a simple linear regression model using PyTorch. I’ll break down each part of the code step-by-step, showing the actual code snippets and explaining what they do, why they’re important, and how they fit into the bigger picture. If you know the basic concepts of Machine Learning, this can be a really strong guide to deepen your understanding. 

# 1.Generating Synthetic Data
The first step is to create synthetic data that follows a linear relationship. This data will be used to train and test the model


```python
import torch

# Define the true parameters of the linear relationship
weight = 0.7
bias = 0.3

# Generate X values and compute corresponding y values
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)  # Shape: (50, 1)
y = weight * X + bias  # Shape: (50, 1)

# Split the data into training and testing sets (80/20 split)
train_split = int(0.8 * len(X))  # 80% of 50 = 40
X_train, y_train = X[:train_split], y[:train_split]  # First 40 samples
X_test, y_test = X[train_split:], y[train_split:]    # Last 10 samples
```

**Explanation:

a. `torch.arange`: Creates a tensor of evenly spaced values from 0 to 1 with a step size of 0.02. The .unsqueeze(dim=1) adds an extra dimension, making it a column vector (shape (50, 1) instead of (50,)), which is necessary for PyTorch’s matrix operations.

b Linear equation: The `y = weight * X + bias` line simulates a perfect linear relationship where `weight = 0.7` and `bias = 0.3`. This gives us ground-truth data to test whether the model can learn these parameters.

c. Why synthetic data?: It’s ideal for educational purposes because we know the true relationship, allowing us to verify the model’s accuracy.

d. Train-test split: The data is split into 80% training (40 samples) and 20% testing (10 samples). This ensures we can train the model on one portion and evaluate its generalization on unseen data.**



# 2. Defining the Model
Next, I define the linear regression model by creating a custom class that inherits from PyTorch’s nn.Module.

```python
from torch import nn

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define learnable parameters with random initial values
        self.weights = nn.Parameter(torch.rand(1), requires_grad=True)
        self.bias = nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self, x):
        # Compute the output: y = weight * x + bias
        return self.weights * x + self.bias
```

**Explanation:

a. `nn.Module`: This is PyTorch’s base class for all neural networks. By subclassing it, we can leverage PyTorch’s built-in functionality like gradient computation.

b. `__init__` method: Initializes the model’s parameters—weights and bias—as nn.Parameter objects. These are randomly initialized with `torch.rand(1)` (a single value) and marked with requires_grad=True so PyTorch tracks their gradients during training.

c. forward method: Defines how input x is transformed into output. For linear regression, this is the equation `y = weight * x + bias`. This method is automatically called when we pass data through the model.


# 3. Training the Model
Training involves optimizing the model’s parameters to minimize the error between its predictions and the true values.
```python
# Set random seed for reproducibility
torch.manual_seed(42)

# Create an instance of the model
model_0 = LinearRegressionModel()

# Define the loss function and optimizer
lossFunction = nn.L1Loss()  # Mean Absolute Error
optimizer = torch.optim.SGD(model_0.parameters(), lr=0.01)  # Stochastic Gradient Descent

# Training loop
epochs = 100
for epoch in range(epochs):
    # Set the model to training mode
    model_0.train()
    
    # Forward pass: Make predictions
    y_trained = model_0(X_train)
    
    # Compute the loss
    loss = lossFunction(y_trained, y_train)
    
    # Zero out previous gradients
    optimizer.zero_grad()
    
    # Backward pass: Compute gradients
    loss.backward()
    
    # Update model parameters
    optimizer.step()
    
    # Optional: Print progress every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
```

**Explanation:**

a. **`torch.manual_seed(42)`**: Ensures the random initialization of weights and bias is reproducible across runs.
b. **`nn.L1Loss`**: This loss function calculates the mean absolute error (MAE) between predictions and true values, which is straightforward and effective for regression tasks.
c. **`torch.optim.SGD`**: Stochastic Gradient Descent adjusts the parameters (weights and bias) based on gradients. The learning rate (`lr=0.01`) controls the step size of these updates.
d. **Training loop**:
  i. **`model_0.train()`**: Activates training mode (not critical here but a good habit for models with layers like dropout).
  ii. **Forward pass**: `y_trained = model_0(X_train)` computes predictions for the training data.
  iii. **Loss**: Compares predictions (`y_trained`) to true values (`y_train`) using MAE.
  iv. **`optimizer.zero_grad()`**: Clears old gradients to avoid interference from previous iterations.
  v. **`loss.backward()`**: Computes the gradients of the loss with respect to weights and bias via backpropagation.
  vi **`optimizer.step()`**: Updates the parameters using the gradients and the learning rate.


 # 4. Evaluating the Model
After training, I evaluate the model on the test data to assess its performance.


```python
# Set the model to evaluation mode
model_0.eval()

# Disable gradient computation for inference
with torch.inference_mode():
    y_testing = model_0(X_test)  # Predictions on test data
    test_loss = lossFunction(y_testing, y_test)  # Test loss
    print(f"Test Loss: {test_loss.item():.4f}")
Explanation:
```
a. `model_0.eval()`: Switches the model to evaluation mode. While it doesn’t affect this simple model, it’s essential for models with layers like batch normalization or dropout.
b. `torch.inference_mode()`: Turns off gradient tracking, reducing memory usage and speeding up inference since we don’t need gradients during evaluation.
c. Test predictions: `y_testing` contains the model’s predictions for `X_test`.
d. Test loss: Measures how well the model performs on unseen data, giving insight into its generalization ability.


# 5. Saving the Model
Finally, I save the trained model so it can be reused later without retraining. We only save state_dict of a model to avoid any complications to the model if used in another system. 

```python
from pathlib import Path

# Define the save path
MODEL_PATH = Path("Model")
MODEL_PATH.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn’t exist
Model_0_Name = "Model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / Model_0_Name

# Save the model’s state dictionary
torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
```


Explanation:

a. `Path` from `pathlib`: A convenient way to handle file paths across operating systems.
b. `MODEL_PATH.mkdir`: Creates a "Model" directory if it doesn’t already exist. `parents=True` ensures nested directories are created if needed, and `exist_ok=True` prevents errors if the directory already exists.
c. `state_dict`: This is a dictionary containing the model’s learned parameters (weights and bias). Saving it instead of the entire model is lightweight and standard practice in PyTorch.
d. `torch.save`: Writes the state_dict to a `.pth` file, allowing the model to be reloaded later with `torch.load`.



## ** How to run 

1. Clone the repo
`git clone https://github.com/your-username/pytorch-simple-linear-regression.git`

2. Run the script
`python MODEL_0.py`


**NOTE : dependencies(matplotlib, numpy) and Python must be installed before running.


# Loading a Saved Model
If you already have a saved model file (e.g., Model_0.pth), you can load it instead of retraining. Here’s how:
```python
# Instantiate the model
model_saved = LinearRegressionModel()

# Load the saved parameters
model_saved.load_state_dict(torch.load(f='Model/Model_0.pth'))

# Set to evaluation mode for inference
model_saved.eval()
```
**Explanation:**

1. First, create an instance of LinearRegressionModel() to define the model structure.
2. Then, use load_state_dict(torch.load('filepath')) to load the saved weights and bias from Model_0.pth into the model.
3. Replace 'Model/Model_0.pth' with the actual path to your saved file if it’s different.
4. Finally, model_saved.eval() prepares the model for predictions without training. You can now use model_saved(X_test).



### Learning Outcomes
**Mastered PyTorch basics: tensors, autograd, and nn.Module.
Implemented a full ML workflow: data prep, model definition, training, evaluation, and visualization.
Gained confidence in debugging and experimenting with PyTorch.**



