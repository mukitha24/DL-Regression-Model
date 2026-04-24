# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
<img width="851" height="815" alt="Screenshot 2026-04-24 082601" src="https://github.com/user-attachments/assets/7f0d862a-758c-4d0b-9126-83dafb137bf1" />


## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: Mukitha V M

### Register Number:21223040119

```
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,8)
        self.fc2 = nn.Linear(8,10)
        self.fc3 = nn.Linear(10,1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

  def forward(self,x):
      x = self.relu(self.fc1(x))
      x = self.relu(self.fc2(x))
      x = self.fc3(x)
      return x



# Initialize the Model, Loss Function, and Optimizer
lig = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(lig. parameters(), lr=0.001)

def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    # Write your code here
    for epoch in range(epochs) :
      optimizer. zero_grad()
      loss = criterion(ai_brain(X_train), y_train)
      loss.backward()
      optimizer.step()
      lig.history['loss'].append(loss.item())
      if epoch % 200 == 0:
          print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

train_model(lig, X_train_tensor, y_train_tensor, criterion, optimizer)

with torch.no_grad():
    test_loss = criterion(lig(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

loss_df = pd.DataFrame(lig.history)

import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = lig(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')
```

### Dataset Information
<img width="224" height="601" alt="Screenshot 2026-04-24 082206" src="https://github.com/user-attachments/assets/2b8cdf71-96fb-486b-87d8-50eb26f0ecc9" />


### OUTPUT
Training Loss Vs Iteration Plot
<img width="406" height="233" alt="Screenshot 2026-04-24 082302" src="https://github.com/user-attachments/assets/59b45565-d8f6-4060-be98-d97240c43cad" />

<img width="829" height="582" alt="image" src="https://github.com/user-attachments/assets/052e6f68-ad9d-4b14-9aec-7e213fd92492" />


### New Sample Data Prediction
<img width="348" height="34" alt="Screenshot 2026-04-24 082410" src="https://github.com/user-attachments/assets/6af284c9-6410-47c8-938b-ab1d87eafa5f" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
