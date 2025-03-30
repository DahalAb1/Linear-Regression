import torch
from torch import nn 
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias


#I am nto sure of how the data was separated 
train_split = int(0.8 * len(X)) 
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]


def plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=None):
  plt.figure(figsize=(10, 7))
  plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
  plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
  if predictions is not None:
    plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
  plt.legend(prop={"size": 14})
  plt.show()

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__() 

        self.weights = nn.Parameter(torch.rand(1), requires_grad=True) 
        self.bias = nn.Parameter(torch.rand(1), requires_grad=True) 

    def forward(self, x): 
        return self.weights * x + self.bias 
    



#seeding for a consistent model output 
torch.manual_seed(42)

model_0 = LinearRegressionModel()
lossFunction = nn.L1Loss()
optimizer = torch.optim.SGD(model_0.parameters(),lr=0.01)


epocss = 100 

epocs = []
loss_training_data = []
loss_testing_data = []


for epoc in range(epocss):

#training the model
   #start the model 
   model_0.train()

   #forward pass, predict activate the forward pass method of the model 
   y_trained = model_0(X_train)

   #calculate the loss 
   loss = lossFunction(y_trained,y_train)

   #zero grad (idk why we do this )
   optimizer.zero_grad()

   #perform back propagation 
   loss.backward()

   #update back propagated values 
   optimizer.step()

#testing the mode 
   model_0.eval()

   with torch.inference_mode():
      y_testing = model_0(X_test)
  
   #calculating the loss 
   test_loss  = lossFunction(y_testing,y_test)

   #analyze the data 
   if epoc % 10 == 0: 
      epocs.append(epoc)
      loss_training_data.append(loss.detach().numpy())
      loss_testing_data.append(test_loss.detach().numpy())
      # print(f'\n model_parameters: {model_0.state_dict()} \n Training_Loss: {loss} \n Testing_Loss: {test_loss}')


def showLoss():
   plt.plot(loss_training_data, epocs, c="r", label="Training Data")
   plt.plot(loss_testing_data, epocs,c='g',label="Testing Data")
   plt.legend(prop={"size": 14})
   plt.show()

showLoss()


def showPrediction():
   with torch.inference_mode():
      y_pred = model_0(X_test)
   plot_predictions(predictions=y_pred) 
   plt.show()


def saveModel():
   MODEL_PATH = Path('Model')
   MODEL_PATH.mkdir(
      #if there's no parent file, say model it will create it 
      parents=True,
      #prevents any error if file is already created 
      exist_ok=True
      )

   Model_0_Name = 'Model_0.pth'
   MODEL_SAVE_PATH = MODEL_PATH/Model_0_Name
   torch.save(obj=model_0.state_dict(),f=MODEL_SAVE_PATH)

