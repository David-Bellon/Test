
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import seaborn as sns
import numpy as np
from sklearn.preprocessing import PowerTransformer


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

df = pd.read_excel(r"C:\Users\dadbc\Desktop\Phy\Repositorios\Test\regression_data.xls")

df = df.drop(columns=["id", "date"])




X = df.drop(columns=["price"])
Y = df["price"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.3, random_state= 42)

pt = PowerTransformer()
pt.fit(X_train)

X_train_scaled = pt.transform(X_train)
X_test_scaled = pt.transform(X_test)
Y_train = np.log(Y_train)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

X_train_scaled = torch.FloatTensor(X_train_scaled.values)
X_test_scaled = torch.FloatTensor(X_test_scaled.values)
Y_train= torch.FloatTensor(Y_train.values)
Y_test = torch.FloatTensor(Y_test.values)



class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.Linear(18, 18)
        self.n2 = nn.Linear(18, 18)
        self.n4 = nn.Linear(18, 18)
        self.output = nn.Linear(18, 1)
    
    def forward(self, x):
        x = F.relu(self.n1(x))
        x = F.relu(self.n2(x))
        x = F.relu(self.n4(x))
        x = self.output(x)
        return x
Red = NeuralNetwork()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(Red.parameters(), lr = 0.001)

epochs = 10000
loss_f = []

for i in range(epochs):
    optimizer.zero_grad()
    y_pred = Red.forward(X_train_scaled)
    y_pred = torch.reshape(y_pred, (y_pred.shape[0],))
    lost = criterion(y_pred, Y_train)
    
    
    loss_f.append(lost)

    #if i % 1200 == 0 and i > 1:
        #optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] / 1.2


    if i % 10 == 0:
        print("Iteracion:", i, "loss",  lost, "lr", optimizer.param_groups[0]["lr"])

    
    
    
    lost.backward()
    optimizer.step()
    

    


prediction = []
with torch.no_grad():
    for val in X_test_scaled:
        pred = Red.forward(val)
        pred = np.exp(pred)
        prediction.append(pred.item())

dev = []
Y_test = list(Y_test)
for i in range(len(prediction)):
    dev.append((prediction[i] - Y_test[i]).item())


dev = pd.DataFrame(dev, columns=["Deviation"])

print(dev.describe())





if dev.max().item() <= 4e+06:
    plt.figure(figsize=(15,15))
    plt.axvline(x = 100000, color = "red")
    plt.axvline(x = -100000, color = "red")
    sns.histplot(dev)
    plt.show()

from sklearn.metrics import r2_score
print(r2_score(prediction, Y_test))