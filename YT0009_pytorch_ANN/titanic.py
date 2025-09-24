r"""
# create a virtual environment:
python -m venv venv 

# activate the virtual environment:
source venv/bin/activate
or
venv\Scripts\activate

# install the required packages:
pip install pandas torch scikit-learn matplotlib seaborn
"""

# %% 1. Import necessary libraries
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# %% 2. Load and preprocess the Titanic dataset
description = r"""
| Column       | Description                                            |
| ------------ | ------------------------------------------------------ |
| survived     | 0 = passenger died, 1 = passenger survived             |
| pclass       | ticket class (1, 2, 3)                                 |
| sex          | passenger gender                                       |
| age          | age (may be `NaN` for missing values)                  |
| sibsp        | number of siblings/spouses aboard                      |
| parch        | number of parents/children aboard                      |
| fare         | ticket fare                                            |
| embarked     | port of embarkation (C, Q, S)                          |
| class        | ticket class description ("First", "Second", "Third")  |
| who          | passenger category (man, woman, child)                 |
| adult_male   | True/False - is the passenger an adult male            |
| deck         | deck (often missing values)                            |
| embark_town  | name of embarkation port                               |
| alive        | "yes"/"no" - did the passenger survive                 |
| alone        | True/False - was the passenger traveling alone         |
"""
df = sns.load_dataset('titanic')
df = df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'who']]

df['age'] = df['age'].fillna(df['age'].mean())
df['sex'] = df['sex'].map({'male':0,'female':1})
df = pd.get_dummies(df, columns=['embarked','who'], dummy_na=True)
for col in df.columns:
    if col.startswith('who_') or col.startswith('embarked_'):
        df[col] = df[col].astype(int)

X = df.drop(columns='survived').values
y = df['survived'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Dataset i DataLoader
# %%
class TitanicDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = TitanicDataset(X_train_scaled, y_train)
test_ds = TitanicDataset(X_test_scaled, y_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

# 4. Model definition and initialization
class TitanicModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, 16)
        self.l2 = nn.Linear(16, 8)
        self.l3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        x = self.sigmoid(x)
        return x

model = TitanicModel(input_size=X_train_scaled.shape[1])

# 5. Set loss function and optimizer
loss_func = nn.BCELoss() # Binary Cross Entropy Loss for binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Adam optimizer - adaptive learning rate

# 6. Model trening
def train_epoch(model, loader):
    model.train()
    total_loss = 0
    correct = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        preds = model(xb).squeeze()   
        predicted = (preds >= 0.5).float()
        correct += (predicted == yb).sum().item()
        loss = loss_func(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    aver_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return aver_loss, accuracy

# 7. Model evaluation
def eval_model(model, loader):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb).squeeze()  
            predicted = (preds >= 0.5).float()
            correct += (predicted == yb).sum().item()
            loss = loss_func(preds, yb)
            total_loss += loss.item() * xb.size(0)
    aver_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return aver_loss, accuracy

# %% 8. Run training
losses = []
accuracies = []
epochs = 500
for epoch in range(epochs):
    train_loss, train_acc = train_epoch(model, train_loader)
    losses.append(train_loss)
    accuracies.append(train_acc)
    print(f"Epoch {epoch+1}: Train loss={train_loss:.4f}, Train acc={train_acc:.4f}")

# %% 9. Plot training loss and accuracy
fig, ax1 = plt.subplots()

# Loss 
ax1.plot(range(len(losses)), losses, 'b-', label='Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Accuracy 
ax2 = ax1.twinx()
ax2.plot(range(len(losses)), accuracies, 'g-', label='Accuracy')
ax2.set_ylabel('Accuracy', color='g')
ax2.tick_params(axis='y', labelcolor='g')

plt.title('Training Loss and Accuracy')
fig.tight_layout()
plt.show()


# %% 10. Run evaluation
test_loss, test_acc = eval_model(model, test_loader)
base_acc = max((y_test == 0).mean(), (y_test == 1).mean()) 
print(f"Test acc={test_acc:.4f}, baseline acc={base_acc:.4f}")


# %% 11. Predict for a random test passenger
import random
rand_idx = random.randint(0, X_test_scaled.shape[0] - 1)

# Get data for this passenger
single_passenger = X_test_scaled[rand_idx]
single_passenger_tensor = torch.tensor(single_passenger, dtype=torch.float32).unsqueeze(0)

# True label
true_label = y_test[rand_idx]

# Make prediction
model.eval()
with torch.no_grad():
    prediction = model(single_passenger_tensor).item()
    survived = int(prediction >= 0.5)
    print(f"Survival probability: {prediction:.2f}")
    print(f"Model predicts: {'Survived' if survived else 'Did not survive'}")
    print(f"Actual: {'Survived' if true_label == 1 else 'Did not survive'}")