import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score

# Regression Model (Steer, Accel, Brake)
class RegressionModel(nn.Module):
    def __init__(self, input_size=25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)  # Steer, Accel, Brake
        )
    
    def forward(self, x):
        return torch.tanh(self.net(x))  # Outputs in [-1, 1]

# Classification Model (Gear_output)
class ClassificationModel(nn.Module):
    def __init__(self, input_size=25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 8)  # 8 classes for Gear_output (-1, 0, 1, 2, 3, 4, 5, 6)
        )
    
    def forward(self, x):
        return self.net(x)  # Logits for softmaximport torch

# Load preprocessed data
X_train = np.load("X_train.npy")
y_reg_train = np.load("y_reg_train.npy")  # Steer, Accel, Brake
y_cls_train = np.load("y_cls_train.npy")  # Gear_output (one-hot)
X_val = np.load("X_val.npy")
y_reg_val = np.load("y_reg_val.npy")
y_cls_val = np.load("y_cls_val.npy")

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_reg_train_tensor = torch.tensor(y_reg_train, dtype=torch.float32)
y_cls_train_tensor = torch.tensor(y_cls_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_reg_val_tensor = torch.tensor(y_reg_val, dtype=torch.float32)
y_cls_val_tensor = torch.tensor(y_cls_val, dtype=torch.float32)

# Initialize models
reg_model = RegressionModel(input_size=25)
cls_model = ClassificationModel(input_size=25)

# Loss and optimizers
reg_criterion = nn.MSELoss()
cls_criterion = nn.CrossEntropyLoss()
reg_optimizer = optim.Adam(reg_model.parameters(), lr=0.001)
cls_optimizer = optim.Adam(cls_model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
batch_size = 128
for epoch in range(num_epochs):
    reg_model.train()
    cls_model.train()
    
    # Mini-batch training
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train_tensor[i:i+batch_size]
        batch_y_reg = y_reg_train_tensor[i:i+batch_size]
        batch_y_cls = y_cls_train_tensor[i:i+batch_size]
        
        # Regression
        reg_optimizer.zero_grad()
        reg_outputs = reg_model(batch_X)
        reg_loss = reg_criterion(reg_outputs, batch_y_reg)
        reg_loss.backward()
        reg_optimizer.step()
        
        # Classification
        cls_optimizer.zero_grad()
        cls_outputs = cls_model(batch_X)
        cls_loss = cls_criterion(cls_outputs, torch.argmax(batch_y_cls, dim=1))
        cls_loss.backward()
        cls_optimizer.step()
    
    # Validation
    reg_model.eval()
    cls_model.eval()
    with torch.no_grad():
        reg_val_outputs = reg_model(X_val_tensor)
        reg_val_loss = reg_criterion(reg_val_outputs, y_reg_val_tensor)
        
        cls_val_outputs = cls_model(X_val_tensor)
        cls_val_loss = cls_criterion(cls_val_outputs, torch.argmax(y_cls_val_tensor, dim=1))
        cls_val_preds = torch.argmax(cls_val_outputs, dim=1).numpy()
        cls_val_true = torch.argmax(y_cls_val_tensor, dim=1).numpy()
        cls_val_accuracy = accuracy_score(cls_val_true, cls_val_preds)
    
    if epoch % 1 == 0:
        print(f"Epoch {epoch}:")
        print(f"  Regression Val Loss: {reg_val_loss.item():.4f}")
        print(f"  Classification Val Loss: {cls_val_loss.item():.4f}, Accuracy: {cls_val_accuracy:.4f}")

# Save models
torch.save(reg_model.state_dict(), "torcs_reg_controller.pth")
torch.save(cls_model.state_dict(), "torcs_cls_controller.pth")