import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np


def train_model(model, X_train_tensor, y_train_tensor):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Training loop
    num_epochs = 100
    train_losses = []
    train_accuracies = []

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += batch_y.size(0)
            correct_predictions += (predicted == batch_y).sum().item()
        
        # Calculate average loss and accuracy for the epoch
        avg_loss = epoch_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

    print("Training completed!")
