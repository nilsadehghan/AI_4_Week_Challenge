
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------- Data Preparation -------------------
# Define transformations: convert images to tensors and normalize them
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))      # Normalize grayscale image to mean=0.5, std=0.5
])

# Load MNIST dataset (train & test sets)
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create DataLoaders for batch processing
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# ------------------- Define the CNN Model -------------------
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Conv layer 1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Conv layer 2
        self.pool = nn.MaxPool2d(kernel_size=2)                  # MaxPooling
        self.fc1 = nn.Linear(64 * 7 * 7, 128)                     # Fully connected layer
        self.fc2 = nn.Linear(128, 10)                             # Output layer for 10 classes
        self.dropout = nn.Dropout(0.25)                           # Dropout to reduce overfitting

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))                     # Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))                     # Conv2 -> ReLU -> Pool
        x = x.view(-1, 64 * 7 * 7)                                # Flatten
        x = F.relu(self.fc1(x))                                  # FC1 -> ReLU
        x = self.dropout(x)                                      # Apply Dropout
        x = self.fc2(x)                                          # Output logits
        return x

# ------------------- Training Setup -------------------
model = CNNNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ------------------- Early Stopping Setup -------------------
patience = 3
best_val_loss = np.inf
epochs_no_improve = 0
early_stop = False
num_epochs = 10
save_path = "best_model.pth"

# ------------------- Training Loop with Early Stopping -------------------
training_loss = []
validation_loss = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Average training loss for this epoch
    epoch_train_loss = running_loss / len(train_loader)
    training_loss.append(epoch_train_loss)

    # ------------------- Validation Phase -------------------
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    # Average validation loss
    val_loss /= len(test_loader)
    validation_loss.append(val_loss)

    # Print epoch results
    print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {epoch_train_loss:.4f} - Validation Loss: {val_loss:.4f}")

    # Check for improvement
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), save_path)
        print(" Validation loss improved. Model saved.")
    else:
        epochs_no_improve += 1
        print(f" No improvement. Patience: {epochs_no_improve}/{patience}")
        if epochs_no_improve >= patience:
            print(" Early stopping triggered.")
            early_stop = True
            break

# ------------------- Plot Loss Curves -------------------
plt.plot(training_loss, label='Training Loss', color='pink')
plt.plot(validation_loss, label='Validation Loss', color='black')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.show()

# ------------------- Load Best Model -------------------
model.load_state_dict(torch.load(save_path))
print("Best model loaded from disk.")

# ------------------- Evaluation: Confusion Matrix & Accuracy -------------------
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
sns.heatmap(cm, annot=True, cmap='coolwarm', fmt='d')
plt.title(f'Confusion Matrix (Accuracy: {accuracy * 100:.2f}%)')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
