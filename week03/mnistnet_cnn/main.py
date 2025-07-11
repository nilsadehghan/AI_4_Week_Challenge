# Model input:
# A grayscale image (28x28 pixels) depicting a handwritten digit (for example, the digit 3).
#
# What the model does:
# The model looks at this image and tries to predict which digit from 0 to 9 it represents







import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix

# ------------------- Data Preparation -------------------
# Define transformations: convert images to tensors and normalize them
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize to mean=0.5, std=0.5 for grayscale images
])

# Load MNIST training and testing datasets
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders for batch processing
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)  # Shuffle training data
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)   # Do not shuffle test data

# ------------------- Define the CNN Model -------------------
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        # First convolutional layer: 1 input channel (grayscale), 32 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Second convolutional layer: 32 input channels, 64 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Max pooling layer with 2x2 window
        self.pool = nn.MaxPool2d(kernel_size=2)
        # Fully connected layer: flatten 64 x 7 x 7 feature maps to 128 neurons
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # Output layer: 128 input neurons, 10 output classes (digits 0-9)
        self.fc2 = nn.Linear(128, 10)
        # Dropout layer to reduce overfitting
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Apply first conv layer, ReLU, and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply second conv layer, ReLU, and max pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten feature maps
        x = x.view(-1, 64 * 7 * 7)
        # Fully connected layer with ReLU
        x = F.relu(self.fc1(x))
        # Apply dropout
        x = self.dropout(x)
        # Output layer (logits)
        x = self.fc2(x)
        return x

# ------------------- Training Setup -------------------
model = CNNNet()  # Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
model.to(device)  # Move model to the selected device

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# ------------------- Early Stopping Parameters -------------------
patience = 3  # Number of epochs to wait for improvement before stopping
best_val_loss = np.inf  # Initialize best validation loss
epochs_no_improve = 0  # Counter for epochs without improvement
early_stop = False  # Flag to indicate if early stopping was triggered
num_epochs = 10  # Maximum number of epochs
save_path = "best_model.pth"  # Path to save the best model

# ------------------- Training Loop with Early Stopping -------------------
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # ------------------- Validation -------------------
    model.eval()
    val_loss = 0.0
    with torch.no_grad():  # No gradient calculation during validation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(test_loader)  # Average validation loss

    print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {running_loss:.4f} - Validation Loss: {val_loss:.4f}")

    # ------------------- Early Stopping Check -------------------
    if val_loss < best_val_loss:
        # Validation loss improved, save the model
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), save_path)
        print(" Validation loss improved. Model saved.")
    else:
        # No improvement in validation loss
        epochs_no_improve += 1
        print(f" No improvement. Patience: {epochs_no_improve}/{patience}")
        if epochs_no_improve >= patience:
            print(" Early stopping triggered.")
            early_stop = True
            break

# After training completes or early stopping is triggered
if not early_stop:
    print("Training completed without early stopping.")

# Load the best model from disk
model.load_state_dict(torch.load(save_path))
print("Best model loaded from disk.")

# confusion_matrix
model.eval()
y_true=[]
y_pred=[]

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _,predicted=torch.max(outputs,1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())


cm=confusion_matrix(y_true,y_pred)
print("confusion_matrix:")
print(cm)
