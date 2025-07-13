🧠 MNIST Digit Classification with CNN (PyTorch)
This project implements a Convolutional Neural Network (CNN) using PyTorch to classify handwritten digits from the MNIST dataset. It includes training, validation with early stopping, confusion matrix evaluation, and loss visualization.

🔍 Project Overview
Dataset: MNIST — 70,000 grayscale 28x28 images of handwritten digits (0-9)

Model: CNN with 2 convolutional layers, ReLU activations, dropout, and fully connected layers

Framework: PyTorch

Features:

Training/Validation split

Early stopping

Accuracy evaluation

Confusion matrix

Loss curve visualization

📁 Project Structure
bash
Copy
Edit
├── data/                 # MNIST dataset (downloaded automatically)
├── best_model.pth        # Best model saved (after early stopping)
├── main.py               # Main training and evaluation script
└── README.md             # You're reading it!
⚙️ Requirements
Install dependencies using pip:

bash
Copy
Edit
pip install torch torchvision scikit-learn matplotlib seaborn
🚀 Usage
Run the training and evaluation script:

bash
Copy
Edit
python main.py
🧠 Model Architecture
Input: [1, 28, 28] grayscale image

Conv2D: 1 → 32 filters, 3x3 kernel + ReLU + MaxPool(2x2)

Conv2D: 32 → 64 filters, 3x3 kernel + ReLU + MaxPool(2x2)

Flatten → Linear (64×7×7 → 128) + ReLU + Dropout(0.25)

Output Linear (128 → 10 classes)

📊 Results
✅ Early stopping used with patience of 3 epochs

📈 Validation Accuracy: ~98%

📉 Loss curves: can be visualized by running the script

📌 Confusion Matrix:

✅ Features Implemented
CNN with 2 convolutional layers

Early stopping

Confusion matrix (scikit-learn)

Accuracy score

Visualizations with Matplotlib & Seaborn

Model saving and reloading