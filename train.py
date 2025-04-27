import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import matplotlib.pyplot as plt
from tqdm import tqdm

# Dataset path (update this to the correct path where the dataset is installed)
data_path = r'C:\Users\kirth\OneDrive\Desktop\Food101\images'  # Update with your Food-101 dataset path

# Transforms for preprocessing
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset using ImageFolder
dataset = ImageFolder(root=data_path, transform=transform)

# Split the dataset into train, validation, and test sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

# Load ConvNeXt-Tiny model with pre-trained weights
weights = ConvNeXt_Tiny_Weights.DEFAULT
model = convnext_tiny(weights=weights)

# Unfreeze all layers for fine-tuning
for param in model.parameters():
    param.requires_grad = True

# Replace the classifier head for 101 classes (Food-101)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, 101)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()

# Training function
def train_model(epochs, initial_lr):
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    train_losses, val_losses, val_accuracies, lrs = [], [], [], []

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        current_lr = scheduler.get_last_lr()[0]
        lrs.append(current_lr)

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [LR: {current_lr:.6f}]", leave=False)
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        scheduler.step()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs}: \n  Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f} | Val Acc = {val_acc:.2f}% | LR = {current_lr:.6f} | Time = {time.time() - start_time:.2f}s")

    return train_losses, val_losses, val_accuracies, lrs

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("✅ Training on:", device)
    model = model.to(device)
    train_losses, val_losses, val_accuracies, lrs = train_model(epochs=25, initial_lr=1e-4)

    # Evaluate on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    print(f"\n✅ Final Test Accuracy: {test_acc:.2f}%")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/food101_convnext_tiny_finetuned.pth")
    print("✅ Model saved to models/food101_convnext_tiny_finetuned.pth")

    # Plotting
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title('Validation Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(lrs, label='Learning Rate')
    plt.title('Learning Rate per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_plots.png")
    plt.show()
