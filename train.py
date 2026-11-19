import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ✅ Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ✅ Paths
data_dir = "faces"  # folder with Real/ and Fake/

# ✅ Transforms (Resize to 128x128 to save memory)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ✅ Dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# ✅ Train-test split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# ✅ DataLoaders (Lower batch size for small VRAM)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# ✅ Model (ResNeXt)
model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: Real, Fake
model = model.to(device)

# ✅ Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# ✅ Training loop
def train_model(num_epochs=3):  # Start with 3 to test
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # ✅ Save model
    torch.save(model.state_dict(), "model.pth")
    print("✅ Model saved as model.pth")

# ✅ Start training
if __name__ == "__main__":
    train_model(num_epochs=3)  # You can increase later after testing
