import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
from tqdm import tqdm
from resnext_lstm import ResNeXtLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class VideoDataset(Dataset):
    def __init__(self, root_dir, seq_len=10, transform=None):
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.transform = transform
        self.video_files = []
        self.labels = []

        for label_dir in ['Real', 'Fake']:
            label_path = os.path.join(root_dir, label_dir)
            if not os.path.isdir(label_path):
                continue
            for filename in os.listdir(label_path):
                if filename.endswith('.mp4'):
                    self.video_files.append(os.path.join(label_path, filename))
                    self.labels.append(0 if label_dir == 'Real' else 1)

    def __len__(self):
        return len(self.video_files)

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, total_frames // self.seq_len)

        for i in range(self.seq_len):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))  # CHANGED: 224x224 for ResNeXt
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames extracted from {video_path}")

        while len(frames) < self.seq_len:
            frames.append(frames[-1])
        return frames[:self.seq_len]

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label = self.labels[idx]
        frames = self.extract_frames(video_path)
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        frames = torch.stack(frames)
        return frames, torch.tensor(label)

# ✅ Transform (CHANGED)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ✅ Load dataset
dataset = VideoDataset(root_dir="../videos", seq_len=10, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# ✅ DataLoaders (CHANGED)
BATCH_SIZE = 16
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ✅ Model
model = ResNeXtLSTM(cnn_dropout=0.3, lstm_dropout=0.5).to(device)

# ✅ Loss and optimizer (CHANGED)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ Accuracy function (CHANGED)
def evaluate(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for frames, labels in loader:
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames)
            _, predicted = torch.max(outputs, 1)
            print("Predicted:", predicted.cpu().numpy(), "Labels:", labels.cpu().numpy())  # Debug
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# ✅ Training
def train_model(num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for frames, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100):
            frames, labels = frames.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        val_acc = evaluate(val_loader)
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    # Save model
    os.makedirs("../model", exist_ok=True)
    torch.save(model.state_dict(), "../model/resnext_lstm.pth")
    print("✅ Model saved as resnext_lstm.pth")

# ✅ Run
if __name__ == "__main__":
    train_model(num_epochs=10)
    print("✅ Training complete")