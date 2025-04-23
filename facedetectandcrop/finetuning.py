import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from facenet_pytorch import InceptionResnetV1
from torch.utils.data import DataLoader
from tqdm import tqdm

data_dir = 'DatasetSplit'
save_model_path = 'finetuned_facenet.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[INFO] Using device: {device}')

data_transforms = {
    'Training': transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),
    'Validation': transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),
    'Testing': transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),
}

image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ['Training', 'Validation', 'Testing']
}

dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=16, shuffle=True)
    for x in ['Training', 'Validation', 'Testing']
}

num_classes = len(image_datasets['Training'].classes)
print(f'[INFO] Jumlah kelas: {num_classes}')

model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 10
for epoch in range(num_epochs):
    print(f"\n[Epoch {epoch+1}/{num_epochs}]")

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloaders['Training']):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in dataloaders['Validation']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    print(f"Validation Accuracy: {val_acc:.4f}")

torch.save(model.state_dict(), save_model_path)
print(f"\n[✓] Model disimpan ke {save_model_path}")
