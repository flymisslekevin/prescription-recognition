import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import pillow_heif
pillow_heif.register_heif_opener()
from torchvision.models import MobileNet_V2_Weights

if __name__ == "__main__":
    # Configuration
    DATA_DIR = 'dataset'
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data Transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    print("TEST")
    # Datasets and Loaders
    train_dataset = datasets.ImageFolder(f'{DATA_DIR}/train', transform=train_transforms, is_valid_file=lambda x: x.lower().endswith(('.jpg', '.jpeg', '.png', '.heic')))
    val_dataset = datasets.ImageFolder(f'{DATA_DIR}/val', transform=val_transforms, is_valid_file=lambda x: x.lower().endswith(('.jpg', '.jpeg', '.png', '.heic')))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Model
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, 1)
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"TEST IF WE GET HERE{type(train_loader)}")
    print(f"TEST IF WE GET HERE{len(train_loader.dataset)}")

    # Training Loops
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                outputs = model(inputs)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total

        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] - Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.4f}')

    # Save Model
    torch.save(model.state_dict(), 'second_mobilenetv2_prescription_label.pth')

    print("Training complete. Model saved.")
