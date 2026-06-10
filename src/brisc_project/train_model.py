import torch
import torch.nn as nn
import torch.optim as optim

from brisc_project.models import get_resnet18
from brisc_project.brisc import get_data_loader


def train_model():
    train_loader, classes = get_data_loader(
        "brain_tumor/brisc2025/classification_task/train",
        batch_size=32
    )

    test_loader, _ = get_data_loader(
        "brain_tumor/brisc2025/classification_task/test",
        batch_size=32
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_resnet18(num_classes=len(classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 3

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Loss: {running_loss:.4f}")
        print(f"Training Accuracy: {correct / total:.4f}")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("Test Accuracy:", correct / total)

    torch.save(model.state_dict(), "models/resnet18_brain_tumor.pth")
    print("Model saved to models/resnet18_brain_tumor.pth")


if __name__ == "__main__":
    train_model()