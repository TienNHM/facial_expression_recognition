import torch
import torch.nn as nn
import torch.optim as optim
from model import EmotionCNN
from dataset import get_dataloaders

# Device setup (GPU nếu có)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get dataloaders
train_loader, val_loader, class_names = get_dataloaders()
print('Class Names: ', class_names)

# Instantiate model, loss, and optimizer
model = EmotionCNN(num_classes=len(class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}")

    # Optional: Evaluate on validation set
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")

# Lưu model vào file .pth
torch.save(model.state_dict(), "models/emotion_cnn.pth")

# Chuyển model về chế độ eval để xuất sang ONNX
model.eval()

# Dummy input (Ảnh grayscale 48x48 với một kênh, ví dụ)
dummy_input = torch.randn(1, 1, 48, 48).to(device)  # Kích thước của ảnh đầu vào

# Xuất mô hình sang ONNX
torch.onnx.export(
    model,
    dummy_input,
    "models/emotion_cnn.onnx",  # Tên file xuất ra
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11  # Sử dụng opset version 11
)

print("✅ Xuất ONNX thành công: models/emotion_cnn.onnx")
