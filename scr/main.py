import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from models import SimpleNet, CNN
from utils import load_image

# --- 1. Adatok betöltése (CIFAR-10) ---
transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True)

# --- 2. Modell kiválasztása ---
model = SimpleNet()  # vagy CNN(num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 3. Tanítás 2 epoch ---
loss_values = []
for epoch in range(2):
    running_loss = 0.0
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(trainloader)
    loss_values.append(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# --- 4. Loss görbe ---
plt.plot(loss_values)
plt.title("Loss over epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# --- 5. Saját kép tesztelése ---
img_path = "./data/mycat.jpg"  # ide töltsék fel a saját képet
img_tensor = load_image(img_path)

model.eval()
with torch.no_grad():
    pred = model(img_tensor)
    probs = torch.softmax(pred, dim=1)
    print("Predicted probabilities:", probs)
    print("Predicted class:", torch.argmax(probs).item())
