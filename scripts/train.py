import torch
from torch.utils.data import DataLoader
from dataset import LowLightDataset, transform
from model import LowLightEnhancer
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LowLightEnhancer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

train_dataset = LowLightDataset("data/low_light","data/high_light", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

num_epochs = 50
os.makedirs("models", exist_ok=True)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for low, high in train_loader:
        low, high = low.to(device), high.to(device)
        optimizer.zero_grad()
        out = model(low)
        loss = criterion(out, high)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")
    # save model checkpoint
    if (epoch+1)%10==0:
        torch.save(model.state_dict(), f"models/epoch_{epoch+1}.pth")
