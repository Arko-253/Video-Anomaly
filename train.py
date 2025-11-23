import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.aa_rae import AA_RAE
from utils.dataset import UCSDFrameDataset
from tqdm import tqdm
import os
os.makedirs("checkpoints", exist_ok=True)


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

if __name__ == "__main__":
    
    train_dataset = UCSDFrameDataset("data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)

    model = AA_RAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    EPOCHS = 30

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            batch = batch.to(device)

            recon = model(batch)
            loss = criterion(recon, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (pbar.n + 1))

        torch.save(model.state_dict(), f"checkpoints/ucsd_epoch_{epoch+1}.pth")

    print("Training complete.")
