# src/train.py

import torch
from torch.nn import TripletMarginLoss
from torch.optim import Adam
from tqdm import tqdm

def train(model, train_loader, device, lr=1e-4, margin=0.5, num_epochs=10):
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = TripletMarginLoss(margin=margin)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for (anchor, positive, negative) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            va, la = anchor[0].to(device), anchor[1].to(device)
            vp, lp = positive[0].to(device), positive[1].to(device)
            vn, ln = negative[0].to(device), negative[1].to(device)

            emb_a = model(va, la)
            emb_p = model(vp, lp)
            emb_n = model(vn, ln)

            loss = criterion(emb_a, emb_p, emb_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    return model
