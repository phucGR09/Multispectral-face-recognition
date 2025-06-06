# main.py

import os
import torch
from src.config import CFG
from src.dataset import load_dataset, preprocess_faces, split_dataframes
from src.model import DualStreamNetwork
from src.train import train_model
from src.evaluate import evaluate_model
from src.utils import save_model

def main():
    # 1. Load raw data
    print("[INFO] Loading dataset...")
    df = load_dataset(CFG.root_dir)
    print(f"[INFO] Loaded {len(df)} images.")

    # 2. Preprocess: align face
    print("[INFO] Preprocessing and aligning faces...")
    df = preprocess_faces(df)

    # 3. Train/test split
    print("[INFO] Splitting into train/test and gallery/probe sets...")
    train_df, test_df, gallery_df, probe_df = split_dataframes(df)

    # 4. Init model & optimizer
    print("[INFO] Initializing model...")
    model = DualStreamNetwork(emb_dim=CFG.embedding_dim).to(CFG.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.learning_rate)

    # 5. Train model
    print("[INFO] Start training...")
    train_model(model, optimizer, train_df, CFG)

    # 6. Save model
    os.makedirs("checkpoints", exist_ok=True)
    save_model(model, optimizer, CFG.epochs, os.path.join("checkpoints", "dual_stream_final.pth"))

    # 7. Evaluate
    print("[INFO] Evaluating model...")
    evaluate_model(model, gallery_df, probe_df)

if __name__ == "__main__":
    main()
