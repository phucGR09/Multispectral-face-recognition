# src/utils.py

import torch
import os

def save_model(model, optimizer, epoch, path):
    """
    Lưu trạng thái mô hình và optimizer.
    Args:
        model (torch.nn.Module): Mô hình cần lưu.
        optimizer (torch.optim.Optimizer): Optimizer đang dùng.
        epoch (int): Epoch hiện tại.
        path (str): Đường dẫn file .pt hoặc .pth để lưu.
    """
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, path)
    print(f"[INFO] Model saved to {path}")


def load_model(model, optimizer, path, device='cpu'):
    """
    Tải lại trạng thái mô hình và optimizer.
    Args:
        model (torch.nn.Module): Mô hình khởi tạo (chưa load weight).
        optimizer (torch.optim.Optimizer): Optimizer đang dùng.
        path (str): Đường dẫn file .pt hoặc .pth đã lưu.
        device (str): Thiết bị chạy ('cpu' hoặc 'cuda').
    Returns:
        model, optimizer, epoch
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found at {path}")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"[INFO] Loaded checkpoint from {path} (epoch {epoch})")
    return model, optimizer, epoch
