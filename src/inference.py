# src/inference.py

import torch
import numpy as np

def infer_identity(model, input_vis, input_lwir, gallery_embeddings, gallery_labels, transform, device, top_k=1):
    """
    Dự đoán danh tính ảnh đầu vào bằng so khớp với gallery
    Args:
        model: mô hình dual-stream đã huấn luyện
        input_vis, input_lwir: ảnh VIS và LWIR đã căn chỉnh (numpy array)
        gallery_embeddings: ma trận vector của gallery
        gallery_labels: nhãn tương ứng của gallery
        transform: transform áp dụng cho ảnh
        device: thiết bị chạy (cuda/cpu)
        top_k: trả về k kết quả gần nhất
    Returns:
        Danh sách top-k nhãn dự đoán cùng điểm tương đồng
    """
    model.eval()
    with torch.no_grad():
        vis_tensor = transform(input_vis.squeeze()).unsqueeze(0).to(device)
        lwir_tensor = transform(input_lwir.squeeze()).unsqueeze(0).to(device)
        embedding = model(vis_tensor, lwir_tensor).cpu().numpy().ravel()

    sims = np.dot(gallery_embeddings, embedding)
    top_indices = np.argsort(sims)[::-1][:top_k]
    results = [(gallery_labels[i], sims[i]) for i in top_indices]
    return results
