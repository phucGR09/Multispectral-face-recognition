# src/evaluate.py

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

def get_dual_embeddings(model, df, transform, device):
    model.eval()
    vecs = []
    with torch.no_grad():
        for _, row in df.iterrows():
            mod = row['Modality']
            other = df[(df['Subject_ID'] == row['Subject_ID']) & (df['Modality'] != mod)].iloc[0]
            vis = row['face_aligned'] if mod == 'VIS' else other['face_aligned']
            lwir = row['face_aligned'] if mod == 'Thermal' else other['face_aligned']

            vis_tensor = transform(vis.squeeze()).unsqueeze(0).to(device)
            lwir_tensor = transform(lwir.squeeze()).unsqueeze(0).to(device)
            emb = model(vis_tensor, lwir_tensor)
            vecs.append(emb.cpu().numpy().ravel())
    return np.vstack(vecs)


def compute_similarity_matrix(probe_embeddings, gallery_embeddings):
    return cosine_similarity(probe_embeddings, gallery_embeddings)


def top_k_accuracy(similarity_matrix, probe_labels, gallery_labels, k=1):
    correct = sum(
        probe_labels[i] in gallery_labels[np.argsort(row)[::-1][:k]]
        for i, row in enumerate(similarity_matrix)
    )
    return correct / len(probe_labels)
