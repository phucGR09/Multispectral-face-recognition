import numpy as np
import torch
from torch.utils.data import Dataset
import random
from torchvision.transforms import Compose, ToTensor, Normalize

# Transform (can be reused externally)
default_transform = Compose([
    lambda x: x.astype("float32"),
    ToTensor(),
    Normalize(0.5, 0.5)
])

class TripletDualDataset(Dataset):
    def __init__(self, df, transform=default_transform):
        self.df = df
        self.transform = transform
        self.subj_to_indices = df.groupby('Subject_ID').indices
        self.labels = df['Subject_ID'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        anchor = self.df.iloc[idx]
        anchor_id = anchor['Subject_ID']
        mod_a = anchor['Modality']
        face_a = anchor['face_aligned']

        other = self.df[(self.df['Subject_ID'] == anchor_id) & (self.df['Modality'] != mod_a)].iloc[0]
        vis_a = face_a if mod_a == 'VIS' else other['face_aligned']
        lwir_a = face_a if mod_a == 'Thermal' else other['face_aligned']

        # Positive
        pos_idx = next(i for i in self.subj_to_indices[anchor_id] if i != idx)
        pos = self.df.iloc[pos_idx]
        mod_p = pos['Modality']
        other_p = self.df[(self.df['Subject_ID'] == anchor_id) & (self.df['Modality'] != mod_p)].iloc[0]
        vis_p = pos['face_aligned'] if mod_p == 'VIS' else other_p['face_aligned']
        lwir_p = pos['face_aligned'] if mod_p == 'Thermal' else other_p['face_aligned']

        # Negative
        neg_subj = random.choice([s for s in self.subj_to_indices if s != anchor_id])
        neg_idx = random.choice(self.subj_to_indices[neg_subj])
        neg = self.df.iloc[neg_idx]
        mod_n = neg['Modality']
        other_n = self.df[(self.df['Subject_ID'] == neg_subj) & (self.df['Modality'] != mod_n)].iloc[0]
        vis_n = neg['face_aligned'] if mod_n == 'VIS' else other_n['face_aligned']
        lwir_n = neg['face_aligned'] if mod_n == 'Thermal' else other_n['face_aligned']

        def prep(vis, lwir):
            return self.transform(vis.squeeze()), self.transform(lwir.squeeze())

        return prep(vis_a, lwir_a), prep(vis_p, lwir_p), prep(vis_n, lwir_n)
