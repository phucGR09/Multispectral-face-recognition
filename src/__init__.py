"""
Multispectral Face Recognition
Dual-stream deep learning model for VIS & LWIR face matching.

Modules:
- config: Project-wide constants and paths
- dataset: Triplet dual-stream dataset
- model: DualStreamNetwork definition
- train: Training pipeline
- evaluate: Top-k accuracy + cosine evaluation
- utils: Face preprocessing and alignment
"""

__version__ = "0.1"
__author__ = "Phan Van Phuc"
