# Multispectral Face Recognition 🔥

This project implements a **multispectral face recognition system** using dual-stream CNNs for **VIS** and **LWIR** images. The architecture is inspired by HyperFace-style dual branches with a **Triplet Loss** training scheme.

## 📁 Project Structure
Multispectral-face-recognition
* main.py # Entry point to run full pipeline
* requirements.txt
* README.md
* src/
  * init.py
  * config.py # Configuration and hyperparameters
  * dataset.py # Dataset + preprocessing + alignment
  * model.py # Dual-stream model (VIS + LWIR)
  * train.py # Triplet training pipeline
  * evaluate.py # Top-k evaluation using cosine similarity
  * inference.py # Embedding extraction + similarity
  * utils.py # Saving/loading models, visualization

## ⚙️ Setup Instructions

### 1. Create environment

If using **Conda**:
```bash
conda create -n multispectral-face python=3.10 pip -y
conda activate multispectral-face

pip install -r requirements.txt

```
## 🏃 Run the Full Pipeline
### 1. Prepare Dataset
Ensure your dataset is extracted in the following structure:
<pre>
/path/to/dataset/
├── Koschan/
│   ├── Expression/
│   └── Illumination/
├── Sarash/
│   └── ...
└── ...
</pre>

### 2. Run pipeline
From the project root:
```bash
python main.py --data_dir "/path/to/dataset" --save_dir "./outputs"
```

This will:

* Parse and align all face images

* Train a dual-stream Triplet CNN

* Extract embeddings

* Evaluate recognition using Top-1 and Top-5 Accuracy

## 📊 Evaluation Metrics
* Top-k Accuracy

* Cosine Similarity Matrix

* ROC-AUC (Optional)

* Gallery-probe matching with fixed pose settings

## 🧠 Requirements
* Python 3.10+ 

* PyTorch

* OpenCV

* Pandas / NumPy / Matplotlib

* scikit-learn

* PIL / tqdm
