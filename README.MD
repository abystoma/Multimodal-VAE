# 🎵 Variational Autoencoders for Joint Audio–Text Representation Learning in Music Clustering

This project explores unsupervised clustering of music using Variational Autoencoders (VAE) and multimodal representations combining audio and lyrics.

The pipeline is designed in three levels of increasing complexity:
- Easy: Audio-only features
- Medium: Multimodal (audio + lyrics)
- Hard: Beta-VAE with structured latent learning

---

## 📂 Project Structure
```
.
├── Easy/
│   ├── main_*.py
│   └── src/
├── Medium/
│   ├── main_*.py
│   └── src/
├── Hard/
│   ├── main_*.py
│   └── src/
├── data/
├── results/
├── plots/
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Create environment

conda create -n vae_music python=3.10  
conda activate vae_music  

### 2. Install dependencies

pip install -r requirements.txt

---

## 📊 Dataset

This project uses the MIR-1K dataset:
- 1000 music clips
- corresponding lyrics

Note: The dataset is predominantly Mandarin. Clustering reflects acoustic and lyrical patterns, not strict language separation.

---

## 🚀 Running the Pipeline

### Easy Task (Audio-only)

python Easy/main_build_dataset.py  
python Easy/main_prepare_data.py  
python Easy/main_train_vae.py  
python Easy/main_evaluate.py  

---

### Medium Task (Audio + Lyrics)

python Medium/main_build_mir1k_medium.py  
python Medium/main_prepare_medium.py  
python Medium/main_train_multimodal_vae_medium.py  
python Medium/main_evaluate_medium.py  

---

### Hard Task (Beta-VAE + Singer Clustering)

python Hard/main_train_beta_vae.py  
python Hard/main_evaluate_hard.py  

---

## 🧠 Method Overview

Easy:
- Handcrafted audio features (MFCC, spectral)
- Fully connected VAE
- KMeans clustering

Medium:
- Mel-spectrogram + TF-IDF lyrics
- Multimodal VAE
- KMeans, Agglomerative, DBSCAN

Hard:
- Beta-VAE (disentangled latent space)
- Latent augmentation for class balance
- Singer-level clustering

---

## 📈 Evaluation Metrics

- Silhouette Score
- Calinski–Harabasz Index
- Davies–Bouldin Index
- NMI, ARI, Purity (Hard task)

---

## 📊 Visualization

- UMAP projections for latent space
- Cluster plots saved as PDF

---

## 📌 Key Insights

- PCA is a strong baseline for simple feature spaces
- Multimodal learning improves latent structure
- Beta-VAE enhances cluster separability
- DBSCAN captures density but may miss global structure

---

## ⚠️ Notes

- GPU recommended for faster training

---

## 👤 Author

Zayed Humayun  
BRAC University  
zayed.humayun@bracu.ac.bd