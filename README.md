# Automated Eye Disease Detection from Retinal OCT Images

A comparative study of classical machine learning and deep learning approaches for classifying retinal diseases from OCT scan images.

**CS 6140 — Machine Learning | Northeastern University**
**Elenta Suzan Jacob | Rahul Gudivada**

---

## About

This project builds an automated system to classify retinal OCT scan images into four disease categories: CNV (Choroidal Neovascularization), DME (Diabetic Macular Edema), Drusen, and Normal. Early and accurate detection of these conditions is critical — they progress silently and can cause permanent vision loss if caught too late. Currently, diagnosis relies on specialists manually reviewing hundreds of scans a day, which is slow and prone to error. An automated classifier could serve as a first-pass screening tool, flagging abnormal scans for urgent review and extending access to regions where specialists are scarce.

We compare five approaches across the board: Linear SVM, RBF SVM, Random Forest, XGBoost, and a fine-tuned ResNet50 CNN. Classical models operate on 2048-dimensional feature vectors extracted from a frozen pretrained ResNet50, while the CNN is trained end-to-end using transfer learning. We also ran a data leakage audit on the dataset and found that 59.3% of the standard test split is byte-identical to training images — so results are reported on both the full test set and a cleaned version with those images removed.

---

## Results

| Model | Full Test Acc | Full F1 | Clean Test Acc | Clean F1 |
|---|---|---|---|---|
| Linear SVM | 94.01% | 0.9400 | 95.69% | 0.8454 |
| RBF SVM | 98.76% | 0.9876 | 97.97% | 0.9249 |
| Random Forest | 96.38% | 0.9636 | 91.12% | 0.6992 |
| XGBoost | 99.28% | 0.9928 | 98.22% | 0.9481 |
| **ResNet50 CNN** | **100.00%** | **1.0000** | **100.00%** | **1.0000** |

---

## Repository Structure

```
Eye-Disease-Detection/
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_SVM.ipynb
│   ├── 04_random_forest.ipynb
│   ├── 06_xgboost.ipynb
│   └── 05_restnet50_cnn.ipynb
├── results/
├── features/
├── checkpoints/
└── README.md
```

---

## Getting Started

### Step 1 — Clone the repo and download the notebooks

```bash
git clone https://github.com/elentajacob/Eye-Disease-Detection.git
```

All notebooks are in the `notebooks/` folder — download them directly from GitHub and open them in your preferred environment.

### Step 2 — Download the data and model files

The dataset, pre-extracted features, and trained model checkpoints are hosted on Google Drive.

**Dataset:**
[Download here](https://drive.google.com/drive/folders/1mhmrsUJfUtvrlel-6HaYZQ-XroUJ_Lf6?usp=drive_link) — place the `OCT2017` folder inside `dataset/`

**Features** (pre-extracted ResNet50 2048-d vectors):
[Download here](https://drive.google.com/drive/folders/1PfT1Db0MWTdWaBYjXaCZ2hhBVTeJOapF?usp=sharing) — place all `.npy` files inside `features/`

**Checkpoints** (trained model files):
[Download here](https://drive.google.com/drive/folders/1ShUhgIiZkCJuYweiRymWKnY0iupWpv3U?usp=sharing) — place all model files inside `checkpoints/`

Your folder should look like this after downloading:

```
Eye-Disease-Detection/
├── dataset/
│   └── OCT2017/
│       ├── train/
│       ├── val/
│       └── test/
├── features/
│   ├── train_features.npy
│   ├── train_labels.npy
│   ├── train_paths.npy
│   ├── val_features.npy
│   ├── val_labels.npy
│   ├── val_paths.npy
│   ├── test_features.npy
│   ├── test_labels.npy
│   └── test_paths.npy
└── checkpoints/
    ├── xgboost_model.pkl
    ├── svm_linearsvc_model.pkl
    └── resnet50_best.pth
```

### Step 3 — Install dependencies

```bash
pip install torch torchvision scikit-learn xgboost grad-cam seaborn numpy matplotlib
```

### Step 4 — Update the base path

Each notebook has a `BASE_DIR` variable at the top. Set it to wherever you have saved the project folder before running anything.

---

## Running the Notebooks

Run them in this order:

1. `01_EDA.ipynb` — data exploration and class distribution
2. `02_feature_extraction.ipynb` — extracts ResNet50 features and saves them to `features/` (skip if you downloaded the features from Drive)
3. `03_SVM.ipynb` — Linear and RBF SVM
4. `04_random_forest.ipynb` — Random Forest
5. `06_xgboost.ipynb` — XGBoost with full leakage audit
6. `05_restnet50_cnn.ipynb` — ResNet50 fine-tuning and Grad-CAM

Notebooks 3 through 6 load features directly from the `features/` folder so you don't need to re-run feature extraction each time. If you want to skip retraining and just run evaluation, all trained models are in the `checkpoints/` folder — load them directly in the respective notebook using joblib for the classical models and torch for the CNN.

---

## Tech Stack

- Python 3.10+
- PyTorch + torchvision
- scikit-learn
- XGBoost
- NumPy / Pandas
- Matplotlib / Seaborn
- pytorch-grad-cam

---

## Dataset

Kermany, D.S., Goldbaum, M., Cai, W., et al. (2018). Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. *Cell*, 172(5), 1122–1131.
Available on [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/kermany2018).
