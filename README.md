# Eye-Disease-Detection
# Automated Eye Disease Detection from Retinal Images

A comparative machine learning study for classifying retinal 
diseases from OCT (Optical Coherence Tomography) scan images.

---

## Problem Statement

Retinal diseases such as CNV, DME, and Drusen are leading 
causes of irreversible blindness globally. This project builds 
an automated classification system to detect these conditions 
from retinal OCT scans, comparing classical ML approaches 
against deep learning.

---

## Dataset

**Kermany OCT Dataset** — Kermany et al., Cell 2018  
- 84,484 retinal OCT images across 4 classes  
- Classes: CNV, DME, Drusen, Normal  
- Source: [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/kermany2018)

---

## Models Compared

| Model | Type | Input |
|---|---|---|
| Support Vector Machine (SVM) | Classical | ResNet50 features |
| Random Forest | Classical | ResNet50 features |
| XGBoost | Classical | ResNet50 features |
| ResNet50 Fine-tuned CNN | Deep Neural Network | Raw images |

---

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| SVM | TBD | TBD |
| Random Forest | TBD | TBD |
| XGBoost | TBD | TBD |
| ResNet50 CNN | TBD | TBD |

*Results will be updated as experiments are completed.*

---

## Project Structure
```
eye-disease-detection/
├── notebooks/          # Jupyter notebooks for each stage
├── src/                # Python source files
├── results/            # Saved metrics, plots, confusion matrices
└── report/             # Final project report
```

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/elentajacob/eye-disease-detection.git
```

### 2. Open in Google Colab
Each notebook in the `notebooks/` folder can be opened 
directly in Google Colab.

### 3. Download the dataset
Follow the instructions in `notebooks/01_EDA.ipynb` to 
download the Kermany dataset from Kaggle.

---

## Tech Stack

- Python 3.x
- PyTorch + torchvision
- scikit-learn
- XGBoost
- NumPy / Pandas
- Matplotlib / Seaborn
- Grad-CAM

---

