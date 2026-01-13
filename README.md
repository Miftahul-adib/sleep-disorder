# A Dual Pipeline Machine Learning Framework for
Automated Multi Class Sleep Disorder Screening
Using Hybrid Resampling and Ensemble Learning


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Accuracy](https://img.shields.io/badge/Accuracy-98.67%25-brightgreen.svg)](https://github.com/yourusername/sleep-disorder-screening)

> **State-of-the-Art Solution for Multi-Class Sleep Disorder Classification**

This repository contains the official implementation of **"A Dual Pipeline Machine Learning Framework for Automated Multi-Class Sleep Disorder Screening Using Hybrid Resampling and Ensemble Learning"** (2026).

Our framework achieves **98.67% accuracy** on the Sleep Health and Lifestyle Dataset, surpassing current benchmarks through a novel dual-pipeline architecture that captures both linear and non-linear feature dependencies.

---

## 🎯 Highlights

- ** State-of-the-Art Performance**: 98.67% accuracy (vs. 97.33% previous best)
- ** Novel Architecture**: Dual-pipeline design for comprehensive feature processing
- ** Advanced Balancing**: SMOTETomek hybrid resampling for class imbalance
- ** Statistical Validation**: Wilcoxon Signed-Rank Test confirms significance (p=0.00391)
- ** Real-Time Ready**: Inference under 400ms for clinical deployment

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Benchmarking](#benchmarking)
- [Citation](#citation)
- [Authors](#authors)
- [License](#license)

---

## 🔍 Overview

Sleep disorders affect millions globally, yet accurate multi-class screening remains challenging. This framework addresses the problem of classifying patients into three categories:

- **Insomnia** (Class 0)
- **Healthy/None** (Class 1)
- **Sleep Apnea** (Class 2)

### Architecture Philosophy

We employ a **bifurcated processing approach**:
```
Input Data
    ├── Pipeline 1 (Statistical Path)
    │   ├── RobustScaler
    │   ├── Mutual Information Selection
    │   └── Linear Discriminant Analysis (LDA)
    │
    └── Pipeline 2 (Wrapper-Based Path)
        ├── MinMaxScaler
        ├── Boruta Feature Selection
        └── Autoencoder Compression
```

This dual approach ensures both linear separability and non-linear interaction capture.

---

## ✨ Key Features

### 1. **Dual-Pipeline Architecture**
- **Pipeline 1**: Optimizes for linear dependencies using statistical methods
- **Pipeline 2**: Captures non-linear interactions through wrapper-based selection

### 2. **Hybrid Resampling Strategy**
Addresses severe class imbalance (Healthy: 219, Apnea: 78, Insomnia: 77):
- **SMOTE**: Synthetic minority oversampling
- **Tomek Links**: Decision boundary cleaning

### 3. **Comprehensive Feature Engineering**
- 8 custom interaction features capturing physiological dependencies
- Tree-based importance ranking
- Cross-validated feature stability analysis

### 4. **Rigorous Validation**
- Stratified 8-fold cross-validation
- Wilcoxon Signed-Rank Test for statistical significance
- Multiple performance metrics (Accuracy, F1, Recall, Precision)

---

## 📊 Dataset

**Sleep Health and Lifestyle Dataset**

The dataset integrates:
- Physiological parameters (heart rate, blood pressure, sleep duration)
- Lifestyle factors (physical activity, stress levels, BMI)
- Demographic information (age, gender, occupation)

**Preprocessing Steps**:
1. Data integrity verification (zero null values confirmed)
2. Occupation grouping to reduce sparsity
3. Label encoding for ordinal variables
4. One-hot encoding for nominal features

---

## 🛠 Methodology

### 1. Data Preprocessing
```python
# Zero null values confirmed
# Occupation grouping applied
# Encoding: Label (ordinal) + One-Hot (nominal)
```

### 2. Feature Engineering
- Created 8 interaction features
- Tree-based importance analysis
- Information gain ranking

### 3. Train-Test Split
- **80/20 stratified split** preserving class distribution

### 4. Class Balancing
```python
from imblearn.combine import SMOTETomek

# Original: Healthy=219, Apnea=78, Insomnia=77
smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
```

### 5. Dual Pipeline Processing

#### Pipeline 1: Statistical
```python
# Scaling: RobustScaler (median/IQR normalization)
# Selection: Mutual Information (relevance filtering)
# Reduction: LDA (linear projection)
```

#### Pipeline 2: Wrapper-Based
```python
# Scaling: MinMaxScaler ([0,1] normalization)
# Selection: Boruta (shadow feature comparison)
# Reduction: Autoencoder (non-linear latent space)
```

### 6. Model Training & Validation
- **9 Classifiers**: Extra Trees, KNN, XGBoost, LightGBM, MLP, etc.
- **Hyperparameter Tuning**: RandomizedSearchCV
- **Cross-Validation**: Stratified 8-fold
- **Statistical Testing**: Wilcoxon Signed-Rank (p < 0.05)

---


### Setup



### Required Packages
```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
imbalanced-learn>=0.9.0
xgboost>=1.5.0
lightgbm>=3.3.0
boruta>=0.3.0
tensorflow>=2.8.0  # For Autoencoders
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
```

---




## 📈 Results

### Top Performing Models

| Pipeline | Model | Configuration | Accuracy | F1 Score | Recall | Precision |
|----------|-------|---------------|----------|----------|--------|-----------|
| 1 | K-Nearest Neighbors | MI + SMOTETomek | **98.67%** | 97.85% | 97.92% | 97.92% |
| 1 | XGBoost | MI + SMOTETomek | **98.67%** | 97.85% | 97.92% | 97.92% |
| 1 | LightGBM | MI + SMOTETomek | **98.67%** | 97.85% | 97.92% | 97.92% |
| 2 | Extra Trees | Boruta + SMOTETomek | **98.67%** | 97.84% | 97.78% | 98.04% |

### Performance Visualization

<p align="center">
  <img src="assets/confusion_matrix.png" width="45%" />
  <img src="assets/roc_curves.png" width="45%" />
</p>

### Statistical Validation

- **Wilcoxon Signed-Rank Test**: p = 0.00391 (W = 36.0)
- **Significance Level**: p < 0.05 ✓
- **Conclusion**: Performance gains are statistically significant

### Computational Efficiency

- **Inference Latency**: < 400ms per prediction
- **Training Time (Best Model)**: 1.04 seconds (KNN, Pipeline 1)
- **Suitable for**: Real-time clinical monitoring

---

## 🏆 Benchmarking

Our framework sets a new state-of-the-art on the Sleep Health and Lifestyle Dataset:

| Study | Year | Model | Accuracy |
|-------|------|-------|----------|
| **This Work** | **2026** | **Extra Trees (Pipeline 2)** | **98.67%** |
| Rahman et al. | 2025 | Gradient Boosting | 97.33% |
| Monowar et al. | 2025 | Ensemble Model | 96.88% |
| Satyavathi et al. | 2025 | Decision Tree | 96.00% |
| Alshammari et al. | 2024 | ANN | 92.92% |
| Hidayat et al. | 2023 | Random Forest | 88.00% |

**Improvement**: +1.34 percentage points over previous best

---

## 📄 Citation

If you use this code or methodology in your research, please cite:
```bibtex
@article{ovi2026dual,
  title={A Dual Pipeline Machine Learning Framework for Automated Multi Class Sleep Disorder Screening Using Hybrid Resampling and Ensemble Learning},
  author={Ovi, Md Sultanul Islam and Munfa, Muhsina Tarannum and Adib, Miftahul Alam and Hasan, Syed Sabbir},
  year={2026}
}
```

---

## 👥 Authors

- **Md Sultanul Islam Ovi** - George Mason University
- **Muhsina Tarannum Munfa** - Metropolitan University, Sylhet
- **Miftahul Alam Adib** - Shahjalal University of Science and Technology
- **Syed Sabbir Hasan** - Shahjalal University of Science and Technology

---

## 📝 License

This project is licensed under the MIT License 



---

## 📞 Contact

For questions or collaboration inquiries:

- **Email**: [your.email@university.edu]
- **Issues**: [GitHub Issues](https://github.com/yourusername/sleep-disorder-screening/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/sleep-disorder-screening/discussions)

---



## 📚 Additional Resources

- [Paper (arXiv)](https://arxiv.org/abs/xxxx.xxxxx)


---

<p align="center">
  <strong>⭐ Star this repo if you find it useful! ⭐</strong>
</p>
