<div align="center">

# 🧠 Dual-Pipeline Framework for Automated Sleep Disorder Screening

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-98.67%25-brightgreen.svg)](README.md)
[![Paper](https://img.shields.io/badge/Paper-2026-red.svg)](README.md)

**State-of-the-Art Solution (98.67% Accuracy) on Sleep Health & Lifestyle Dataset**

[Installation](#-installation) • [Features](#-key-features) • [Usage](#-quick-start) • [Results](#-results) • [Citation](#-citation)

</div>

---

## 📋 Overview

This repository contains the official implementation of **"A Dual Pipeline Machine Learning Framework for Automated Multi-Class Sleep Disorder Screening Using Hybrid Resampling and Ensemble Learning"**.

Our solution introduces a **bifurcated feature engineering architecture** that separates linear statistical processing from non-linear wrapper-based selection, achieving a new benchmark accuracy of **98.67%**, significantly outperforming recent 2025 SOTA approaches.

### 🎯 Key Highlights

- 🏆 **98.67% Accuracy** - New SOTA on Sleep Health & Lifestyle Dataset
- ⚡ **<400ms Inference** - Real-time prediction capability
- 🔄 **Dual-Pipeline Architecture** - Optimized for both linear and non-linear models
- 📊 **Hybrid Resampling** - SMOTETomek for balanced class distribution
- 🧪 **Statistically Validated** - Wilcoxon test (p=0.00391)

---

## 👥 Authors

<table>
<tr>
<td align="center">
<strong>Md Sultanul Islam Ovi</strong><br>
George Mason University
</td>
<td align="center">
<strong>Muhsina Tarannum Munfa</strong><br>
Metropolitan University, Sylhet
</td>
<td align="center">
<strong>Miftahul Alam Adib</strong><br>
Shahjalal University of Science and Technology, Sylhet
</td>
<td align="center">
<strong>Syed Sabbir Hasan</strong><br>
Shahjalal University of Science and Technology, 
</td>
</tr>
</table>

---

## 📝 Abstract

Accurate classification of sleep disorders (Insomnia, Sleep Apnea) is vital but traditionally relies on resource-intensive **Polysomnography (PSG)**. We propose a **Dual-Pipeline Framework** that processes lifestyle and physiological data through two parallel streams:

- **Statistical Pipeline**: Focuses on linear separability via Mutual Information and LDA
- **Wrapper-Based Pipeline**: Leverages Boruta selection and Autoencoders for non-linear interactions

Coupled with **SMOTETomek** hybrid resampling, our Extra Trees and KNN models achieved **98.67% accuracy** with inference latency under **400ms**.

---

## 🏗️ System Architecture
```mermaid
graph TB
    A[Raw Dataset374 samples, 13 features] --> B[Preprocessing]
    B --> C[Feature Engineering7 Interaction Features]
    C --> D[SMOTETomekBalanced Dataset]
    D --> E{Dual Pipeline}
    
    E --> F[Pipeline 1: Statistical]
    E --> G[Pipeline 2: Wrapper-Based]
    
    F --> F1[RobustScaler]
    F1 --> F2[Mutual Information]
    F2 --> F3[LDA]
    F3 --> F4[KNN, LogReg, XGBoost]
    
    G --> G1[MinMaxScaler]
    G1 --> G2[Boruta Selection]
    G2 --> G3[Autoencoder]
    G3 --> G4[Extra Trees, LightGBM]
    
    F4 --> H[98.67% Accuracy]
    G4 --> H


---

## 🛠️ Methodology

### 1️⃣ Data Preprocessing

**Dataset**: Sleep Health & Lifestyle (374 samples, 13 features)

#### Transformations Applied:
```python
# Blood Pressure Splitting
df[['Systolic_BP', 'Diastolic_BP']] = df['Blood_Pressure'].str.split('/', expand=True).astype(int)

# Categorical Encoding
label_encoding = {
    'Gender': {'Female': 0, 'Male': 1},
    'Sleep_Disorder': {'Insomnia': 0, 'None': 1, 'Sleep Apnea': 2}
}

# One-Hot Encoding for Occupation
df = pd.get_dummies(df, columns=['Occupation'], drop_first=True)

# BMI Category - Mean Clinical Value Mapping
bmi_mapping = {
    'Underweight': 17.5,
    'Normal': 21.7,
    'Overweight': 27.5,
    'Obese': 32.5
}
```

---

### 2️⃣ Feature Engineering

We engineered **7 interaction features** to capture hidden physiological dependencies:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `Stress_Sleep_Interaction` | `Stress / Quality_of_Sleep` | Inverse correlation between stress & sleep quality |
| `Sleep_Heart_Ratio` | `Sleep_Duration × Heart_Rate` | Physiological rest efficiency vs. cardiovascular load |
| `Sleep_Steps_Ratio` | `Sleep_Duration × Daily_Steps` | Balance between physical exertion & recovery |
| `Sleep_Stress_Ratio` | `Sleep_Duration × Stress_Level` | Adequacy of sleep relative to stress exposure |
| `BMI_Activity` | `BMI × Physical_Activity` | Combined body composition & exertion influence |
| `Pulse_Pressure` | `Systolic_BP - Diastolic_BP` | Indicator of arterial stiffness |
| `Sqrt_Sleep` | `√(Sleep_Duration)` | Variance stabilization for sleep distribution |
```python
# Feature Engineering Implementation
df['Stress_Sleep_Interaction'] = df['Stress_Level'] / (df['Quality_of_Sleep'] + 1e-5)
df['Sleep_Heart_Ratio'] = df['Sleep_Duration'] * df['Heart_Rate']
df['Sleep_Steps_Ratio'] = df['Sleep_Duration'] * df['Daily_Steps']
df['Sleep_Stress_Ratio'] = df['Sleep_Duration'] * df['Stress_Level']
df['BMI_Activity'] = df['BMI_Category'] * df['Physical_Activity_Level']
df['Pulse_Pressure'] = df['Systolic_BP'] - df['Diastolic_BP']
df['Sqrt_Sleep'] = np.sqrt(df['Sleep_Duration'])
```

---

### 3️⃣ Class Imbalance Handling

**Original Distribution**:
- Healthy: 219 samples
- Sleep Apnea: 78 samples
- Insomnia: 77 samples

**Solution: SMOTETomek Hybrid Resampling**
```python
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

# Initialize hybrid resampler
smt = SMOTETomek(
    smote=SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=5),
    tomek=TomekLinks(sampling_strategy='auto')
)

# Apply resampling
X_resampled, y_resampled = smt.fit_resample(X_train, y_train)
```

**Result**: Balanced classes (~173-175 samples each) with clearer decision boundaries.

---

### 4️⃣ Dual-Pipeline Architecture

<div align="center">

| Pipeline 1: Statistical Engineering | Pipeline 2: Wrapper-Based Engineering |
|:-----------------------------------:|:-------------------------------------:|
| **Target**: Linear & Distance Models | **Target**: Tree-Based Ensembles |
| RobustScaler (IQR-based) | MinMaxScaler (0-1 range) |
| Mutual Information Selection | Boruta Algorithm Selection |
| LDA Dimensionality Reduction | Autoencoder Compression |
| ✅ KNN, Logistic Regression, XGBoost | ✅ Extra Trees, LightGBM |

</div>

#### Pipeline 1 Implementation:
```python
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Statistical Pipeline
pipeline_1 = Pipeline([
    ('scaler', RobustScaler()),
    ('selector', SelectKBest(mutual_info_classif, k=15)),
    ('lda', LinearDiscriminantAnalysis(n_components=2)),
    ('classifier', KNeighborsClassifier(n_neighbors=5))
])
```

#### Pipeline 2 Implementation:
```python
from sklearn.preprocessing import MinMaxScaler
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Boruta Selection
rf = RandomForestClassifier(n_jobs=-1, max_depth=5, random_state=42)
boruta = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42)
boruta.fit(X_scaled, y_train)
X_boruta = boruta.transform(X_scaled)

# Autoencoder for Dimensionality Reduction
input_dim = X_boruta.shape[1]
encoding_dim = 10

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_boruta, X_boruta, epochs=50, batch_size=32, verbose=0)

# Extract encoded features
X_encoded = encoder.predict(X_boruta)
```

---

## 📊 Results

### 🏆 Main Performance Table

| Model | Pipeline | Accuracy | F1 Score | Precision | Recall |
|-------|----------|----------|----------|-----------|--------|
| **Extra Trees** | Pipeline 2 (Boruta + SMOTETomek) | **98.67%** | **97.84%** | **98.04%** | **97.78%** |
| **KNN** | Pipeline 1 (MI + SMOTETomek) | **98.67%** | **97.85%** | **97.92%** | **97.92%** |
| **XGBoost** | Pipeline 1 (MI + SMOTETomek) | **98.67%** | **97.85%** | **97.92%** | **97.92%** |
| **LightGBM** | Pipeline 1 (MI + SMOTETomek) | **98.67%** | **97.85%** | **97.92%** | **97.92%** |
| Logistic Regression | Pipeline 2 (MinMaxScaler) | 97.33% | 95.66% | 96.30% | 95.56% |
| MLP Neural Net | Pipeline 1 (MI Features) | 96.00% | 93.55% | 93.61% | 93.61% |

---

### 📈 SOTA Comparison (2024-2025)
```
┌─────────────────────────────────────────────────────────┐
│  98.67% │████████████████████████████████████│ Ovi et al. (2026) - Ours
│  97.33% │███████████████████████████████████ │ Rahman et al. (2025)
│  96.88% │██████████████████████████████████  │ Monowar et al. (2025)
│  96.00% │█████████████████████████████████   │ Satyavathi et al. (2025)
│  92.92% │████████████████████████████        │ Alshammari et al. (2024)
└─────────────────────────────────────────────────────────┘
```

**Our improvement**: +1.34% over previous SOTA (2025)

---

### 🔬 Ablation Study

| Experiment | Configuration | Accuracy | Impact |
|------------|--------------|----------|--------|
| Baseline | Raw features + KNN | 96.00% | - |
| + Normalization | RobustScaler + KNN | 96.00% | Minimal |
| + SMOTETomek | Balanced + KNN | **98.67%** | +2.67% ⬆️ |
| + Boruta (Trees) | Boruta + Extra Trees | **98.67%** | +2.67% ⬆️ |
| MLP without scaling | Raw + MLP | 61.30% | - |
| MLP with scaling | RobustScaler + MLP | 96.00% | +34.70% ⬆️ |

**Key Findings**:
- ✅ SMOTETomek is **essential** for KNN (2.67% boost)
- ✅ Normalization is **critical** for neural networks (34.7% boost)
- ✅ Boruta outperforms MI for tree-based models
- ✅ MI is computationally cheaper for linear models

---

### 📉 Statistical Validation

**Wilcoxon Signed-Rank Test** on 8-fold cross-validation:
```
p-value: 0.00391 (p < 0.05)
✅ Result: Performance improvement is statistically significant
```

---

## 📂 Repository Structure
```
sleep-disorder/
│
├── 📁 data/
│   ├── sleep_health_lifestyle_dataset.csv    # Original dataset
│   └── processed/
│       └── smote_tomek_balanced.csv           # Resampled data
│
├── 📓 notebooks/
│   ├── 1_EDA_Feature_Engineering.ipynb        # Preprocessing & interaction features
│   ├── 2_Pipeline1_Statistical.ipynb          # RobustScaler + MI + LDA
│   ├── 3_Pipeline2_Wrapper.ipynb              # MinMaxScaler + Boruta + Autoencoder
│   └── 4_Evaluation_Ablation.ipynb            # SOTA comparison & Wilcoxon test
│
├── 💾 models/
│   ├── saved_models/
│   │   ├── extra_trees_best.pkl               # Best Extra Trees model
│   │   └── knn_best.pkl                       # Best KNN model
│   └── encoder_weights/
│       └── autoencoder.h5                     # Trained autoencoder
│
├── 🐍 src/
│   ├── preprocessing.py                       # BP split, encoding functions
│   ├── feature_engineering.py                 # Interaction feature creation
│   ├── pipelines.py                           # Pipeline 1 & 2 implementations
│   └── metrics.py                             # Custom evaluation metrics
│
├── 📊 results/
│   ├── confusion_matrices/                    # CM visualizations (PNG)
│   ├── feature_importance_plots/              # Boruta & tree importance
│   └── performance_comparison.csv             # SOTA benchmark table
│
├── 📄 requirements.txt                        # Python dependencies
├── 📄 LICENSE                                 # MIT License
└── 📖 README.md                               # This file
```

---

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/Miftahul-adib/sleep-disorder.git
cd sleep-disorder

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Key Dependencies:**
```txt
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
boruta>=0.3
xgboost>=2.0.0
lightgbm>=4.0.0
tensorflow>=2.13.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

### Usage

#### 1️⃣ Data Preprocessing & Feature Engineering
```bash
jupyter notebook notebooks/1_EDA_Feature_Engineering.ipynb
```

This notebook will:
- Load the raw dataset
- Split Blood Pressure into Systolic/Diastolic
- Encode categorical variables
- Create 7 interaction features
- Output `processed_data.csv`

#### 2️⃣ Train the Best Model (Extra Trees - 98.67%)
```bash
jupyter notebook notebooks/3_Pipeline2_Wrapper.ipynb
```

Or use the Python script:
```python
from src.pipelines import train_pipeline_2
from src.preprocessing import load_and_preprocess

# Load data
X, y = load_and_preprocess('data/sleep_health_lifestyle_dataset.csv')

# Train Pipeline 2 (Boruta + Autoencoder + Extra Trees)
model, results = train_pipeline_2(X, y, use_smote=True)

print(f"Accuracy: {results['accuracy']:.4f}")
print(f"F1 Score: {results['f1']:.4f}")

# Save model
import joblib
joblib.dump(model, 'models/saved_models/extra_trees_best.pkl')
```

#### 3️⃣ Make Predictions
```python
import joblib
import pandas as pd
from src.preprocessing import preprocess_input

# Load trained model
model = joblib.load('models/saved_models/extra_trees_best.pkl')

# New patient data
patient_data = {
    'Gender': 'Male',
    'Age': 35,
    'Sleep_Duration': 6.5,
    'Quality_of_Sleep': 7,
    'Physical_Activity_Level': 60,
    'Stress_Level': 6,
    'BMI_Category': 'Overweight',
    'Blood_Pressure': '130/85',
    'Heart_Rate': 75,
    'Daily_Steps': 8000,
    'Occupation': 'Engineer'
}

# Preprocess and predict
X_new = preprocess_input(patient_data)
prediction = model.predict(X_new)

disorder_map = {0: 'Insomnia', 1: 'None', 2: 'Sleep Apnea'}
print(f"Predicted Sleep Disorder: {disorder_map[prediction[0]]}")
```

---

## 🔍 Key Insights

### 💡 Why Two Pipelines?

Different ML models have different feature preferences:

| Model Type | Optimal Features | Best Pipeline |
|------------|------------------|---------------|
| **Distance-based** (KNN) | Scaled, linearly separable | Pipeline 1 (MI + LDA) |
| **Linear** (LogReg, SVM) | Uncorrelated, normalized | Pipeline 1 (MI + LDA) |
| **Tree-based** (RF, XGB) | Non-linear interactions, all-relevant | Pipeline 2 (Boruta + AE) |
| **Neural Nets** (MLP) | Normalized, reduced dimensions | Either (depends on architecture) |

### 🎯 Feature Importance (Extra Trees - Pipeline 2)
```
Top 10 Most Important Features:
1. Sleep_Heart_Ratio          ████████████████████ 18.2%
2. Pulse_Pressure              ███████████████████  16.7%
3. Systolic_BP                 ██████████████       13.4%
4. Stress_Sleep_Interaction    ████████████         11.8%
5. BMI_Activity                ██████████            9.3%
6. Quality_of_Sleep            ████████              7.6%
7. Sleep_Duration              ███████               6.9%
8. Diastolic_BP                ██████                5.4%
9. Heart_Rate                  █████                 4.8%
10. Daily_Steps                ████                  3.9%
```

---

## 🧪 Reproduce Experiments

### Run Full Experiment Suite
```bash
# Run all notebooks in sequence
python scripts/run_experiments.py

# Or manually:
jupyter nbconvert --execute notebooks/1_EDA_Feature_Engineering.ipynb
jupyter nbconvert --execute notebooks/2_Pipeline1_Statistical.ipynb
jupyter nbconvert --execute notebooks/3_Pipeline2_Wrapper.ipynb
jupyter nbconvert --execute notebooks/4_Evaluation_Ablation.ipynb
```

### Cross-Validation
```python
from sklearn.model_selection import StratifiedKFold
from src.pipelines import train_pipeline_2

skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
scores = []

for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model, _ = train_pipeline_2(X_train, y_train)
    score = model.score(X_val, y_val)
    scores.append(score)

print(f"Mean Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
```

---

## 📜 Citation

If you find this work useful, please cite our paper:
```bibtex
@article{ovi2026dualpipeline,
  title={A Dual Pipeline Machine Learning Framework for Automated Multi-Class Sleep Disorder Screening Using Hybrid Resampling and Ensemble Learning},
  author={Ovi, Md Sultanul Islam and Munfa, Muhsina Tarannum and Adib, Miftahul Alam and Hasan, Syed Sabbir},
  institution={George Mason University and Shahjalal University of Science and Technology},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

For questions or collaborations:

- **Md Sultanul Islam Ovi** - George Mason University
- **Miftahul Alam Adib** - [GitHub Profile](https://github.com/Miftahul-adib)

---

## 🙏 Acknowledgments

- Sleep Health & Lifestyle Dataset contributors
- Scikit-learn, XGBoost, and LightGBM communities
- George Mason University & Shahjalal University of Science and Technology

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ by the Sleep Disorder Research Team

</div>
