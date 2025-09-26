# 🧬 Tumor Diagnostics with Machine Learning

## 📌 Overview
This project applies machine learning to tumor diagnostics using the **Breast Cancer Wisconsin dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original)).  
The objective was to compare several classification models and identify the most reliable approach for predicting whether a tumor is **benign** or **malignant**.

---

## 🎯 Objectives
- Preprocess clinical cytology data for machine learning.
- Train and evaluate multiple classification models.
- Compare results across accuracy, precision, recall, F1-score, and ROC-AUC.
- Highlight the most suitable model for tumor diagnostics.

---

## 📊 Methods
- **Data**: 683 tumor samples, 10 cellular features (e.g., clump thickness, uniformity of cell size, bare nuclei).
- **Models tested**:
  - Logistic Regression
  - K-Nearest Neighbors
  - Support Vector Machine (SVM)
  - Random Forest
- **Evaluation metrics**: Accuracy, Precision, Recall, F1-score, AUC.
- **Implementation**: Python (scikit-learn) in Jupyter Notebook.

---

## 🔑 Key Results
| Model                | Accuracy | Recall | F1-score | AUC  |
|-----------------------|----------|--------|----------|------|
| Logistic Regression   | 95.6%    | 94.0%  | 94.0%    | 0.99 |
| K-Nearest Neighbors   | 95.6%    | 96.0%  | 94.1%    | 0.99 |
| Support Vector Mach.  | 95.6%    | 98.0%  | 94.2%    | 0.99 |
| **Random Forest**     | **97.1%**| **98.0%** | **96.1%** | **1.00** |

- **Random Forest** achieved the strongest overall performance, balancing high recall and precision.  
- **Recall** is critical in tumor diagnostics — missing malignant cases must be minimized.  
- Results varied slightly with different data splits, underlining the importance of cross-validation.

---

## 💡 Implications
- Ensemble methods like Random Forest are highly effective for structured medical datasets.
- Multiple models (Random Forest, SVM) can achieve high accuracy in tumor diagnostics.
- Highlights how machine learning can support reliable, automated clinical decision-making.

---

## 📁 Repository Structure
```
├── data/
│ └── Tumor_Data_Sample.csv # 20 sample rows from dataset
├── notebooks/
│ ├── k-nearest_neighbor.ipynb
│ ├── logistic_regression.ipynb
│ ├── random_forest.ipynb
│ └── support_vector_machine.ipynb
├── report/
│ ├── Tumor_Diagnostics_Executive_Summary.pdf
│ └── Tumor_Diagnostics_Classification_Report.pdf
├── README.md # Link to full dataset on UCI ML Repository
└── LICENSE
```
---

## 🚀 Getting Started

### 1. Clone Repository
```bash
git clone https://github.com/asahedev/tumor-diagnostics-classification.git
cd tumor-diagnostics-classification
```

### 2. Run Jupyter Notebooks

---

## 📚 References

[UCI Machine Learning Repository: Breast Cancer Wisconsin Datase](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original))

[Scikit-learn Documentation](https://scikit-learn.org/stable/)

---

## 📝 License

Distributed under the MIT License. See LICENSE for details.
