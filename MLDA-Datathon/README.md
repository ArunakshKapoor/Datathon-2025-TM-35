# CTG Fetal Health Classification Pipeline

A comprehensive machine learning pipeline for classifying fetal health states using Cardiotocography (CTG) data. This implementation uses Extra Trees Classifier with advanced preprocessing, feature selection, and handling of class imbalance.

## 📋 Overview

This pipeline processes CTG data to classify fetal health into three categories:
- **Normal** (NSP=1)
- **Suspect** (NSP=2) 
- **Pathological** (NSP=3)

The system emphasizes medical safety with special focus on minimizing False Negatives for the Pathological class, which represents critical fetal distress cases.

## 🏗️ Architecture

### Data Flow
1. **Data Loading & Cleaning** → Remove outliers, handle missing values
2. **Feature Selection** → Top 18 most important features
3. **Data Standardization** → StandardScaler for normalization
4. **Class Balancing** → Borderline-SMOTE for imbalanced data
5. **Model Training** → Extra Trees Classifier
6. **Evaluation** → Comprehensive medical safety analysis
7. **Model Persistence** → Save/Load trained models

### Key Features
- **Automated Feature Selection**: Uses top 18 features identified through EDA
- **Class Imbalance Handling**: Borderline-SMOTE for better minority class representation
- **Medical Safety Focus**: Special emphasis on Pathological class detection
- **Model Persistence**: Save and reload trained models with all components
- **Comprehensive Evaluation**: Multiple metrics with medical interpretation

## 🚀 Installation & Dependencies

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib openpyxl
```

### Required Libraries
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `scikit-learn`: Machine learning algorithms
- `imbalanced-learn`: SMOTE implementation
- `matplotlib`: Visualization
- `seaborn`: Enhanced visualizations
- `joblib`: Model serialization

## 💻 Usage

### Basic Training

```python
from ctg_pipeline import CTGFetalHealthPipeline

# Initialize pipeline
pipeline = CTGFetalHealthPipeline(random_state=42, n_estimators=300)

# Train model
metrics = pipeline.train('path_to_your_data.xlsx', save_model=True)

# Get model summary
print(pipeline.get_model_summary())
```

### Loading Pre-trained Model

```python
# Initialize and load existing model
pipeline = CTGFetalHealthPipeline()
pipeline.load_model('ctg_extra_trees_model.pkl')

# Check model status
print(pipeline.get_model_summary())
```

### Making Predictions

#### Batch Predictions
```python
# Assuming X_new is your new data DataFrame
predictions, probabilities = pipeline.predict(X_new)
```

#### Single Prediction
```python
sample_features = {
    'LB': 120, 'AC': 0, 'FM': 0, 'UC': 0, 'DL': 0, 'DS': 0, 'DP': 0,
    'ASTV': 12, 'MSTV': 0.5, 'ALTV': 0, 'MLTV': 2.4, 'Width': 10,
    'Min': 50, 'Max': 170, 'Nmax': 0, 'Nzeros': 0, 'Mode': 120,
    'Mean': 137, 'Median': 121, 'Variance': 73, 'Tendency': 1
}

result = pipeline.predict_single(sample_features)
print(f"Prediction: {result['prediction_label']}")
print(f"Probabilities: {result['probabilities']}")
```

## 📊 Model Details

### Algorithm: Extra Trees Classifier
- **Ensemble Method**: Multiple decision trees with random splits
- **Hyperparameters**:
  - `n_estimators`: 300
  - `max_depth`: None (unlimited)
  - `max_features`: 'sqrt'
  - `class_weight`: 'balanced'
  - `bootstrap`: False

### Feature Selection
The pipeline uses the top 18 most important features identified through exploratory data analysis:
```python
['ASTV', 'AC.1', 'MSTV', 'ALTV', 'LB', 'Mean', 'Mode', 'UC.1', 
 'Median', 'Nmax', 'Min', 'Max', 'Width', 'Tendency', 'FM.1', 
 'Variance', 'MLTV', 'DL.1']
```

### Data Preprocessing
1. **Outlier Removal**: Interquartile Range (IQR) method
2. **Standardization**: StandardScaler (zero mean, unit variance)
3. **Class Balancing**: Borderline-SMOTE with k_neighbors=5

## 📈 Evaluation Metrics

### Primary Metrics (Recommended)
- **Macro Averages**: Equal weight to all classes
- **Class-wise Metrics**: Individual performance per class
- **Pathological Recall**: Critical safety metric

### Medical Safety Analysis
- False Negative analysis for each class
- Alerts for poor Pathological class detection
- Recall thresholds for medical safety:
  - < 0.7: CRITICAL ALERT
  - < 0.85: WARNING
  - ≥ 0.85: GOOD

## 💾 Model Persistence

### Saving Models
Models are automatically saved after training and include:
- Trained Extra Trees model
- Fitted StandardScaler
- Feature names and selections
- Training metrics
- Model parameters

### File Structure
```
model_weights/
└── ctg_extra_trees_model.pkl
    ├── model: Trained ExtraTreesClassifier
    ├── scaler: Fitted StandardScaler
    ├── top_k_features: Selected feature list
    ├── feature_names: All original features
    ├── training_metrics: Performance metrics
    └── model_params: Training parameters
```

## 🏥 Medical Considerations

### Critical Focus Areas
1. **Pathological Class Detection**: Highest priority for patient safety
2. **False Negative Minimization**: Missing critical cases has severe consequences
3. **Model Interpretability**: Feature importance for clinical understanding

### Safety Thresholds
- **Pathological Recall ≥ 0.85**: Acceptable for clinical use
- **Pathological Recall < 0.7**: Requires immediate model improvement
- **Macro F1 > 0.80**: Good overall performance

## 🔧 Customization

### Modifying Hyperparameters
```python
pipeline = CTGFetalHealthPipeline(
    n_estimators=500,
    random_state=123,
    model_save_path='custom_models'
)
```

### Changing Feature Selection
Override the `top_k_features` attribute after initialization:
```python
pipeline.top_k_features = ['your', 'custom', 'features']
```

## 📋 Output Examples

### Training Output
```
COMPREHENSIVE MODEL EVALUATION
======================================================================
Overall Accuracy: 0.9450

CLASS-WISE METRICS (Most Important):
  Normal       - Precision: 0.9600, Recall: 0.9700, F1: 0.9650
  Suspect      - Precision: 0.8800, Recall: 0.8500, F1: 0.8640
  Pathological - Precision: 0.8200, Recall: 0.7800, F1: 0.7990

MEDICAL SAFETY ANALYSIS
==================================================
Pathological:
  True Positives: 39
  False Negatives: 11 (Missed Pathological cases)
  False Positives: 9 (Incorrectly predicted as Pathological)
  Recall: 0.7800
  WARNING: Recall 0.7800 for pathological cases
  Consider improving detection of critical cases
```

## 🚨 Error Handling

The pipeline includes comprehensive error handling for:
- Missing data files
- Invalid feature inputs
- Model loading failures
- Prediction errors
- Data formatting issues

## 📝 License

This project is intended for educational and research purposes. Clinical use requires additional validation and regulatory approval.

## 🤝 Contributing

When extending this pipeline:
1. Maintain the medical safety focus
2. Preserve comprehensive evaluation metrics
3. Ensure model interpretability
4. Document all changes thoroughly

## 📚 References

1. Cardiotocography Dataset Source
2. Scikit-learn Documentation
3. Imbalanced-learn Documentation
4. Clinical guidelines for fetal monitoring

---

**Note**: This pipeline is designed for research purposes. Clinical deployment requires rigorous validation and regulatory compliance.