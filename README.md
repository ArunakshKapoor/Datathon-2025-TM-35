# Datathon-Lifeline-2025---Team-TM-35
# Cardiotography (CTG) Analysis Project

A comprehensive machine learning project for analyzing Cardiotography data to predict fetal health states.

## Project Structure

```
├── Data Exploration/
│   └── EDA.ipynb                 # Complete exploratory data analysis notebook (data cleaning, visualization, data augmentation , feature selection , data standardization, Hyperparameter tuning, model training & testing)
│
├── Training and Testing/
│   └── ctg_heart_pipeline.py     # Main pipeline script for CTG heart data processing and model training (Training and Testing file)
│
├── Dataset/                      # Contains the Cardiotography dataset files
├── images/                       # Visualizations and plots generated during analysis
├── model_weights/                # Saved model weights and checkpoints
├── .vscode/                      # VS Code configuration files
├── .venv/                        # Python virtual environment
├── LICENSE                       # Project license
└── README.md                     # This file
```

## Key Components

### 1. Data Exploration (`Data Exploration/EDA.ipynb`)
- **Complete end-to-end analysis** of the Cardiotography dataset
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA) with visualizations
- Feature engineering and selection
- Data Standardization
- Data Augmentation
- Model training and evaluation
- Performance metrics and results

### 2. Model Pipeline (`Training and Testing/ctg_heart_pipeline.py`)
- **Production-ready pipeline** for CTG data processing
- Automated data preprocessing
- Model training workflow
- Evaluation and testing procedures

## Getting Started

1. **Setup Environment**
   ```bash
   # Activate virtual environment
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Install dependencies (if needed)
   pip install -r requirements.txt
   ```

2. **Explore the Data**
   - Please start with `Data Exploration/EDA.ipynb` for comprehensive analysis
   - Understand data patterns, distributions, and relationships

3. **Run the Pipeline**
   ```
    refer to "Training and Testing/ctg_heart_pipeline.py"
   ```

## Dataset Information

The project uses Cardiotography (CTG) data for fetal health classification. The dataset includes various measurements from cardiotocograms used to assess fetal well-being.

## Model Outputs

- **Trained Models**: Saved in `model_weights/`
- **Visualizations**: Generated plots and charts in `images/`
- **Analysis Results**: Comprehensive findings in the EDA notebook

## License

See `LICENSE` file for detailed license information.

## Next Steps

1. Review the EDA notebook for detailed analysis insights
2. Run the pipeline script to reproduce model training
3. Check the model weights for pre-trained models
4. Refer to visualizations in the images folder for data understanding

For any questions or issues, please refer to the detailed documentation within each component folder.