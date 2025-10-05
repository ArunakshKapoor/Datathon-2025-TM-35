import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.preprocessing import StandardScaler

class CTGFetalHealthPipeline:
    """
    A complete machine learning pipeline for the Cardiotocography (CTG) dataset.
    Performs data cleaning, feature selection (top 18 by importance), data augmentation (SMOTE),
    trains an ExtraTreesClassifier (selected for highest macro F1-score as per the EDA notebook, and evaluates with
    emphasis on low False Negatives for the Pathological class (NSP=3).
    """

    def __init__(self, random_state=42, n_estimators=300, model_save_path='model_weights'):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.model_save_path = model_save_path
        self.top_k_features = ['ASTV', 'AC.1', 'MSTV', 'ALTV', 'LB', 'Mean', 'Mode', 'UC.1', 'Median', 'Nmax', 'Min', 'Max', 'Width', 'Tendency', 'FM.1', 'Variance', 'MLTV', 'DL.1']
        self.model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            bootstrap=False,
            max_depth=None,
            max_features='sqrt',
            min_samples_leaf=1,
            min_samples_split=2,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
        self.smote = BorderlineSMOTE(
            random_state=42,
            kind='borderline-1',
            k_neighbors=5,
            m_neighbors=10
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        self.X_test = None
        self.y_test = None
        self.X_train = None
        self.y_train = None
        self.training_metrics = None

    def load_and_clean_data(self, file_path):
        """
        Load the CTG Dataset and perform cleaning
        - removing any null values
        - removing any duplicated
        - removing rows with NaN and missing values 
        - dropping irrelevant columns
        """
        df = pd.read_excel(file_path, sheet_name='Data', header=1)
        unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
        df_new = df.drop(columns=unnamed_cols)
        # Drop non feature columns
        df_new = df_new.drop(columns=['b', 'e', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'DR', 'A', 'B', 'C', 'D','E', 'AD', 'DE', 'LD', 'FS', 'SUSP', 'CLASS'])
        # Removing rows with NaN values for columns 
        df_new = df_new.dropna()

        # Drop duplicates 
        df_new.drop_duplicates(keep='first', inplace=True)

        # Set feature names and target
        self.feature_names = [col for col in df_new.columns if col != 'NSP']
        # removing outliers from it 
        df_iqr = self.remove_outliers_iqr(df_new, df_new.columns)
        return df_iqr
    
    @staticmethod
    def remove_outliers_iqr(df, cols):
        df_clean = df.copy()

        for col in cols:
            if col == 'NSP':
                continue
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1 
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
        return df_clean

    def save_model(self, filename='ctg_extra_trees_model.pkl'):
        """
        Save the trained model, scaler, and feature information
        """
        if not self.is_trained:
            print("Model not trained yet. Cannot save.")
            return False
        
        # Create directory if it doesn't exist
        os.makedirs(self.model_save_path, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'top_k_features': self.top_k_features,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'model_params': {
                'n_estimators': self.n_estimators,
                'random_state': self.random_state
            }
        }
        
        file_path = os.path.join(self.model_save_path, filename)
        joblib.dump(model_data, file_path)
        print(f"Model saved successfully to: {file_path}")
        return True

    def load_model(self, filename='ctg_extra_trees_model.pkl'):
        """
        Load a previously trained model
        """
        file_path = os.path.join(self.model_save_path, filename)
        
        try:
            model_data = joblib.load(file_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.top_k_features = model_data['top_k_features']
            self.feature_names = model_data['feature_names']
            self.training_metrics = model_data['training_metrics']
            self.is_trained = True
            
            print(f"Model loaded successfully from: {file_path}")
            print("Model training metrics from saved model:")
            self.print_metrics_summary(self.training_metrics)
            return True
            
        except FileNotFoundError:
            print(f"Model file not found: {file_path}")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def train(self, file_path, test_size=0.25, save_model=True):
        """
        Train the pipeline: clean, split, augment, select features, train model.
        Returns trained model and evaluation metrics.
        """
        print("Loading and cleaning data...")
        df_iqr = self.load_and_clean_data(file_path)
        
        X = df_iqr.drop(columns=['NSP'])
        y = df_iqr['NSP']
        
        # split the data in train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )
        
        # Store for later use
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

        # standardizing using standard scaler
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create scaled DataFrames
        df_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
        df_train_scaled['NSP'] = y_train.reset_index(drop=True)

        df_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
        df_test_scaled['NSP'] = y_test.reset_index(drop=True)
        
        # Select top features
        X_train_reduced = df_train_scaled[self.top_k_features]
        X_test_reduced = df_test_scaled[self.top_k_features]
        
        print(f"Original shape: {df_train_scaled.shape}")
        print(f"Reduced shape: {X_train_reduced.shape}")

        # Apply Borderline-SMOTE
        print("Applying Borderline-SMOTE...")
        X_train_resampled, y_train_resampled = self.smote.fit_resample(X_train_reduced, y_train)
        
        print("Borderline-SMOTE Applied Successfully!")
        print(f"Training samples: {X_train_reduced.shape[0]} -> {X_train_resampled.shape[0]}")
        print(f"Class distribution after SMOTE: {pd.Series(y_train_resampled).value_counts().sort_index()}")

        # Train the model
        print("Training Extra Trees Classifier...")
        self.model.fit(X_train_resampled, y_train_resampled)
        self.is_trained = True
        
        # Evaluate on test set
        print("Evaluating model...")
        y_pred = self.model.predict(X_test_reduced)
        
        # Comprehensive evaluation
        self.training_metrics = self.evaluate_model(y_test, y_pred, X_test_reduced)
        
        # Save model if requested
        if save_model:
            self.save_model()
        
        return self.training_metrics

    def evaluate_model(self, y_true, y_pred, X_test=None):
        """
        Comprehensive evaluation with emphasis on medical metrics
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*70)
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # All averaging methods
        precision_macro = precision_score(y_true, y_pred, average='macro')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        
        precision_weighted = precision_score(y_true, y_pred, average='weighted')
        recall_weighted = recall_score(y_true, y_pred, average='weighted')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        # Class-wise metrics (most important)
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Print results
        print(f"Overall Accuracy: {accuracy:.4f}")
        
        print("CLASS-WISE METRICS (Most Important):")
        class_names = ['Normal', 'Suspect', 'Pathological']
        for i, name in enumerate(class_names):
            print(f"  {name:12} - Precision: {precision_per_class[i]:.4f}, "
                  f"Recall: {recall_per_class[i]:.4f}, F1: {f1_per_class[i]:.4f}")
        
        print("AVERAGED METRICS:")
        print(f"  MACRO       - Precision: {precision_macro:.4f}, "
              f"Recall: {recall_macro:.4f}, F1: {f1_macro:.4f} (RECOMMENDED)")
        print(f"  WEIGHTED    - Precision: {precision_weighted:.4f}, "
              f"Recall: {recall_weighted:.4f}, F1: {f1_weighted:.4f}")
        
        # Medical safety analysis
        self.medical_safety_analysis(y_true, y_pred, cm)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm, class_names)
        
        # Feature importance
        if X_test is not None and hasattr(self.model, 'feature_importances_'):
            self.plot_feature_importance()
        
        return {
            'accuracy': accuracy,
            'macro': {'precision': precision_macro, 'recall': recall_macro, 'f1': f1_macro},
            'weighted': {'precision': precision_weighted, 'recall': recall_weighted, 'f1': f1_weighted},
            'per_class': {
                'precision': precision_per_class,
                'recall': recall_per_class, 
                'f1': f1_per_class
            },
            'confusion_matrix': cm
        }

    def medical_safety_analysis(self, y_true, y_pred, cm):
        """
        Analyze medical safety aspects, especially False Negatives
        """
        print("\n" + "="*50)
        print("MEDICAL SAFETY ANALYSIS")
        print("="*50)
        
        class_names = ['Normal', 'Suspect', 'Pathological']
        
        for i, class_name in enumerate(class_names):
            # False Negatives: Actual is class i but predicted as other classes
            fn = cm[i, :].sum() - cm[i, i]
            
            # False Positives: Predicted as class i but actual is other classes
            fp = cm[:, i].sum() - cm[i, i]
            
            # True Positives and Recall
            tp = cm[i, i]
            recall = recall_score(y_true, y_pred, average=None)[i]
            
            print(f"{class_name}:")
            print(f"  True Positives: {tp}")
            print(f"  False Negatives: {fn} (Missed {class_name} cases)")
            print(f"  False Positives: {fp} (Incorrectly predicted as {class_name})")
            print(f"  Recall: {recall:.4f}")
            
            # Medical alerts
            if class_name == 'Pathological':
                if recall < 0.7:
                    print(f"  CRITICAL ALERT: Missing {fn} pathological cases!")
                    print("  This could lead to undetected fetal distress!")
                elif recall < 0.85:
                    print(f"  WARNING: Recall {recall:.4f} for pathological cases")
                    print("  Consider improving detection of critical cases")
                else:
                    print(f"  GOOD: High recall ({recall:.4f}) for pathological cases")

    def plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Number of Cases'})
        plt.title('Confusion Matrix - Extra Trees Classifier')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, top_n=15):
        """Plot feature importance"""
        if not self.is_trained:
            print("Model not trained yet!")
            return
            
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importance - Extra Trees Classifier")
        bars = plt.bar(range(top_n), importance[indices][:top_n])
        plt.xticks(range(top_n), [self.top_k_features[i] for i in indices[:top_n]], rotation=45)
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        
        # Add value labels on bars
        for bar, importance_val in zip(bars, importance[indices][:top_n]):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{importance_val:.4f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
        # Print top features
        print(f"Top {top_n} Most Important Features:")
        for i in range(min(top_n, len(importance))):
            print(f"{i+1:2d}. {self.top_k_features[indices[i]]:20} - {importance[indices[i]]:.4f}")

    def predict(self, X_new):
        """
        Make predictions on new data
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Scale the data
        X_scaled = self.scaler.transform(X_new)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        # Select top features
        X_reduced = X_scaled_df[self.top_k_features]
        
        # Make predictions
        predictions = self.model.predict(X_reduced)
        probabilities = self.model.predict_proba(X_reduced)
        
        return predictions, probabilities

    def predict_single(self, features_dict):
        """
        Make prediction for a single sample
        features_dict: dictionary with feature names as keys and values
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Create DataFrame from single sample
        X_single = pd.DataFrame([features_dict])
        
        # Scale and select features
        X_scaled = self.scaler.transform(X_single)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names)
        X_reduced = X_scaled_df[self.top_k_features]
        
        # Make prediction
        prediction = self.model.predict(X_reduced)[0]
        probability = self.model.predict_proba(X_reduced)[0]
        
        class_names = {1: 'Normal', 2: 'Suspect', 3: 'Pathological'}
        
        return {
            'prediction': prediction,
            'prediction_label': class_names[prediction],
            'probabilities': {
                'Normal': probability[0],
                'Suspect': probability[1],
                'Pathological': probability[2]
            }
        }

    def get_model_summary(self):
        """
        Get summary of the trained model
        """
        if not self.is_trained:
            return "Model not trained yet."
        
        summary = f"""
        MODEL SUMMARY - Extra Trees Classifier
        {'='*40}
        Number of estimators: {self.n_estimators}
        Features used: {len(self.top_k_features)}
        Classes: Normal (1), Suspect (2), Pathological (3)
        Training status: {'Trained' if self.is_trained else 'Not trained'}
        Model saved: {'Yes' if self.training_metrics else 'No'}
        
        Top 5 Features:
        """
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            indices = np.argsort(importance)[::-1]
            for i in range(min(5, len(importance))):
                summary += f"  {i+1}. {self.top_k_features[indices[i]]}\n"
        
        return summary

    def print_metrics_summary(self, metrics):
        """
        Print a summary of model metrics
        """
        if not metrics:
            print("No metrics available.")
            return
        
        print("MODEL PERFORMANCE SUMMARY")
        print("=" * 30)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1: {metrics['macro']['f1']:.4f}")
        print(f"Weighted F1: {metrics['weighted']['f1']:.4f}")
        
        print("Class-wise Performance:")
        class_names = ['Normal', 'Suspect', 'Pathological']
        for i, name in enumerate(class_names):
            print(f"  {name}: F1 = {metrics['per_class']['f1'][i]:.4f}, "
                  f"Recall = {metrics['per_class']['recall'][i]:.4f}")

# Example usage and testing
if __name__ == "__main__":
    # Initialize the pipeline
    pipeline = CTGFetalHealthPipeline(random_state=42, n_estimators=300)
    
    # Option 1: Train a new model
    try:
        print("Training new model...")
        metrics = pipeline.train('CTG.xls', save_model=True)
        
        # Print model summary
        print(pipeline.get_model_summary())
        
    except FileNotFoundError:
        print("Training file not found. Trying to load existing model...")
        
        # Option 2: Load existing model
        if pipeline.load_model():
            print("Successfully loaded pre-trained model.")
        else:
            print("No pre-trained model found. Please provide training data.")
    
    # Example of making predictions
    if pipeline.is_trained:
        print("Model is ready for predictions.")
        
        # Example single prediction
        sample_features = {
            'LB': 120, 'AC': 0, 'FM': 0, 'UC': 0, 'DL': 0, 'DS': 0, 'DP': 0,
            'ASTV': 12, 'MSTV': 0.5, 'ALTV': 0, 'MLTV': 2.4, 'Width': 10,
            'Min': 50, 'Max': 170, 'Nmax': 0, 'Nzeros': 0, 'Mode': 120,
            'Mean': 137, 'Median': 121, 'Variance': 73, 'Tendency': 1
        }
        
        try:
            result = pipeline.predict_single(sample_features)
            print(f"Prediction: {result['prediction_label']}")
            print(f"Probabilities: {result['probabilities']}")
        except Exception as e:
            print(f"Prediction error: {e}")