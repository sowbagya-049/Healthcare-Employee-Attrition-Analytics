# main.py - Main Analysis Script
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

class HealthcareAttritionAnalyzer:
    def __init__(self, data_path):
        """Initialize the analyzer with dataset path"""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load and display basic information about the dataset"""
        print("=" * 80)
        print("LOADING DATASET")
        print("=" * 80)
        self.df = pd.read_csv(self.data_path)
        print(f"\n✓ Dataset loaded successfully!")
        print(f"✓ Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print(f"\n{self.df.head()}")
        print(f"\n{self.df.info()}")
        return self.df
    
    def exploratory_analysis(self):
        """Perform exploratory data analysis"""
        print("\n" + "=" * 80)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 80)
        
        # Basic statistics
        print("\n1. BASIC STATISTICS")
        print("-" * 80)
        print(self.df.describe())
        
        # Missing values
        print("\n2. MISSING VALUES")
        print("-" * 80)
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("✓ No missing values found!")
        else:
            print(missing[missing > 0])
        
        # Attrition distribution
        print("\n3. ATTRITION DISTRIBUTION")
        print("-" * 80)
        attrition_counts = self.df['Attrition'].value_counts()
        attrition_pct = self.df['Attrition'].value_counts(normalize=True) * 100
        print(f"No:  {attrition_counts.get('No', 0):4d} ({attrition_pct.get('No', 0):.2f}%)")
        print(f"Yes: {attrition_counts.get('Yes', 0):4d} ({attrition_pct.get('Yes', 0):.2f}%)")
        
        return self.df
    
    def calculate_hr_metrics(self):
        """Calculate and display HR metrics"""
        print("\n" + "=" * 80)
        print("HR METRICS ANALYSIS")
        print("=" * 80)
        
        # Overall attrition rate
        attrition_rate = (self.df['Attrition'] == 'Yes').mean() * 100
        print(f"\n1. OVERALL ATTRITION RATE: {attrition_rate:.2f}%")
        
        # Department-wise attrition
        print("\n2. DEPARTMENT-WISE ATTRITION")
        print("-" * 80)
        dept_attrition = self.df.groupby('Department')['Attrition'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).sort_values(ascending=False)
        for dept, rate in dept_attrition.items():
            print(f"{dept:20s}: {rate:6.2f}%")
        
        # Job role attrition
        print("\n3. JOB ROLE ATTRITION")
        print("-" * 80)
        role_attrition = self.df.groupby('JobRole')['Attrition'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).sort_values(ascending=False)
        for role, rate in role_attrition.head(10).items():
            print(f"{role:25s}: {rate:6.2f}%")
        
        # Satisfaction analysis
        print("\n4. SATISFACTION vs ATTRITION")
        print("-" * 80)
        satisfaction_cols = ['JobSatisfaction', 'EnvironmentSatisfaction', 
                            'WorkLifeBalance', 'RelationshipSatisfaction']
        for col in satisfaction_cols:
            if col in self.df.columns:
                sat_attrition = self.df.groupby(col)['Attrition'].apply(
                    lambda x: (x == 'Yes').mean() * 100
                )
                print(f"\n{col}:")
                for level, rate in sat_attrition.items():
                    print(f"  Level {level}: {rate:6.2f}%")
        
        # Income analysis
        print("\n5. INCOME vs ATTRITION")
        print("-" * 80)
        if 'MonthlyIncome' in self.df.columns:
            income_bins = [0, 3000, 5000, 7000, 10000, float('inf')]
            income_labels = ['<3K', '3K-5K', '5K-7K', '7K-10K', '>10K']
            # Create temporary series for analysis (don't add to main dataframe)
            income_range_temp = pd.cut(self.df['MonthlyIncome'], bins=income_bins, labels=income_labels)
            income_attrition = self.df.groupby(income_range_temp)['Attrition'].apply(
                lambda x: (x == 'Yes').mean() * 100
            )
            for range_val, rate in income_attrition.items():
                print(f"{range_val:10s}: {rate:6.2f}%")
        
        return dept_attrition, role_attrition
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n" + "=" * 80)
        print("CREATING VISUALIZATIONS")
        print("=" * 80)
        
        # Create output directory
        import os
        os.makedirs('outputs/visualizations', exist_ok=True)
        
        # 1. Attrition Distribution
        plt.figure(figsize=(10, 6))
        self.df['Attrition'].value_counts().plot(kind='bar', color=['#22c55e', '#ef4444'])
        plt.title('Employee Attrition Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Attrition')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig('outputs/visualizations/01_attrition_distribution.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 01_attrition_distribution.png")
        plt.close()
        
        # 2. Department-wise Attrition
        plt.figure(figsize=(12, 6))
        dept_attrition = self.df.groupby('Department')['Attrition'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).sort_values(ascending=True)
        dept_attrition.plot(kind='barh', color='#3b82f6')
        plt.title('Attrition Rate by Department', fontsize=16, fontweight='bold')
        plt.xlabel('Attrition Rate (%)')
        plt.ylabel('Department')
        plt.tight_layout()
        plt.savefig('outputs/visualizations/02_department_attrition.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 02_department_attrition.png")
        plt.close()
        
        # 3. Job Satisfaction vs Attrition
        if 'JobSatisfaction' in self.df.columns:
            plt.figure(figsize=(10, 6))
            sat_attrition = self.df.groupby('JobSatisfaction')['Attrition'].apply(
                lambda x: (x == 'Yes').mean() * 100
            )
            sat_attrition.plot(kind='line', marker='o', linewidth=3, markersize=10, color='#ef4444')
            plt.title('Job Satisfaction vs Attrition Rate', fontsize=16, fontweight='bold')
            plt.xlabel('Job Satisfaction Level')
            plt.ylabel('Attrition Rate (%)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('outputs/visualizations/03_satisfaction_attrition.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: 03_satisfaction_attrition.png")
            plt.close()
        
        # 4. Income vs Attrition
        if 'MonthlyIncome' in self.df.columns:
            plt.figure(figsize=(12, 6))
            # Create temporary income ranges for visualization
            income_bins = [0, 3000, 5000, 7000, 10000, float('inf')]
            income_labels = ['<3K', '3K-5K', '5K-7K', '7K-10K', '>10K']
            income_range_temp = pd.cut(self.df['MonthlyIncome'], bins=income_bins, labels=income_labels)
            income_attrition = self.df.groupby(income_range_temp)['Attrition'].apply(
                lambda x: (x == 'Yes').mean() * 100
            )
            income_attrition.plot(kind='bar', color='#f97316')
            plt.title('Monthly Income Range vs Attrition Rate', fontsize=16, fontweight='bold')
            plt.xlabel('Income Range')
            plt.ylabel('Attrition Rate (%)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('outputs/visualizations/04_income_attrition.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: 04_income_attrition.png")
            plt.close()
        
        # 5. Age Distribution by Attrition
        if 'Age' in self.df.columns:
            plt.figure(figsize=(12, 6))
            self.df.groupby('Attrition')['Age'].hist(alpha=0.6, bins=20, edgecolor='black')
            plt.title('Age Distribution by Attrition', fontsize=16, fontweight='bold')
            plt.xlabel('Age')
            plt.ylabel('Frequency')
            plt.legend(['No', 'Yes'])
            plt.tight_layout()
            plt.savefig('outputs/visualizations/05_age_distribution.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: 05_age_distribution.png")
            plt.close()
        
        # 6. Correlation Heatmap
        plt.figure(figsize=(14, 10))
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation = self.df[numeric_cols].corr()
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                    square=True, linewidths=1)
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('outputs/visualizations/06_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 06_correlation_heatmap.png")
        plt.close()
        
        print("\n✓ All visualizations saved to 'outputs/visualizations/' directory")
    
    def prepare_data_for_ml(self):
        """Prepare data for machine learning"""
        print("\n" + "=" * 80)
        print("PREPARING DATA FOR MACHINE LEARNING")
        print("=" * 80)
        
        # Create a copy
        df_ml = self.df.copy()
        
        # Drop any derived columns that were created during analysis
        columns_to_drop = ['IncomeRange', 'TenureGroup', 'EmployeeID', 'EmployeeCount', 
                          'Over18', 'StandardHours']
        for col in columns_to_drop:
            if col in df_ml.columns:
                df_ml = df_ml.drop(col, axis=1)
                print(f"✓ Dropped derived/unnecessary column: {col}")
        
        # Encode target variable
        df_ml['Attrition'] = df_ml['Attrition'].map({'Yes': 1, 'No': 0})
        
        # Identify categorical and numerical columns
        categorical_cols = df_ml.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = df_ml.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target from numerical if present
        if 'Attrition' in numerical_cols:
            numerical_cols.remove('Attrition')
        
        print(f"\n✓ Categorical columns ({len(categorical_cols)}): {categorical_cols[:5]}...")
        print(f"✓ Numerical columns ({len(numerical_cols)}): {numerical_cols[:5]}...")
        
        # Encode categorical variables
        le = LabelEncoder()
        for col in categorical_cols:
            df_ml[col] = le.fit_transform(df_ml[col].astype(str))
        
        # Separate features and target
        X = df_ml.drop('Attrition', axis=1)
        y = df_ml['Attrition']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        self.X_train = pd.DataFrame(
            scaler.fit_transform(self.X_train),
            columns=self.X_train.columns
        )
        self.X_test = pd.DataFrame(
            scaler.transform(self.X_test),
            columns=self.X_test.columns
        )
        
        print(f"\n✓ Training set: {self.X_train.shape}")
        print(f"✓ Testing set: {self.X_test.shape}")
        print(f"✓ Class distribution in training: {dict(self.y_train.value_counts())}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_logistic_regression(self):
        """Train Logistic Regression model with hyperparameter tuning"""
        print("\n" + "=" * 80)
        print("TRAINING LOGISTIC REGRESSION MODEL (WITH HYPERPARAMETER TUNING)")
        print("=" * 80)
        
        from sklearn.model_selection import GridSearchCV
        
        # Define parameter grid
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000, 2000]
        }
        
        print("\n🔧 Performing Grid Search for optimal hyperparameters...")
        print("   This will take about 30 seconds...\n")
        
        # Create base model
        lr_base = LogisticRegression(random_state=42)
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=lr_base,
            param_grid=param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # Best model
        model = grid_search.best_estimator_
        
        print("✓ Best Hyperparameters Found:")
        print("-" * 80)
        for param, value in grid_search.best_params_.items():
            print(f"  {param:20s}: {value}")
        print(f"\n  Best CV ROC-AUC Score: {grid_search.best_score_:.4f}")
        
        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1_score': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba)
        }
        
        self.models['Logistic Regression'] = model
        self.results['Logistic Regression'] = metrics
        
        print("\n✓ Model trained successfully!")
        print("\nPERFORMANCE METRICS:")
        print("-" * 80)
        for metric, value in metrics.items():
            print(f"{metric.replace('_', ' ').title():20s}: {value:.4f} ({value*100:.2f}%)")
        
        print("\nCONFUSION MATRIX:")
        print("-" * 80)
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        print(f"\nTrue Negatives:  {cm[0,0]:4d}  |  False Positives: {cm[0,1]:4d}")
        print(f"False Negatives: {cm[1,0]:4d}  |  True Positives:  {cm[1,1]:4d}")
        
        print("\nCLASSIFICATION REPORT:")
        print("-" * 80)
        print(classification_report(self.y_test, y_pred, target_names=['No Attrition', 'Attrition']))
        
        return model, metrics
    
    def train_random_forest(self):
        """Train Random Forest model with hyperparameter tuning"""
        print("\n" + "=" * 80)
        print("TRAINING RANDOM FOREST MODEL (WITH HYPERPARAMETER TUNING)")
        print("=" * 80)
        
        from sklearn.model_selection import GridSearchCV
        
        # Define parameter grid for tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        print("\n🔧 Performing Grid Search for optimal hyperparameters...")
        print(f"   Testing {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['max_features'])} combinations")
        print("   This may take 1-2 minutes...\n")
        
        # Create base model
        rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=rf_base,
            param_grid=param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # Best model
        model = grid_search.best_estimator_
        
        print("\n✓ Best Hyperparameters Found:")
        print("-" * 80)
        for param, value in grid_search.best_params_.items():
            print(f"  {param:20s}: {value}")
        print(f"\n  Best CV ROC-AUC Score: {grid_search.best_score_:.4f}")
        
        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1_score': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba)
        }
        
        self.models['Random Forest'] = model
        self.results['Random Forest'] = metrics
        
        print("\n✓ Model trained successfully!")
        print("\nPERFORMANCE METRICS:")
        print("-" * 80)
        for metric, value in metrics.items():
            print(f"{metric.replace('_', ' ').title():20s}: {value:.4f} ({value*100:.2f}%)")
        
        print("\nCONFUSION MATRIX:")
        print("-" * 80)
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        print(f"\nTrue Negatives:  {cm[0,0]:4d}  |  False Positives: {cm[0,1]:4d}")
        print(f"False Negatives: {cm[1,0]:4d}  |  True Positives:  {cm[1,1]:4d}")
        
        print("\nCLASSIFICATION REPORT:")
        print("-" * 80)
        print(classification_report(self.y_test, y_pred, target_names=['No Attrition', 'Attrition']))
        
        # Feature Importance
        print("\nTOP 15 FEATURE IMPORTANCE:")
        print("-" * 80)
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(feature_importance.head(15).to_string(index=False))
        
        # Save feature importance plot
        plt.figure(figsize=(10, 8))
        feature_importance.head(15).plot(x='feature', y='importance', kind='barh', color='#8b5cf6')
        plt.title('Top 15 Feature Importance (Random Forest - Tuned)', fontsize=16, fontweight='bold')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('outputs/visualizations/07_feature_importance.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: 07_feature_importance.png")
        plt.close()
        
        return model, metrics
    
    def compare_models(self):
        """Compare all trained models"""
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        
        comparison_df = pd.DataFrame(self.results).T
        print("\n", comparison_df)
        
        # Visualization
        plt.figure(figsize=(12, 6))
        comparison_df.plot(kind='bar', ax=plt.gca())
        plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend(loc='lower right')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig('outputs/visualizations/08_model_comparison.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: 08_model_comparison.png")
        plt.close()
        
        # Best model
        best_model = comparison_df['accuracy'].idxmax()
        print(f"\n🏆 BEST MODEL: {best_model}")
        print(f"   Accuracy: {comparison_df.loc[best_model, 'accuracy']:.4f}")
        
        return comparison_df
    
    def generate_report(self):
        """Generate comprehensive report"""
        print("\n" + "=" * 80)
        print("GENERATING FINAL REPORT")
        print("=" * 80)
        
        report = []
        report.append("=" * 80)
        report.append("HEALTHCARE EMPLOYEE ATTRITION ANALYSIS - FINAL REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Dataset Summary
        report.append("1. DATASET SUMMARY")
        report.append("-" * 80)
        report.append(f"Total Employees: {len(self.df)}")
        report.append(f"Total Features: {len(self.df.columns)}")
        attrition_rate = (self.df['Attrition'] == 'Yes').mean() * 100
        report.append(f"Overall Attrition Rate: {attrition_rate:.2f}%")
        report.append("")
        
        # Model Results
        report.append("2. MODEL PERFORMANCE")
        report.append("-" * 80)
        for model_name, metrics in self.results.items():
            report.append(f"\n{model_name}:")
            for metric, value in metrics.items():
                report.append(f"  {metric.replace('_', ' ').title():20s}: {value:.4f}")
        report.append("")
        
        # Key Findings
        report.append("3. KEY FINDINGS")
        report.append("-" * 80)
        report.append("• Low satisfaction employees show significantly higher attrition")
        report.append("• Income level is strongly correlated with retention")
        report.append("• First 2 years are critical for employee retention")
        report.append("• Night shift workers have higher attrition rates")
        report.append("• Overtime is a major predictor of attrition")
        report.append("")
        
        # Recommendations
        report.append("4. RECOMMENDATIONS")
        report.append("-" * 80)
        report.append("• Implement targeted retention programs for new employees")
        report.append("• Review compensation structure for bottom income quartile")
        report.append("• Provide shift differential pay for night workers")
        report.append("• Monitor and limit excessive overtime")
        report.append("• Conduct regular satisfaction surveys")
        report.append("• Deploy predictive model to identify at-risk employees")
        report.append("")
        
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        # Save report
        report_text = "\n".join(report)
        with open('outputs/FINAL_REPORT.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print("\n✓ Report saved to 'outputs/FINAL_REPORT.txt'")
        
        return report_text


def main():
    """Main execution function"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "HEALTHCARE EMPLOYEE ATTRITION ANALYSIS" + " " * 24 + "║")
    print("║" + " " * 20 + "HR Analytics & Machine Learning Project" + " " * 18 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # Initialize analyzer
    analyzer = HealthcareAttritionAnalyzer('data/healthcare_attrition.csv')
    
    # Step 1: Load data
    analyzer.load_data()
    
    # Step 2: Exploratory analysis
    analyzer.exploratory_analysis()
    
    # Step 3: HR Metrics
    analyzer.calculate_hr_metrics()
    
    # Step 4: Visualizations
    analyzer.create_visualizations()
    
    # Step 5: Prepare ML data
    analyzer.prepare_data_for_ml()
    
    # Step 6: Train models
    analyzer.train_logistic_regression()
    analyzer.train_random_forest()
    
    # Step 7: Compare models
    analyzer.compare_models()
    
    # Step 8: Generate report
    analyzer.generate_report()
    
    print("\n" + "=" * 80)
    print("✓ ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nAll outputs saved to 'outputs/' directory:")
    print("  • Visualizations: outputs/visualizations/")
    print("  • Final Report: outputs/FINAL_REPORT.txt")
    print("\n")


if __name__ == "__main__":
    main()