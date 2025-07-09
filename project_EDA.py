import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from contextlib import redirect_stdout
import warnings
import os

warnings.filterwarnings('ignore')

# ================================
# Ensure Plots Folder Exists
# ================================
if not os.path.exists('plots'):
    os.makedirs('plots')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the dataset
# Note: Replace 'heart_failure_clinical_records_dataset.csv' with your actual file path
df = pd.read_csv('heart_failure.csv')

# ================================
# VISUALIZATION FUNCTIONS
# ================================

def create_comprehensive_plots():
    """Create comprehensive visualizations for the heart failure dataset"""
    
    # Set up the plotting parameters
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # 1. Target Distribution
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Heart Failure Dataset - Exploratory Data Analysis', fontsize=16, fontweight='bold')
    
    # Target distribution
    target_counts = df['DEATH_EVENT'].value_counts()
    axes[0, 0].pie(target_counts.values, labels=['Survived', 'Died'], autopct='%1.1f%%', 
                   colors=['lightgreen', 'lightcoral'], startangle=90)
    axes[0, 0].set_title('Patient Survival Distribution')
    
    # Age distribution
    axes[0, 1].hist(df['age'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].set_title('Age Distribution')
    axes[0, 1].set_xlabel('Age (years)')
    axes[0, 1].set_ylabel('Frequency')
    
    # Age vs Death Event
    df.boxplot(column='age', by='DEATH_EVENT', ax=axes[0, 2])
    axes[0, 2].set_title('Age Distribution by Survival Status')
    axes[0, 2].set_xlabel('Death Event (0=Survived, 1=Died)')
    
    # Ejection Fraction distribution
    axes[1, 0].hist(df['ejection_fraction'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 0].set_title('Ejection Fraction Distribution')
    axes[1, 0].set_xlabel('Ejection Fraction (%)')
    axes[1, 0].set_ylabel('Frequency')
    
    # Serum Creatinine distribution
    axes[1, 1].hist(df['serum_creatinine'], bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 1].set_title('Serum Creatinine Distribution')
    axes[1, 1].set_xlabel('Serum Creatinine (mg/dL)')
    axes[1, 1].set_ylabel('Frequency')
    
    # Follow-up time
    axes[1, 2].hist(df['time'], bins=20, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 2].set_title('Follow-up Time Distribution')
    axes[1, 2].set_xlabel('Time (days)')
    axes[1, 2].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('plots/eda_main_grid.png')
    plt.close()
    
    # 2. Correlation Analysis
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.select_dtypes(include=[np.number]).corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix of Clinical Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png')
    plt.close()
    
    # 3. Feature distributions by survival status
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Key Features Distribution by Survival Status', fontsize=14, fontweight='bold')
    
    # Ejection Fraction
    for outcome in [0, 1]:
        subset = df[df['DEATH_EVENT'] == outcome]
        label = 'Survived' if outcome == 0 else 'Died'
        axes[0, 0].hist(subset['ejection_fraction'], alpha=0.7, label=label, bins=15)
    axes[0, 0].set_title('Ejection Fraction by Survival')
    axes[0, 0].set_xlabel('Ejection Fraction (%)')
    axes[0, 0].legend()
    
    # Serum Creatinine
    for outcome in [0, 1]:
        subset = df[df['DEATH_EVENT'] == outcome]
        label = 'Survived' if outcome == 0 else 'Died'
        axes[0, 1].hist(subset['serum_creatinine'], alpha=0.7, label=label, bins=15)
    axes[0, 1].set_title('Serum Creatinine by Survival')
    axes[0, 1].set_xlabel('Serum Creatinine (mg/dL)')
    axes[0, 1].legend()
    
    # Age
    for outcome in [0, 1]:
        subset = df[df['DEATH_EVENT'] == outcome]
        label = 'Survived' if outcome == 0 else 'Died'
        axes[1, 0].hist(subset['age'], alpha=0.7, label=label, bins=15)
    axes[1, 0].set_title('Age by Survival')
    axes[1, 0].set_xlabel('Age (years)')
    axes[1, 0].legend()
    
    # Time
    for outcome in [0, 1]:
        subset = df[df['DEATH_EVENT'] == outcome]
        label = 'Survived' if outcome == 0 else 'Died'
        axes[1, 1].hist(subset['time'], alpha=0.7, label=label, bins=15)
    axes[1, 1].set_title('Follow-up Time by Survival')
    axes[1, 1].set_xlabel('Time (days)')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('plots/features_by_survival.png')
    plt.close()
    
    # 4. Categorical features analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Categorical Features Analysis', fontsize=14, fontweight='bold')
    
    categorical_features = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
    
    for i, feature in enumerate(categorical_features):
        row = i // 3
        col = i % 3
        
        # Create cross-tabulation
        ct = pd.crosstab(df[feature], df['DEATH_EVENT'])
        ct.plot(kind='bar', ax=axes[row, col], color=['lightgreen', 'lightcoral'])
        axes[row, col].set_title(f'{feature.replace("_", " ").title()} vs Survival')
        axes[row, col].set_xlabel(feature.replace("_", " ").title())
        axes[row, col].set_ylabel('Count')
        axes[row, col].legend(['Survived', 'Died'])
        axes[row, col].tick_params(axis='x', rotation=0)
    
    # Remove the empty subplot
    fig.delaxes(axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('plots/categorical_features.png')
    plt.close()

# ================================
# 7. STATISTICAL ANALYSIS
# ================================
def perform_statistical_tests():
    """Perform statistical tests to identify significant features"""
    print("Statistical Tests for Feature Significance:")
    print("=" * 50)
    # Numerical features - t-test
    numerical_features = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 
                         'platelets', 'serum_creatinine', 'serum_sodium', 'time']
    print("\nT-tests for Numerical Features:")
    for feature in numerical_features:
        if feature in df.columns:
            survived = df[df['DEATH_EVENT'] == 0][feature]
            died = df[df['DEATH_EVENT'] == 1][feature]
            t_stat, p_value = stats.ttest_ind(survived, died)
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"{feature}: t-statistic = {t_stat:.3f}, p-value = {p_value:.6f} {significance}")
    # Categorical features - Chi-square test
    print("\nChi-square Tests for Categorical Features:")
    categorical_features = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
    for feature in categorical_features:
        if feature in df.columns:
            contingency_table = pd.crosstab(df[feature], df['DEATH_EVENT'])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"{feature}: Chi-square = {chi2:.3f}, p-value = {p_value:.6f} {significance}")
    print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05")

# ================================
# 8. ADVANCED ANALYSIS
# ================================
def advanced_analysis():
    """Perform advanced analysis including PCA and clustering"""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numerical_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    print("Principal Component Analysis:")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_[:5]}")
    print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)[:5]}")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
             np.cumsum(pca.explained_variance_ratio_), 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA - Cumulative Explained Variance')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    colors = ['red' if x == 1 else 'blue' for x in df['DEATH_EVENT']]
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA - First Two Components')
    plt.legend(['Survived', 'Died'])
    plt.tight_layout()
    plt.savefig('plots/pca_analysis.png')
    plt.close()
    feature_importance = pd.DataFrame({
        'Feature': numerical_cols,
        'PC1': abs(pca.components_[0]),
        'PC2': abs(pca.components_[1])
    }).sort_values('PC1', ascending=False)
    print("\nFeature Importance in First Principal Component:")
    print(feature_importance)

# ================================
# 9. SURVIVAL ANALYSIS
# ================================
def survival_analysis():
    """Perform survival analysis"""
    # Kaplan-Meier like analysis using follow-up time
    print("Follow-up Time Analysis:")
    # Time statistics by outcome
    time_stats = df.groupby('DEATH_EVENT')['time'].agg(['mean', 'median', 'std', 'min', 'max'])
    print(time_stats)
    # Survival by age groups
    df['age_group'] = pd.cut(df['age'], bins=[0, 60, 70, 80, 100], 
                            labels=['<60', '60-70', '70-80', '80+'])
    survival_by_age = df.groupby('age_group')['DEATH_EVENT'].agg(['count', 'sum', 'mean'])
    survival_by_age.columns = ['Total', 'Deaths', 'Mortality_Rate']
    survival_by_age['Survival_Rate'] = 1 - survival_by_age['Mortality_Rate']
    print("\nSurvival Analysis by Age Groups:")
    print(survival_by_age)
    # Plot survival by age groups
    plt.figure(figsize=(10, 6))
    age_groups = survival_by_age.index
    survival_rates = survival_by_age['Survival_Rate'] * 100
    plt.bar(age_groups, survival_rates, color='skyblue', edgecolor='black')
    plt.title('Survival Rate by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Survival Rate (%)')
    plt.ylim(0, 100)
    for i, v in enumerate(survival_rates):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('plots/survival_by_age_group.png')
    plt.close()

# ================================
# 10. KEY INSIGHTS AND RECOMMENDATIONS
# ================================
def generate_insights():
    """Generate key insights from the analysis"""
    print("\n10. KEY INSIGHTS AND RECOMMENDATIONS")
    print("-" * 50)
    # Calculate key statistics
    mortality_rate = df['DEATH_EVENT'].mean() * 100
    avg_age = df['age'].mean()
    avg_ejection_fraction = df['ejection_fraction'].mean()
    avg_follow_up = df['time'].mean()
    print("KEY FINDINGS:")
    print(f"• Overall mortality rate: {mortality_rate:.1f}%")
    print(f"• Average patient age: {avg_age:.1f} years")
    print(f"• Average ejection fraction: {avg_ejection_fraction:.1f}%")
    print(f"• Average follow-up period: {avg_follow_up:.0f} days")
    # Most important features (based on correlation with death event)
    numeric_df = df.select_dtypes(include=[np.number])  # Only numeric columns
    correlations = numeric_df.corr()['DEATH_EVENT'].to_dict()
    print(f"\nMOST IMPORTANT FEATURES (by correlation with mortality):")
    for i, (feature, corr) in enumerate(list(correlations.items())[1:6]):  # Skip DEATH_EVENT itself
        print(f"{i+1}. {feature}: {corr:.3f}")
    print("\nCLINICAL RECOMMENDATIONS:")
    print("• Focus on monitoring ejection fraction and serum creatinine levels")
    print("• Pay special attention to older patients (>70 years)")
    print("• Consider early intervention for patients with low ejection fraction")
    print("• Regular follow-up is crucial for heart failure patients")
    print("• Implement risk stratification based on key clinical indicators")

# ================================
# ADDITIONAL UTILITY FUNCTIONS
# ================================

def export_summary_report():
    """Export a summary report of the analysis"""
    numeric_df = df.select_dtypes(include=[np.number])  # Only numeric columns
    summary = {
        'Dataset Info': {
            'Total Patients': len(df),
            'Total Features': len(df.columns),
            'Mortality Rate': f"{df['DEATH_EVENT'].mean()*100:.1f}%",
            'Average Age': f"{df['age'].mean():.1f} years",
            'Follow-up Period': f"{df['time'].mean():.0f} days (avg)"
        },
        'Key Statistics': df.describe().to_dict(),
        'Missing Values': df.isnull().sum().to_dict(),
        'Correlation with Mortality': numeric_df.corr()['DEATH_EVENT'].to_dict()
    }
    return summary

# ================================
# MAIN - Redirect ALL PRINT to report.txt
# ================================
if __name__ == "__main__":
    with open("report.txt", "w") as f, redirect_stdout(f):

        print("=" * 60)
        print("HEART FAILURE CLINICAL RECORDS - COMPREHENSIVE EDA")
        print("Author: Cholpon Zhakshylykova")
        print("=" * 60)

        # ================================
        # 1. DATASET OVERVIEW
        # ================================
        print("\n1. DATASET OVERVIEW")
        print("-" * 30)
        print(f"Dataset Shape: {df.shape}")
        print(f"Number of Patients: {df.shape[0]}")
        print(f"Number of Features: {df.shape[1]}")
        print("\nColumn Information:")
        print(df.info())
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nColumn Names:")
        print(df.columns.tolist())

        # ================================
        # 2. FEATURE DESCRIPTIONS
        # ================================
        print("\n2. FEATURE DESCRIPTIONS")
        print("-" * 30)

        feature_descriptions = {
            'age': 'Age of the patient (years)',
            'anaemia': 'Decrease of red blood cells or hemoglobin (boolean)',
            'creatinine_phosphokinase': 'Level of the CPK enzyme in the blood (mcg/L)',
            'diabetes': 'If the patient has diabetes (boolean)',
            'ejection_fraction': 'Percentage of blood leaving the heart at each contraction (%)',
            'high_blood_pressure': 'If the patient has hypertension (boolean)',
            'platelets': 'Platelets in the blood (kiloplatelets/mL)',
            'serum_creatinine': 'Level of serum creatinine in the blood (mg/dL)',
            'serum_sodium': 'Level of serum sodium in the blood (mEq/L)',
            'sex': 'Woman or man (binary)',
            'smoking': 'If the patient smokes or not (boolean)',
            'time': 'Follow-up period (days)',
            'DEATH_EVENT': 'If the patient deceased during the follow-up period (boolean) - TARGET'
        }

        for feature, description in feature_descriptions.items():
            if feature in df.columns:
                print(f"• {feature}: {description}")

        # ================================
        # 3. DATA QUALITY ASSESSMENT
        # ================================
        print("\n3. DATA QUALITY ASSESSMENT")
        print("-" * 30)
        print("Missing Values:")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found!")
        print("\nDuplicate Rows:")
        duplicates = df.duplicated().sum()
        print(f"Number of duplicate rows: {duplicates}")
        print("\nData Types:")
        print(df.dtypes)

        # ================================
        # 4. DESCRIPTIVE STATISTICS
        # ================================
        print("\n4. DESCRIPTIVE STATISTICS")
        print("-" * 30)
        print("Numerical Features Summary:")
        print(df.describe())
        print("\nCategorical Features Summary:")
        categorical_cols = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'DEATH_EVENT']
        for col in categorical_cols:
            if col in df.columns:
                print(f"\n{col}:")
                print(df[col].value_counts())
                percentages = (df[col].value_counts(normalize=True) * 100).round(2)
                print(f"Percentage distribution (%):\n{percentages}")

        # ================================
        # 5. TARGET VARIABLE ANALYSIS
        # ================================
        print("\n5. TARGET VARIABLE ANALYSIS")
        print("-" * 30)
        if 'DEATH_EVENT' in df.columns:
            target_dist = df['DEATH_EVENT'].value_counts()
            print(f"Target Distribution:")
            print(f"Survived (0): {target_dist[0]} patients ({target_dist[0]/len(df)*100:.1f}%)")
            print(f"Died (1): {target_dist[1]} patients ({target_dist[1]/len(df)*100:.1f}%)")
            print(f"\nSurvival Rate: {(1 - df['DEATH_EVENT'].mean())*100:.1f}%")
            print(f"Mortality Rate: {df['DEATH_EVENT'].mean()*100:.1f}%")

        # ================================
        # RUN ALL ANALYSES
        # ================================
        print("\n" + "="*60)
        print("RUNNING COMPREHENSIVE ANALYSIS...")
        print("="*60)
        try:
            create_comprehensive_plots()
            perform_statistical_tests()
            advanced_analysis()
            survival_analysis()
            generate_insights()
            print("\n" + "="*60)
            print("EDA COMPLETED SUCCESSFULLY!")
            print("="*60)
        except Exception as e:
            print(f"Error in analysis: {e}")
            print("Please ensure the dataset is loaded correctly and all required libraries are installed.")

        # ================================
        # ADDITIONAL REPORT
        # ================================
        print("\nFINAL SUMMARY:")
        summary = export_summary_report()
        for key, value in summary['Dataset Info'].items():
            print(f"• {key}: {value}")
        print("\nKey Statistics:")
        for stat_key, stat_val in summary['Key Statistics'].items():
            print(f"{stat_key}: {stat_val}")
        print("\nMissing Values:")
        for mkey, mval in summary['Missing Values'].items():
            print(f"{mkey}: {mval}")
        print("\nCorrelation with Mortality:")
        for ckey, cval in summary['Correlation with Mortality'].items():
            print(f"{ckey}: {cval:.3f}")
        print("\n" + "="*60)
        print("END OF ANALYSIS")
        print("Author: Cholpon Zhakshylykova")
        print("Dataset: Heart Failure Clinical Records (UCI)")
        print("="*60)
