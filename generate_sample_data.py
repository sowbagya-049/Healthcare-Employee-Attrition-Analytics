#!/usr/bin/env python3
"""
Healthcare Employee Attrition Dataset Generator
Generates synthetic healthcare employee data with realistic attrition patterns
"""

import pandas as pd
import numpy as np
import os
import sys

# Set random seed for reproducibility
np.random.seed(42)

def generate_healthcare_attrition_data(n_samples=4410):
    """
    Generate synthetic healthcare employee attrition dataset
    
    Parameters:
    -----------
    n_samples : int
        Number of employee records to generate (default: 4410)
    
    Returns:
    --------
    DataFrame : Synthetic employee data with 19 features
    """
    
    print(f"Generating {n_samples:,} employee records...")
    print("-" * 60)
    
    # ========================
    # 1. DEMOGRAPHICS
    # ========================
    print("✓ Generating demographics...")
    age = np.random.randint(22, 66, n_samples)
    gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.45, 0.55])
    education = np.random.choice([1, 2, 3, 4, 5], n_samples, 
                                 p=[0.10, 0.25, 0.35, 0.20, 0.10])
    
    # ========================
    # 2. WORK-RELATED DATA
    # ========================
    print("✓ Generating work-related data...")
    
    # Departments
    departments = ['Nursing', 'Admin', 'Doctors', 'Technical', 'Support']
    department = np.random.choice(departments, n_samples, 
                                  p=[0.35, 0.20, 0.15, 0.18, 0.12])
    
    # Job roles by department
    job_roles_dict = {
        'Nursing': ['Nurse', 'Head Nurse', 'Nurse Practitioner', 'Clinical Nurse'],
        'Admin': ['Admin Staff', 'HR Specialist', 'Receptionist', 'Office Manager'],
        'Doctors': ['Doctor', 'Surgeon', 'Specialist', 'Resident'],
        'Technical': ['Lab Technician', 'Radiology Tech', 'Pharmacy Tech', 'Medical Tech'],
        'Support': ['Janitor', 'Security', 'Food Service', 'Maintenance']
    }
    
    job_role = []
    for dept in department:
        job_role.append(np.random.choice(job_roles_dict[dept]))
    
    # Shifts
    shifts = np.random.choice(['Day', 'Night', 'Rotating'], n_samples, 
                             p=[0.45, 0.30, 0.25])
    
    # Tenure
    years_at_company = np.random.exponential(5, n_samples).clip(0, 40)
    years_in_current_role = (years_at_company * np.random.uniform(0.3, 1.0, n_samples)).clip(0, 40)
    
    # ========================
    # 3. SATISFACTION METRICS
    # ========================
    print("✓ Generating satisfaction metrics...")
    
    job_satisfaction = np.random.choice([1, 2, 3, 4], n_samples, 
                                       p=[0.20, 0.28, 0.32, 0.20])
    environment_satisfaction = np.random.choice([1, 2, 3, 4], n_samples, 
                                               p=[0.18, 0.30, 0.32, 0.20])
    work_life_balance = np.random.choice([1, 2, 3, 4], n_samples, 
                                        p=[0.22, 0.28, 0.30, 0.20])
    relationship_satisfaction = np.random.choice([1, 2, 3, 4], n_samples, 
                                                p=[0.15, 0.25, 0.35, 0.25])
    
    # ========================
    # 4. COMPENSATION
    # ========================
    print("✓ Generating compensation data...")
    
    # Base income by job role
    base_income_map = {
        'Nurse': 4500, 'Head Nurse': 6500, 'Nurse Practitioner': 7500, 'Clinical Nurse': 5000,
        'Admin Staff': 3500, 'HR Specialist': 5000, 'Receptionist': 3000, 'Office Manager': 5500,
        'Doctor': 12000, 'Surgeon': 18000, 'Specialist': 15000, 'Resident': 6000,
        'Lab Technician': 4000, 'Radiology Tech': 5500, 'Pharmacy Tech': 4500, 'Medical Tech': 4800,
        'Janitor': 2500, 'Security': 3200, 'Food Service': 2800, 'Maintenance': 3000
    }
    
    monthly_income = np.array([
        base_income_map.get(role, 4000) * np.random.uniform(0.8, 1.3) 
        for role in job_role
    ])
    
    # Adjust income for tenure
    tenure_bonus = 1 + (years_at_company * 0.02)
    monthly_income = monthly_income * tenure_bonus.clip(1, 1.6)
    
    hourly_rate = monthly_income / 160  # 160 hours/month
    
    overtime = np.random.choice(['Yes', 'No'], n_samples, p=[0.35, 0.65])
    
    # ========================
    # 5. OTHER FACTORS
    # ========================
    print("✓ Generating additional factors...")
    
    distance_from_home = np.random.exponential(10, n_samples).clip(1, 50)
    performance_rating = np.random.choice([1, 2, 3, 4], n_samples, 
                                         p=[0.05, 0.15, 0.50, 0.30])
    training_times_last_year = np.random.poisson(2, n_samples).clip(0, 6)
    
    # ========================
    # 6. CALCULATE ATTRITION
    # ========================
    print("✓ Calculating attrition probabilities...")
    
    # Base attrition probability (lower base for more realistic rates)
    attrition_prob = np.full(n_samples, 0.08)
    
    # Factor 1: Satisfaction (STRONG predictor - lower = much higher attrition)
    satisfaction_factor = (5 - job_satisfaction) * 0.08  # Increased from 0.05
    attrition_prob += satisfaction_factor
    
    work_life_factor = (5 - work_life_balance) * 0.06  # Increased from 0.04
    attrition_prob += work_life_factor
    
    environment_factor = (5 - environment_satisfaction) * 0.05
    attrition_prob += environment_factor
    
    # Factor 2: Income (STRONG predictor - lower = much higher attrition)
    income_percentile = pd.Series(monthly_income).rank(pct=True).values
    income_factor = (1 - income_percentile) * 0.20  # Increased from 0.15
    attrition_prob += income_factor
    
    # Factor 3: Overtime (STRONG predictor)
    overtime_factor = np.where(overtime == 'Yes', 0.12, 0)  # Increased from 0.08
    attrition_prob += overtime_factor
    
    # Factor 4: Shift type (night/rotating = higher attrition)
    shift_adjustment = {'Day': 0, 'Night': 0.10, 'Rotating': 0.07}  # Increased
    shift_factor = np.array([shift_adjustment[s] for s in shifts])
    attrition_prob += shift_factor
    
    # Factor 5: Tenure (CRITICAL - new employees much more likely to leave)
    tenure_factor = np.where(years_at_company < 1, 0.20, 0)  # Very high for <1 year
    tenure_factor += np.where((years_at_company >= 1) & (years_at_company < 2), 0.15, 0)
    tenure_factor += np.where((years_at_company >= 2) & (years_at_company < 3), 0.08, 0)
    tenure_factor += np.where(years_at_company > 10, -0.12, 0)  # Increased loyalty bonus
    attrition_prob += tenure_factor
    
    # Factor 6: Distance from home (longer = higher attrition)
    distance_factor = np.where(distance_from_home > 20, 0.06, 0)
    distance_factor += np.where(distance_from_home > 30, 0.04, 0)  # Additional penalty
    attrition_prob += distance_factor
    
    # Factor 7: Age (younger = higher attrition, older = more stable)
    age_factor = np.where(age < 25, 0.10, 0)  # Very young employees
    age_factor += np.where((age >= 25) & (age < 30), 0.06, 0)
    age_factor += np.where(age > 50, -0.08, 0)  # Experienced employees stay
    attrition_prob += age_factor
    
    # Factor 8: Performance (low performance = higher attrition)
    performance_factor = (5 - performance_rating) * 0.04
    attrition_prob += performance_factor
    
    # Factor 9: Training (less training = higher attrition)
    training_factor = np.where(training_times_last_year == 0, 0.08, 0)
    training_factor += np.where(training_times_last_year >= 4, -0.05, 0)
    attrition_prob += training_factor
    
    # Factor 10: Combined risk factors (amplify when multiple factors present)
    # High-risk combination: Low satisfaction + Low income + Overtime
    high_risk_combo = (
        (job_satisfaction <= 2) & 
        (income_percentile < 0.3) & 
        (overtime == 'Yes')
    )
    attrition_prob += np.where(high_risk_combo, 0.15, 0)
    
    # Very low risk combination: High satisfaction + High income + No overtime + Long tenure
    low_risk_combo = (
        (job_satisfaction >= 3) & 
        (income_percentile > 0.7) & 
        (overtime == 'No') & 
        (years_at_company > 5)
    )
    attrition_prob += np.where(low_risk_combo, -0.15, 0)
    
    # Clip probabilities to realistic range (5% to 65%)
    attrition_prob = np.clip(attrition_prob, 0.05, 0.65)
    
    # Generate attrition using probabilities
    attrition = np.random.binomial(1, attrition_prob)
    attrition = np.where(attrition == 1, 'Yes', 'No')
    
    # ========================
    # 7. CREATE DATAFRAME
    # ========================
    print("✓ Creating dataset...")
    
    df = pd.DataFrame({
        'Age': age.astype(int),
        'Gender': gender,
        'Education': education,
        'Department': department,
        'JobRole': job_role,
        'Shift': shifts,
        'YearsAtCompany': np.round(years_at_company, 1),
        'YearsInCurrentRole': np.round(years_in_current_role, 1),
        'JobSatisfaction': job_satisfaction,
        'EnvironmentSatisfaction': environment_satisfaction,
        'WorkLifeBalance': work_life_balance,
        'RelationshipSatisfaction': relationship_satisfaction,
        'MonthlyIncome': np.round(monthly_income, 2),
        'HourlyRate': np.round(hourly_rate, 2),
        'OverTime': overtime,
        'DistanceFromHome': np.round(distance_from_home, 1),
        'PerformanceRating': performance_rating,
        'TrainingTimesLastYear': training_times_last_year,
        'Attrition': attrition
    })
    
    return df


def print_dataset_summary(df):
    """Print summary statistics of the generated dataset"""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    print(f"\n📈 Attrition Distribution:")
    attrition_counts = df['Attrition'].value_counts()
    for status, count in attrition_counts.items():
        pct = (count / len(df)) * 100
        print(f"   {status:3s}: {count:5,} ({pct:5.2f}%)")
    
    print(f"\n🏢 Department Distribution:")
    dept_counts = df['Department'].value_counts()
    for dept, count in dept_counts.items():
        pct = (count / len(df)) * 100
        print(f"   {dept:15s}: {count:5,} ({pct:5.2f}%)")
    
    print(f"\n💰 Income Statistics:")
    print(f"   Mean:   ${df['MonthlyIncome'].mean():8,.2f}")
    print(f"   Median: ${df['MonthlyIncome'].median():8,.2f}")
    print(f"   Min:    ${df['MonthlyIncome'].min():8,.2f}")
    print(f"   Max:    ${df['MonthlyIncome'].max():8,.2f}")
    
    print(f"\n⏱️  Tenure Statistics:")
    print(f"   Mean:   {df['YearsAtCompany'].mean():5.1f} years")
    print(f"   Median: {df['YearsAtCompany'].median():5.1f} years")
    print(f"   Min:    {df['YearsAtCompany'].min():5.1f} years")
    print(f"   Max:    {df['YearsAtCompany'].max():5.1f} years")
    
    print(f"\n📋 First 5 Rows:")
    print(df.head().to_string())
    
    print("\n" + "=" * 60)


def main():
    """Main execution function"""
    
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "Healthcare Attrition Data Generator" + " " * 13 + "║")
    print("╚" + "═" * 58 + "╝")
    print("\n")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Check for custom size argument
    n_samples = 4410
    if len(sys.argv) > 1:
        try:
            n_samples = int(sys.argv[1])
            print(f"ℹ️  Custom size requested: {n_samples:,} records\n")
        except ValueError:
            print(f"⚠️  Invalid size argument. Using default: {n_samples:,}\n")
    
    # Generate dataset
    try:
        df = generate_healthcare_attrition_data(n_samples)
        
        # Save to CSV
        output_path = 'data/healthcare_attrition.csv'
        df.to_csv(output_path, index=False)
        
        print(f"✓ Dataset saved to: {output_path}")
        
        # Print summary
        print_dataset_summary(df)
        
        print("\n✅ SUCCESS! Dataset generated and saved.")
        print("\n📌 Next Steps:")
        print("   1. Run analysis: python main.py")
        print("   2. Or launch dashboard: streamlit run app.py")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("Please check the error message and try again.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()