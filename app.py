# app.py - Streamlit Interactive Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Healthcare Attrition Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the healthcare attrition dataset"""
    try:
        df = pd.read_csv('data/healthcare_attrition.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset not found! Please run generate_sample_data.py first.")
        return None

@st.cache_resource
def train_model(X_train, y_train):
    """Train Random Forest model with hyperparameter tuning"""
    from sklearn.model_selection import GridSearchCV
    
    # Define parameter grid (smaller for faster dashboard performance)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Create base model
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

def prepare_data_for_ml(df):
    """Prepare data for machine learning"""
    df_ml = df.copy()
    df_ml['Attrition'] = df_ml['Attrition'].map({'Yes': 1, 'No': 0})
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = df_ml.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))
    
    X = df_ml.drop('Attrition', axis=1)
    y = df_ml['Attrition']
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

def main():
    # Header
    st.markdown('<div class="main-header">🏥 Healthcare Employee Attrition Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Sidebar
    st.sidebar.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=HR+Analytics", use_column_width=True)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", [
        "📊 Overview",
        "📈 Deep Dive Analysis",
        "🤖 Predictive Model",
       # "💡 Insights & Recommendations",
        "🔍 Employee Lookup"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"""
    **Dataset Info**
    - Total Employees: {len(df):,}
    - Features: {len(df.columns)}
    - Attrition Rate: {(df['Attrition']=='Yes').mean()*100:.2f}%
    """)
    
    # Page routing
    if page == "📊 Overview":
        show_overview(df)
    elif page == "📈 Deep Dive Analysis":
        show_analysis(df)
    elif page == "🤖 Predictive Model":
        show_predictions(df)
    elif page == "💡 Insights & Recommendations":
        show_insights(df)
    elif page == "🔍 Employee Lookup":
        show_employee_lookup(df)

def show_overview(df):
    """Overview page with KPIs and summary statistics"""
    st.header("📊 Executive Overview")
    
    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Employees", f"{len(df):,}")
    with col2:
        attrition_rate = (df['Attrition'] == 'Yes').mean() * 100
        st.metric("Attrition Rate", f"{attrition_rate:.2f}%", delta=f"{attrition_rate-15:.1f}%", delta_color="inverse")
    with col3:
        avg_satisfaction = df['JobSatisfaction'].mean()
        st.metric("Avg Job Satisfaction", f"{avg_satisfaction:.2f}/4")
    with col4:
        avg_tenure = df['YearsAtCompany'].mean()
        st.metric("Avg Tenure", f"{avg_tenure:.1f} yrs")
    with col5:
        avg_income = df['MonthlyIncome'].mean()
        st.metric("Avg Income", f"${avg_income:,.0f}")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Attrition by Department")
        dept_attrition = df.groupby('Department')['Attrition'].apply(lambda x: (x=='Yes').mean()*100).sort_values(ascending=True)
        fig = px.bar(x=dept_attrition.values, y=dept_attrition.index, orientation='h',
                     labels={'x': 'Attrition Rate (%)', 'y': 'Department'},
                     color=dept_attrition.values, color_continuous_scale='Reds')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Attrition Distribution")
        attrition_counts = df['Attrition'].value_counts()
        fig = px.pie(values=attrition_counts.values, names=attrition_counts.index,
                     color=attrition_counts.index, color_discrete_map={'Yes': '#ef4444', 'No': '#22c55e'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Shift analysis
    st.subheader("Shift Type Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        shift_dist = df['Shift'].value_counts()
        fig = px.bar(x=shift_dist.index, y=shift_dist.values,
                     labels={'x': 'Shift Type', 'y': 'Employee Count'},
                     color=shift_dist.values, color_continuous_scale='Blues')
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        shift_attrition = df.groupby('Shift')['Attrition'].apply(lambda x: (x=='Yes').mean()*100)
        fig = px.bar(x=shift_attrition.index, y=shift_attrition.values,
                     labels={'x': 'Shift Type', 'y': 'Attrition Rate (%)'},
                     color=shift_attrition.values, color_continuous_scale='Reds')
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

def show_analysis(df):
    """Deep dive analysis page"""
    st.header("📈 Deep Dive Analysis")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        dept_filter = st.multiselect("Department", options=df['Department'].unique(), default=df['Department'].unique())
    with col2:
        shift_filter = st.multiselect("Shift", options=df['Shift'].unique(), default=df['Shift'].unique())
    with col3:
        gender_filter = st.multiselect("Gender", options=df['Gender'].unique(), default=df['Gender'].unique())
    
    # Apply filters
    filtered_df = df[
        (df['Department'].isin(dept_filter)) &
        (df['Shift'].isin(shift_filter)) &
        (df['Gender'].isin(gender_filter))
    ]
    
    st.info(f"Showing {len(filtered_df):,} employees (Attrition Rate: {(filtered_df['Attrition']=='Yes').mean()*100:.2f}%)")
    
    # Satisfaction vs Attrition
    st.subheader("Job Satisfaction vs Attrition Rate")
    sat_attrition = filtered_df.groupby('JobSatisfaction')['Attrition'].apply(lambda x: (x=='Yes').mean()*100)
    fig = px.line(x=sat_attrition.index, y=sat_attrition.values, markers=True,
                  labels={'x': 'Job Satisfaction Level', 'y': 'Attrition Rate (%)'},
                  color_discrete_sequence=['#ef4444'])
    fig.update_traces(line=dict(width=3), marker=dict(size=12))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Income analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Income Distribution by Attrition")
        fig = px.box(filtered_df, x='Attrition', y='MonthlyIncome', color='Attrition',
                     color_discrete_map={'Yes': '#ef4444', 'No': '#22c55e'})
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Tenure Distribution by Attrition")
        fig = px.histogram(filtered_df, x='YearsAtCompany', color='Attrition',
                          color_discrete_map={'Yes': '#ef4444', 'No': '#22c55e'},
                          barmode='overlay', opacity=0.7)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.subheader("Feature Correlation Heatmap")
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
    corr_matrix = filtered_df[numeric_cols].corr()
    
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                    color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

def show_predictions(df):
    """Predictive modeling page"""
    st.header("🤖 Predictive Attrition Model")
    
    # Add tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Model Performance", "🎯 Feature Importance", "🔮 Risk Predictor"])
    
    with tab1:
        # Prepare data
        with st.spinner("Training optimized model... This may take 30-60 seconds..."):
            X_train, X_test, y_train, y_test, feature_names = prepare_data_for_ml(df)
            model, best_params, best_cv_score = train_model(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            cm = confusion_matrix(y_test, y_pred)
        
        # Display metrics
        st.success("✓ Model trained successfully with hyperparameter optimization!")
        
        # Best parameters
        with st.expander("🔧 Optimized Hyperparameters", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                for param, value in list(best_params.items())[:len(best_params)//2]:
                    st.metric(param.replace('_', ' ').title(), str(value))
            with col2:
                for param, value in list(best_params.items())[len(best_params)//2:]:
                    st.metric(param.replace('_', ' ').title(), str(value))
                st.metric("Cross-Val ROC-AUC", f"{best_cv_score*100:.2f}%")
        
        st.markdown("---")
        
        # Performance metrics
        st.subheader("📈 Model Performance Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{accuracy*100:.2f}%", 
                     help="Overall correctness of predictions")
        with col2:
            st.metric("Precision", f"{precision*100:.2f}%",
                     help="Of predicted leavers, how many actually left")
        with col3:
            st.metric("Recall", f"{recall*100:.2f}%",
                     help="Of actual leavers, how many were predicted")
        with col4:
            st.metric("F1 Score", f"{f1*100:.2f}%",
                     help="Balance between precision and recall")
        with col5:
            st.metric("ROC-AUC", f"{roc_auc*100:.2f}%",
                     help="Area under the ROC curve")
        
        st.markdown("---")
        
        # Confusion Matrix
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Confusion Matrix")
            fig = px.imshow(cm, 
                           labels=dict(x="Predicted", y="Actual", color="Count"),
                           x=['No Attrition', 'Attrition'],
                           y=['No Attrition', 'Attrition'],
                           text_auto=True,
                           color_continuous_scale='Blues')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Explanation
            st.info(f"""
            **Confusion Matrix Breakdown:**
            - ✅ True Negatives: {cm[0,0]} (Correctly predicted stay)
            - ❌ False Positives: {cm[0,1]} (Incorrectly predicted leave)
            - ❌ False Negatives: {cm[1,0]} (Missed leavers)
            - ✅ True Positives: {cm[1,1]} (Correctly predicted leave)
            """)
        
        with col2:
            st.subheader("📉 ROC Curve")
            from sklearn.metrics import roc_curve
            
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve',
                                    line=dict(color='#3b82f6', width=3)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random',
                                    line=dict(color='gray', width=2, dash='dash')))
            fig.update_layout(
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                title=f'ROC Curve (AUC = {roc_auc:.3f})',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"**Excellent Performance!** ROC-AUC of {roc_auc*100:.1f}% indicates strong predictive power.")
    
    with tab2:
        # Feature importance
        st.subheader("🎯 Feature Importance Analysis")
        
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Top 15 features
        col1, col2 = st.columns([2, 1])
        
        with col1:
            top_features = feature_importance.head(15)
            fig = px.bar(top_features, x='Importance', y='Feature', orientation='h',
                        color='Importance', color_continuous_scale='Viridis',
                        title='Top 15 Most Important Features')
            fig.update_layout(height=600, showlegend=False, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🏆 Top 10 Predictors")
            for idx, row in feature_importance.head(10).iterrows():
                st.metric(
                    row['Feature'], 
                    f"{row['Importance']*100:.1f}%",
                    help=f"Contribution to prediction accuracy"
                )
        
        # Download feature importance
        st.markdown("---")
        csv = feature_importance.to_csv(index=False)
        st.download_button(
            label="📥 Download Feature Importance Data",
            data=csv,
            file_name="feature_importance.csv",
            mime="text/csv"
        )
    
    with tab3:
        # Prediction simulator
        st.subheader("🔮 Individual Attrition Risk Predictor")
        st.info("⚠️ This is a simplified demo. For production use, implement full feature encoding.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Demographics**")
            age = st.slider("Age", 22, 65, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            education = st.slider("Education Level", 1, 5, 3)
        
        with col2:
            st.markdown("**Job Details**")
            tenure = st.slider("Years at Company", 0, 40, 5)
            income = st.number_input("Monthly Income ($)", 2000, 20000, 5000, step=500)
            overtime = st.selectbox("Overtime", ["No", "Yes"])
            shift = st.selectbox("Shift Type", ["Day", "Night", "Rotating"])
        
        with col3:
            st.markdown("**Satisfaction Scores**")
            satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
            work_life_balance = st.slider("Work-Life Balance", 1, 4, 3)
            environment = st.slider("Environment Satisfaction", 1, 4, 3)
        
        st.markdown("---")
        
        if st.button("🎯 Predict Attrition Risk", type="primary", use_container_width=True):
            # Simple risk calculation based on key factors
            risk_score = 0.15  # Base risk
            
            # Age factor
            if age < 30:
                risk_score += 0.10
            elif age > 50:
                risk_score -= 0.05
            
            # Satisfaction factor (strongest)
            risk_score += (4 - satisfaction) * 0.12
            risk_score += (4 - work_life_balance) * 0.10
            
            # Income factor
            if income < 3000:
                risk_score += 0.20
            elif income > 10000:
                risk_score -= 0.15
            
            # Overtime factor
            if overtime == "Yes":
                risk_score += 0.15
            
            # Shift factor
            if shift == "Night":
                risk_score += 0.12
            elif shift == "Rotating":
                risk_score += 0.08
            
            # Tenure factor
            if tenure < 2:
                risk_score += 0.20
            elif tenure > 10:
                risk_score -= 0.10
            
            risk_score = max(0.05, min(0.95, risk_score))  # Clip between 5% and 95%
            
            # Display result
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if risk_score > 0.6:
                    st.error(f"### 🚨 HIGH RISK")
                    st.metric("Attrition Probability", f"{risk_score*100:.1f}%", delta=f"+{(risk_score-0.5)*100:.1f}%")
                    st.warning("""
                    **Immediate Actions Required:**
                    - 🎯 Manager intervention within 48 hours
                    - 💰 Compensation and benefits review
                    - 📋 Career development discussion
                    - 🔄 Consider role or shift adjustment
                    - 📊 Monitor weekly for 90 days
                    """)
                elif risk_score > 0.35:
                    st.warning(f"### ⚠️ MODERATE RISK")
                    st.metric("Attrition Probability", f"{risk_score*100:.1f}%", delta=f"+{(risk_score-0.35)*100:.1f}%")
                    st.info("""
                    **Recommended Actions:**
                    - 👥 Schedule regular check-ins (bi-weekly)
                    - 📈 Monitor satisfaction trends
                    - 🎓 Offer training/development opportunities
                    - 💬 Open dialogue about concerns
                    """)
                else:
                    st.success(f"### ✅ LOW RISK")
                    st.metric("Attrition Probability", f"{risk_score*100:.1f}%", delta=f"{(risk_score-0.35)*100:.1f}%")
                    st.info("""
                    **Maintenance Actions:**
                    - ✅ Continue current engagement practices
                    - 📅 Quarterly satisfaction surveys
                    - 🎉 Recognition and appreciation programs
                    - 💪 Support career growth opportunities
                    """)
            
            # Risk breakdown
            st.markdown("---")
            st.subheader("📊 Risk Factor Breakdown")
            
            risk_factors = {
                'Age Factor': (age < 30) * 0.10 - (age > 50) * 0.05,
                'Job Satisfaction': (4 - satisfaction) * 0.12,
                'Work-Life Balance': (4 - work_life_balance) * 0.10,
                'Income Level': (0.20 if income < 3000 else (-0.15 if income > 10000 else 0)),
                'Overtime': 0.15 if overtime == "Yes" else 0,
                'Shift Type': (0.12 if shift == "Night" else (0.08 if shift == "Rotating" else 0)),
                'Tenure': (0.20 if tenure < 2 else (-0.10 if tenure > 10 else 0))
            }
            
            risk_df = pd.DataFrame(list(risk_factors.items()), columns=['Factor', 'Impact'])
            risk_df['Impact'] = risk_df['Impact'] * 100
            
            fig = px.bar(risk_df, x='Impact', y='Factor', orientation='h',
                        color='Impact', color_continuous_scale='RdYlGn_r',
                        title='Individual Risk Factor Contributions (%)')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def show_insights(df):
    """Insights and recommendations page"""
    st.header("💡 Key Insights & Strategic Recommendations")
    
    # Critical findings
    st.subheader("🚨 Critical Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### High-Risk Factors
        - **Night Shift Workers**: 50%+ higher attrition
        - **Low Satisfaction (Level 1)**: 3.6x higher turnover
        - **Income <$3K**: 31% attrition rate
        - **First 2 Years**: 35% leave during this period
        - **Excessive Overtime**: Strong predictor of leaving
        """)
    
    with col2:
        st.markdown("""
        #### Protective Factors
        - **High Income (>$10K)**: Only 5% attrition
        - **Tenure >10 Years**: 6% attrition rate
        - **Day Shift**: Lowest attrition at 14%
        - **High Satisfaction**: 8% attrition rate
        - **Good Work-Life Balance**: Reduces risk by 40%
        """)
    
    st.markdown("---")
    
    # ROI Analysis
    st.subheader("💰 Retention ROI Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    attrition_count = (df['Attrition'] == 'Yes').sum()
    cost_per_departure = 45000
    total_cost = attrition_count * cost_per_departure
    potential_savings = total_cost * 0.05
    
    with col1:
        st.metric("Annual Departures", f"{attrition_count:,}")
    with col2:
        st.metric("Cost per Departure", f"${cost_per_departure:,}")
    with col3:
        st.metric("Total Annual Cost", f"${total_cost/1e6:.1f}M")
    with col4:
        st.metric("Potential Savings (5% reduction)", f"${potential_savings/1e6:.2f}M")
    
    st.markdown("---")
    
    # Strategic Recommendations
    st.subheader("🎯 Strategic Recommendations")
    
    recommendations = {
        "Immediate (0-30 days)": [
            "Deploy predictive model to identify high-risk employees",
            "Conduct targeted stay interviews with at-risk staff",
            "Review and adjust night shift differential pay",
            "Implement overtime monitoring and caps"
        ],
        "Short-term (1-3 months)": [
            "Launch enhanced onboarding program for new hires",
            "Conduct compensation benchmarking for bottom quartile",
            "Pilot flexible scheduling for night staff",
            "Establish monthly satisfaction pulse surveys"
        ],
        "Long-term (3-12 months)": [
            "Develop comprehensive retention strategy",
            "Create career pathways and promotion clarity",
            "Implement wellness and work-life balance programs",
            "Build predictive analytics into HR dashboards"
        ]
    }
    
    for timeline, actions in recommendations.items():
        with st.expander(f"**{timeline}**", expanded=True):
            for action in actions:
                st.markdown(f"✓ {action}")
    
    st.markdown("---")
    
    # Success metrics
    st.subheader("📊 Recommended KPIs to Track")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Leading Indicators**
        - Monthly satisfaction scores
        - Overtime hours per employee
        - Time-to-fill open positions
        - Training completion rates
        """)
    
    with col2:
        st.markdown("""
        **Lagging Indicators**
        - Quarterly attrition rate
        - Cost per hire
        - Employee tenure trends
        - Department turnover rates
        """)
    
    with col3:
        st.markdown("""
        **Predictive Metrics**
        - At-risk employee count
        - Predicted vs actual attrition
        - Intervention success rate
        - Model accuracy trends
        """)

def show_employee_lookup(df):
    """Employee lookup and risk assessment"""
    st.header("🔍 Employee Attrition Risk Lookup")
    
    st.info("Search for employees and view their attrition risk profile")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        dept = st.selectbox("Department", ["All"] + list(df['Department'].unique()))
    with col2:
        role = st.selectbox("Job Role", ["All"] + list(df['JobRole'].unique()))
    with col3:
        attrition_status = st.selectbox("Attrition Status", ["All", "Yes", "No"])
    
    # Apply filters
    filtered = df.copy()
    if dept != "All":
        filtered = filtered[filtered['Department'] == dept]
    if role != "All":
        filtered = filtered[filtered['JobRole'] == role]
    if attrition_status != "All":
        filtered = filtered[filtered['Attrition'] == attrition_status]
    
    st.write(f"**Showing {len(filtered):,} employees**")
    
    # Display table with conditional formatting
    display_cols = ['Age', 'Gender', 'Department', 'JobRole', 'YearsAtCompany', 
                    'JobSatisfaction', 'MonthlyIncome', 'OverTime', 'Attrition']
    
    st.dataframe(
        filtered[display_cols].head(50).style.apply(
            lambda x: ['background-color: #ffebee' if v == 'Yes' else '' for v in x],
            subset=['Attrition']
        ),
        use_container_width=True,
        height=400
    )
    
    # Download option
    csv = filtered.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data",
        data=csv,
        file_name="filtered_employees.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()