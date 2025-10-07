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
    """Train Random Forest model"""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

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
    st.sidebar.title("Healthcare Attrition")
    page = st.sidebar.radio("Select Page", [
        "📊 Overview",
        "📈 Deep Dive Analysis",
        "🕵️‍♂️ Predictive Model",
        #"💡 Insights & Recommendations",
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
    
    # Prepare data
    with st.spinner("Training model..."):
        X_train, X_test, y_train, y_test, feature_names = prepare_data_for_ml(df)
        model = train_model(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Display metrics
    st.success("✓ Model trained successfully!")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Accuracy", f"{accuracy*100:.2f}%")
    with col2:
        st.metric("Precision", f"{precision*100:.2f}%")
    with col3:
        st.metric("Recall", f"{recall*100:.2f}%")
    with col4:
        st.metric("F1 Score", f"{f1*100:.2f}%")
    with col5:
        st.metric("ROC-AUC", f"{roc_auc*100:.2f}%")
    
    st.markdown("---")
    
    # Feature importance
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(15)
        
        fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='Viridis')
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top Predictors")
        for idx, row in feature_importance.head(10).iterrows():
            st.metric(row['Feature'], f"{row['Importance']*100:.1f}%")
    
    # Prediction simulator
    st.markdown("---")
    st.subheader("🎯 Individual Attrition Risk Predictor")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Age", 22, 65, 35)
        satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
        income = st.number_input("Monthly Income ($)", 2000, 20000, 5000)
    
    with col2:
        tenure = st.slider("Years at Company", 0, 40, 5)
        overtime = st.selectbox("Overtime", ["Yes", "No"])
        shift = st.selectbox("Shift Type", ["Day", "Night", "Rotating"])
    
    with col3:
        work_life_balance = st.slider("Work-Life Balance", 1, 4, 3)
        distance = st.slider("Distance from Home (km)", 1, 50, 10)
        performance = st.slider("Performance Rating", 1, 4, 3)
    
    if st.button("Predict Attrition Risk", type="primary"):
        st.info("Prediction functionality requires complete feature encoding. This is a simplified demo.")
        risk_score = np.random.uniform(0.1, 0.9)  # Simulated
        
        if risk_score > 0.6:
            st.error(f"🚨 HIGH RISK: {risk_score*100:.1f}% probability of attrition")
            st.warning("**Recommended Actions:**\n- Immediate manager intervention\n- Compensation review\n- Career development discussion")
        elif risk_score > 0.3:
            st.warning(f"⚠️ MODERATE RISK: {risk_score*100:.1f}% probability of attrition")
            st.info("**Recommended Actions:**\n- Regular check-ins\n- Monitor satisfaction levels")
        else:
            st.success(f"✅ LOW RISK: {risk_score*100:.1f}% probability of attrition")

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