import os
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ============================================================
# PAGE CONFIGURATION & STYLING
# ============================================================
st.set_page_config(
    page_title="Rice Type Classifier",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #27ae60;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover { 
        background-color: #2ecc71; 
        color: white;
        transform: translateY(-2px); 
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATA LOADING & PREPROCESSING
# ============================================================
@st.cache_data
def load_and_clean_data():
    file_path = "riceClassification.csv"
    if not os.path.exists(file_path):
        return None
        
    df = pd.read_csv(file_path)
    
    # Drop ID column if it exists
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
        
    # Map target to 0 and 1 if it isn't already
    if df['Class'].dtype == 'object':
        df['Class'] = df['Class'].map({'Cammeo': 0, 'Osmancik': 1})
        
    return df

# ============================================================
# MODEL TRAINING ENGINE
# ============================================================
@st.cache_resource
def train_all_models(df):
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Stratified split to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Scale the features (Required for SVM, KNN, Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    results = {}
    fitted_models = {}

    # Train and evaluate all models
    for name, model in models.items():
        # Tree-based models don't strictly need scaling, but it doesn't hurt them. 
        # We use scaled data for all to keep the pipeline unified.
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        
        # Store metrics
        results[name] = {
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "F1 Score": f1_score(y_test, preds)
        }
        fitted_models[name] = model

    return fitted_models, scaler, results, X.columns.tolist()

def get_feature_bounds(df, features):
    bounds = {}
    for col in features:
        bounds[col] = {
            "min": float(df[col].min()), 
            "max": float(df[col].max()), 
            "mean": float(df[col].mean())
        }
    return bounds

# ============================================================
# MAIN APP EXECUTION
# ============================================================
df = load_and_clean_data()

if df is None:
    st.error("⚠️ Dataset not found! Please ensure 'riceClassification.csv' is in the same folder as this script.")
    st.stop()

models_dict, fitted_scaler, evaluation_results, feature_list = train_all_models(df)
bounds = get_feature_bounds(df, feature_list)

# App Header
st.title("🌾 Rice Type Classifier (Cammeo vs. Osmancik)")
st.markdown("Enter grain metrics below to classify the species of rice. Navigate to the **Data & Models** tab to explore the EDA, Heatmaps, and algorithm performance.")

# Create tabs
tab_predict, tab_insights = st.tabs(["🔮 Make a Prediction", "📊 Data & Models"])

# ------------------------------------------------------------
# TAB 1: PREDICTION ENGINE
# ------------------------------------------------------------
with tab_predict:
    st.markdown("### 1. Select a Classification Model")
    selected_model_name = st.selectbox(
        "Choose the Algorithm to use for prediction:",
        list(models_dict.keys()),
        index=0 # Default to Logistic Regression based on its 99% accuracy in your notebook
    )
    
    st.markdown("### 2. Enter Rice Grain Metrics")
    
    # Generate input fields dynamically in a 3-column layout
    cols = st.columns(3)
    user_inputs = {}
    
    for i, feature in enumerate(feature_list):
        with cols[i % 3]:
            user_inputs[feature] = st.number_input(
                feature, 
                min_value=0.0, 
                max_value=bounds[feature]["max"] * 2.0, # Allow slightly larger inputs than max
                value=bounds[feature]["mean"],
                format="%.5f"
            )

    st.markdown("---")

    if st.button("Classify Rice Grain", type="primary", use_container_width=True):
        input_df = pd.DataFrame([user_inputs])
        input_scaled = fitted_scaler.transform(input_df)
        active_model = models_dict[selected_model_name]
        prediction = active_model.predict(input_scaled)[0]
        
        try:
            probabilities = active_model.predict_proba(input_scaled)[0]
            confidence = max(probabilities) * 100
        except AttributeError:
            confidence = None

        rice_type = "Osmancik" if prediction == 1 else "Cammeo"
        result_color = "#2ecc71" if prediction == 1 else "#f1c40f"
        
        st.markdown(
            f"""
            <div style="background-color: {result_color}; padding: 20px; border-radius: 10px; text-align: center; color: #2c3e50;">
                <h2 style="margin:0; color: #2c3e50;">Classification Result: <b>{rice_type}</b></h2>
            </div>
            """, unsafe_allow_html=True
        )
        
        if confidence:
            st.caption(f"**Model Confidence:** {confidence:.2f}% (Using {selected_model_name})")

# ------------------------------------------------------------
# TAB 2: EDA & MODEL EVALUATION
# ------------------------------------------------------------
with tab_insights:
    st.markdown("### 🏆 Model Performance Comparison")
    
    # Model Performance Table
    metrics_df = pd.DataFrame(evaluation_results).T
    st.dataframe(metrics_df.style.highlight_max(axis=0, color="#2ecc71"), use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 📈 Exploratory Data Analysis")
    
    # Configure Seaborn to prevent Dark Mode text clipping
    sns.set_theme(style="whitegrid", rc={
        "figure.facecolor": "white", 
        "axes.facecolor": "white", 
        "text.color": "black", 
        "axes.labelcolor": "black", 
        "xtick.color": "black", 
        "ytick.color": "black"
    })

    # Sub-tabs for Graphs
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["🔥 Correlation Heatmap", "📏 Area & Perimeter Plots", "📊 Class Distributions"])

    # 1. Heatmap
    with viz_tab1:
        st.markdown("#### Feature Correlation Heatmap")
        fig1, ax1 = plt.subplots(figsize=(12, 8), facecolor='white')
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1, ax=ax1)
        ax1.set_title("Correlation Heatmap", fontsize=14, fontweight='bold', color='black')
        st.pyplot(fig1, theme=None)

    # 2. Area Scatter & Perimeter Box Plot (From your Notebook)
    with viz_tab2:
        colA, colB = st.columns(2)
        
        with colA:
            st.markdown("#### Scatter Plot of Area")
            fig2, ax2 = plt.subplots(figsize=(8, 5), facecolor='white')
            ax2.scatter(range(len(df['Area'])), df['Area'], color='#3498db', alpha=0.5)
            ax2.set_xlabel("Index", color="black")
            ax2.set_ylabel("Area", color="black")
            ax2.set_title("Scatter Plot of Area", fontsize=14, fontweight='bold', color='black')
            st.pyplot(fig2, theme=None)
            
        with colB:
            st.markdown("#### Box Plot of Perimeter")
            fig3, ax3 = plt.subplots(figsize=(6, 5), facecolor='white')
            ax3.boxplot(df['Perimeter'])
            ax3.set_ylabel("Perimeter", color="black")
            ax3.set_title("Box Plot of Perimeter", fontsize=14, fontweight='bold', color='black')
            st.pyplot(fig3, theme=None)

    # 3. Distributions
    with viz_tab3:
        colC, colD = st.columns(2)
        
        plot_df = df.copy()
        plot_df['Class Name'] = plot_df['Class'].map({0: 'Cammeo', 1: 'Osmancik'})
        
        with colC:
            st.markdown("#### Target Class Distribution")
            fig4, ax4 = plt.subplots(figsize=(6, 5), facecolor='white')
            sns.countplot(data=plot_df, x='Class Name', palette=["#f1c40f", "#2ecc71"], ax=ax4)
            ax4.set_ylabel("Count", color="black")
            ax4.set_xlabel("Rice Class", color="black")
            st.pyplot(fig4, theme=None)
            
        with colD:
            st.markdown("#### Area vs Perimeter by Class")
            fig5, ax5 = plt.subplots(figsize=(8, 5), facecolor='white')
            sns.scatterplot(data=plot_df, x='Area', y='Perimeter', hue='Class Name', palette=["#f1c40f", "#2ecc71"], alpha=0.7, ax=ax5)
            st.pyplot(fig5, theme=None)
