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
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Rice Type Classifier",
    page_icon="🌾",
    layout="wide"
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier()
    }

    results = {}
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        
        results[name] = {
            "Accuracy": accuracy_score(y_test, preds),
            "F1 Score": f1_score(y_test, preds)
        }
        fitted_models[name] = model

    return fitted_models, scaler, results, X.columns.tolist()

# ============================================================
# MAIN APP EXECUTION & UI
# ============================================================
df = load_and_clean_data()

if df is None:
    st.error("⚠️ Dataset not found! Please ensure 'riceClassification.csv' is in the exact same folder as this script.")
    st.stop()

models_dict, fitted_scaler, evaluation_results, feature_list = train_all_models(df)

# --- Session State for Easy UX (Auto-Fill) ---
if 'input_data' not in st.session_state:
    # Set default values to the mean of the dataset initially
    st.session_state.input_data = {feat: float(df[feat].mean()) for feat in feature_list}

def load_random_sample(class_label):
    """Picks a random grain of the specified class and updates the input fields."""
    sample = df[df['Class'] == class_label].sample(1).iloc[0]
    for feat in feature_list:
        st.session_state.input_data[feat] = float(sample[feat])

# --- UI Layout ---
st.title("🌾 Easy Rice Type Classifier")
st.markdown("Predict whether a rice grain is **Cammeo** or **Osmancik** based on its physical properties.")

# SIDEBAR: Settings & Auto-Fill feature
with st.sidebar:
    st.header("⚙️ Settings")
    selected_model_name = st.selectbox("Choose AI Model:", list(models_dict.keys()), index=0)
    
    st.markdown("---")
    st.markdown("### 🪄 Auto-Fill Inputs")
    st.caption("Testing the app? Click a button below to load real measurements from the dataset automatically!")
    
    if st.button("Load Random 'Cammeo' Grain", use_container_width=True):
        load_random_sample(0)
    if st.button("Load Random 'Osmancik' Grain", use_container_width=True):
        load_random_sample(1)

# TABS
tab_predict, tab_insights = st.tabs(["🔮 Make a Prediction", "📊 Data Insights & Graphs"])

# ------------------------------------------------------------
# TAB 1: PREDICTION ENGINE
# ------------------------------------------------------------
with tab_predict:
    st.markdown("### Grain Measurements")
    
    # Generate input fields in a neat 4-column grid
    cols = st.columns(4)
    for i, feature in enumerate(feature_list):
        with cols[i % 4]:
            # The value is tied to session_state so the Auto-Fill buttons update it seamlessly
            st.session_state.input_data[feature] = st.number_input(
                feature, 
                value=st.session_state.input_data[feature],
                format="%.2f"
            )

    st.markdown("---")

    if st.button("Classify Rice Grain", type="primary", use_container_width=True):
        input_df = pd.DataFrame([st.session_state.input_data])
        input_scaled = fitted_scaler.transform(input_df)
        
        active_model = models_dict[selected_model_name]
        prediction = active_model.predict(input_scaled)[0]
        
        try:
            probabilities = active_model.predict_proba(input_scaled)[0]
            confidence = max(probabilities) * 100
        except AttributeError:
            confidence = None

        rice_type = "Osmancik" if prediction == 1 else "Cammeo"
        result_color = "#2ecc71" if prediction == 1 else "#e67e22"
        
        st.markdown(
            f"""
            <div style="background-color: {result_color}; padding: 20px; border-radius: 10px; text-align: center; color: white;">
                <h2 style="margin:0; color: white;">Prediction: <b>{rice_type}</b></h2>
            </div>
            """, unsafe_allow_html=True
        )
        
        if confidence:
            st.caption(f"**Confidence:** {confidence:.2f}%  |  **Model used:** {selected_model_name}")

# ------------------------------------------------------------
# TAB 2: EDA & GRAPHS
# ------------------------------------------------------------
with tab_insights:
    # 1. Model Performance Table
    st.markdown("### 🏆 Model Performance Comparison")
    metrics_df = pd.DataFrame(evaluation_results).T
    st.dataframe(metrics_df.style.highlight_max(axis=0, color="#2ecc71"), use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 📈 Data Visualizations")
    
    # Safely create a label column for plotting
    plot_df = df.copy()
    plot_df['Rice Type'] = plot_df['Class'].map({0: 'Cammeo', 1: 'Osmancik'})
    
    # Safe Plotting configuration
    sns.set_style("whitegrid")
    
    colA, colB = st.columns(2)
    
    with colA:
        st.markdown("**Box Plot: Area by Rice Type**")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=plot_df, x='Rice Type', y='Area', ax=ax1, palette="Set2")
        st.pyplot(fig1)
        plt.close(fig1) # Free memory safely
        
    with colB:
        st.markdown("**Scatter Plot: Area vs Perimeter**")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=plot_df, x='Area', y='Perimeter', hue='Rice Type', alpha=0.6, ax=ax2, palette="Set2")
        st.pyplot(fig2)
        plt.close(fig2)

    st.markdown("**Feature Correlation Heatmap**")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    # Drop 'Class' for the correlation heatmap to focus on measurements
    numeric_features = df.drop(columns=['Class'])
    sns.heatmap(numeric_features.corr(), annot=True, cmap='Blues', fmt='.2f', ax=ax3)
    st.pyplot(fig3)
    plt.close(fig3)
