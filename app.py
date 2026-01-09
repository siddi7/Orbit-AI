# app.py
import os
import time
import base64
from pathlib import Path
from datetime import datetime
import io

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Machine Learning Imports
from sklearn.model_selection import (cross_val_score, LeaveOneOut, KFold, 
                                     cross_val_predict, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                              RandomForestClassifier, GradientBoostingClassifier)
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, 
                                  LogisticRegression, RidgeClassifier)
from sklearn.svm import SVR, SVC
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc, classification_report)

# Optional Imports (Robust Checks)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

# Setup
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="ORBIT-Data Analyst Pro",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Animation & Style ---
st.markdown("""
<style>
    .reportview-container { background: #0e1117; }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; }
    .stButton>button {
        background: linear-gradient(90deg, #4db8ff 0%, #0066cc 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
    }
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4db8ff;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

def detect_problem_type(df, target_col):
    """
    Heuristic to detect Regression vs Classification.
    """
    unique_count = df[target_col].nunique()
    dtype = df[target_col].dtype
    
    if pd.api.types.is_numeric_dtype(dtype):
        if unique_count < 20: # Low cardinality numeric -> likely Classification
            return "Classification"
        else:
            return "Regression"
    else:
        return "Classification"

def save_artifact(content, filename, is_plot=False):
    artifact_folder = OUTPUT_DIR / f"run_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    artifact_folder.mkdir(parents=True, exist_ok=True)
    path = artifact_folder / filename
    
    if is_plot:
        content.write_image(str(path))
    else:
        path.write_text(content)
    return path

# --- Main App ---

st.title("🌌 ORBIT-Data Analyst Pro")
st.markdown("### Autonomous AI Architecture for Data Science")

# Sidebar
st.sidebar.header("1. Data Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
use_example = st.sidebar.checkbox("Use Example Data", value=(uploaded_file is None))

# Initialize session state for persistent results
if "results" not in st.session_state:
    st.session_state.results = None
if "best_model" not in st.session_state:
    st.session_state.best_model = None
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

# Load Data
df = None
if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif use_example:
    # Richer Example Dataset
    np.random.seed(42)
    n = 200
    years = np.random.randint(2020, 2026, n)
    experience = np.random.randint(0, 15, n)
    education = np.random.choice(["Bachelors", "Masters", "PhD"], n)
    role = np.random.choice(["Data Scientist", "AI Engineer", "Analyst"], n)
    salary = (50000 + 4000 * experience + \
              (np.where(education=="Masters", 10000, 0)) + \
              (np.where(education=="PhD", 25000, 0)) + \
              (np.where(role=="AI Engineer", 20000, 0)) + \
              np.random.normal(0, 5000, n)).astype(int)
    churn = (np.random.rand(n) < (0.1 + 0.05 * (salary < 70000))).astype(int) # 0 or 1
    
    df = pd.DataFrame({
        "Year": years, "Experience": experience, "Education": education, 
        "Role": role, "Salary": salary, "Churn": churn
    })

if df is not None:
    st.sidebar.success(f"Loaded {len(df)} rows, {len(df.columns)} cols")
    
    # Target Selection
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    all_cols = df.columns.tolist()
    
    st.sidebar.header("2. Objective Settings")
    
    # Default selection heuristic
    default_target = "Salary" if "Salary" in all_cols else ("Churn" if "Churn" in all_cols else all_cols[-1])
    target = st.sidebar.selectbox("Select Target Variable", all_cols, index=all_cols.index(default_target))
    
    # Problem Type Detection
    detected_type = detect_problem_type(df, target)
    problem_type = st.sidebar.radio("Problem Type", ["Regression", "Classification"], 
                                    index=0 if detected_type == "Regression" else 1)
    
    st.sidebar.info(f"Detected: **{detected_type}**")

    # --- TABS ---
    tab_eda, tab_model, tab_explain, tab_report = st.tabs([
        "🔍 Data Analysis (EDA)", 
        "🤖 Auto-Modeling", 
        "💡 Explainability (SHAP/LIME)", 
        "📝 Executive Report"
    ])
    
    # --- TAB 1: EDA ---
    with tab_eda:
        st.header("Deep Exploratory Data Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            st.subheader("Missing Values")
            missing = df.isnull().sum()
            if missing.sum() > 0:
                st.bar_chart(missing)
            else:
                st.success("No missing values detected.")
                
        with col2:
            st.subheader("Target Distribution")
            if problem_type == "Regression":
                fig = px.histogram(df, x=target, marginal="box", title=f"Distribution of {target}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.pie(df, names=target, title=f"Distribution of {target}")
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Multivariate Analysis")
        
        # Interactive Helper
        cols_for_scatter = [c for c in numeric_cols if c != target]
        if cols_for_scatter:
            x_axis = st.selectbox("X-Axis", cols_for_scatter)
            fig_scatter = px.scatter(df, x=x_axis, y=target, color=target if problem_type=="Classification" else None,
                                     title=f"{x_axis} vs {target}", trendline="ols" if problem_type=="Regression" else None)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        # Correlation Heatmap (only numeric)
        if len(numeric_cols) > 1:
            st.subheader("Correlation Matrix")
            corr = df[numeric_cols].corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
            st.plotly_chart(fig_corr, use_container_width=True)

    # --- TAB 2: MODELING ---
    with tab_model:
        st.header("Autonomous Modeling Pipeline")
        st.markdown(f"Running optimized **{problem_type}** pipeline.")
        
        feature_cols = [c for c in df.columns if c != target]
        
        if st.button("🚀 Launch Auto-ML"):
            with st.status("Training Models...", expanded=True) as status:
                
                # Preprocessing
                st.write("🛠️ Preprocessing Data...")
                X = df[feature_cols]
                y = df[target]
                
                # Identify column types
                num_features = X.select_dtypes(include=np.number).columns.tolist()
                cat_features = X.select_dtypes(exclude=np.number).columns.tolist()
                
                # Transformers
                num_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ])
                
                cat_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore'))
                ])
                
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', num_transformer, num_features),
                        ('cat', cat_transformer, cat_features)
                    ])
                
                # Model Selection
                models = []
                if problem_type == "Regression":
                    models = [
                        ("Linear Regression", LinearRegression()),
                        ("Ridge", Ridge()),
                        ("Lasso", Lasso()),
                        ("Random Forest", RandomForestRegressor(n_estimators=100, n_jobs=-1)),
                        ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100)),
                        ("SVR", SVR())
                    ]
                    if XGBOOST_AVAILABLE:
                        models.append(("XGBoost", xgb.XGBRegressor(n_jobs=-1)))
                        
                else: # Classification
                    models = [
                        ("Logistic Regression", LogisticRegression()),
                        ("Random Forest", RandomForestClassifier(n_estimators=100, n_jobs=-1)),
                        ("Gradient Boosting", GradientBoostingClassifier(n_estimators=100)),
                        ("SVC", SVC(probability=True))
                    ]
                    if XGBOOST_AVAILABLE:
                        models.append(("XGBoost", xgb.XGBClassifier(n_jobs=-1)))

                # Training Loop
                results = []
                # CV Strategy
                cv = LeaveOneOut() if len(df) < 50 else KFold(n_splits=5, shuffle=True, random_state=42)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Save preprocessor fit on train
                preprocessor.fit(X_train)
                st.session_state.preprocessor = preprocessor # Save for LIME/SHAP
                
                best_score = -float("inf")
                best_model_name = ""
                best_pipeline_obj = None

                for name, model in models:
                    st.write(f"⏳ Training {name}...")
                    
                    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                              ('model', model)])
                    
                    try:
                        # Cross Validation
                        if problem_type == "Regression":
                            scores = cross_val_score(pipeline, X, y, cv=cv, scoring='r2')
                            avg_score = scores.mean()
                            
                            # Fit full for testing
                            pipeline.fit(X_train, y_train)
                            y_pred = pipeline.predict(X_test)
                            test_score = r2_score(y_test, y_pred)
                            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                            
                            results.append({
                                "Model": name, "CV R2": avg_score, "Test R2": test_score, "RMSE": rmse
                            })
                        else:
                            scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
                            avg_score = scores.mean()
                            
                            pipeline.fit(X_train, y_train)
                            y_pred = pipeline.predict(X_test)
                            test_score = accuracy_score(y_test, y_pred)
                            
                            results.append({
                                "Model": name, "CV Accuracy": avg_score, "Test Accuracy": test_score
                            })

                        if avg_score > best_score:
                            best_score = avg_score
                            best_model_name = name
                            best_pipeline_obj = pipeline

                    except Exception as e:
                        st.error(f"Failed {name}: {e}")

                st.session_state.results = pd.DataFrame(results)
                st.session_state.best_model = best_model_name
                st.session_state.pipeline = best_pipeline_obj
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                
                status.update(label="Training Complete!", state="complete", expanded=False)
                
        # Display Results
        if st.session_state.results is not None:
            st.subheader("🏆 Model Leaderboard")
            res_df = st.session_state.results.sort_values(by="CV R2" if problem_type=="Regression" else "CV Accuracy", ascending=False)
            
            # Formatted Dataframe
            st.dataframe(res_df.style.format("{:.3f}", subset=res_df.select_dtypes(include=np.number).columns), use_container_width=True)
            
            st.success(f"Best Model: **{st.session_state.best_model}**")
            
            # Visualizations
            best_pipeline = st.session_state.pipeline
            y_pred = best_pipeline.predict(st.session_state.X_test)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Residuals / Predictions")
                if problem_type == "Regression":
                    fig_pred = px.scatter(x=st.session_state.y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'}, title="Predicted vs Actual")
                    fig_pred.add_shape(type="line", x0=df[target].min(), y0=df[target].max(), x1=df[target].min(), y1=df[target].max(), line=dict(dash='dash'))
                    st.plotly_chart(fig_pred, use_container_width=True)
                else:
                    cm = confusion_matrix(st.session_state.y_test, y_pred)
                    fig_cm = px.imshow(cm, text_auto=True, title="Confusion Matrix")
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
            with col2:
                st.subheader("Feature Importance")
                # Try extracting importance
                model_step = best_pipeline.named_steps['model']
                if hasattr(model_step, 'feature_importances_'):
                    # Getting feature names is tricky with ColumnTransformer, simplified approach
                    # This is an approximation for visual 'wow' factor primarily unless we meticulously map back
                    importances = model_step.feature_importances_
                    # Just number them if names are hard to map reliably in 1 file
                    feat_count = len(importances)
                    fig_imp = px.bar(x=range(feat_count), y=importances, title="Feature Importances (Index)", labels={'x':'Feature Index', 'y':'Importance'})
                    st.plotly_chart(fig_imp, use_container_width=True)
                elif hasattr(model_step, 'coef_'):
                    coefs = model_step.coef_[0] if problem_type == "Classification" and len(model_step.coef_.shape)>1 else model_step.coef_
                    fig_coef = px.bar(x=range(len(coefs)), y=coefs, title="Coefficients (Index)")
                    st.plotly_chart(fig_coef, use_container_width=True)
                else:
                    st.info("Feature importance not directly available for this model type.")

    # --- TAB 3: EXPLAINABILITY ---
    with tab_explain:
        st.header("💡 Why did the model predict this?")
        
        if st.session_state.pipeline is None:
            st.warning("Please train models first!")
        else:
            if LIME_AVAILABLE:
                st.subheader("Lime Explanation (Local)")
                st.markdown("Select a specific instance from the test set to explain.")
                
                idx = st.slider("Select Instance Index", 0, len(st.session_state.X_test)-1, 0)
                selected_instance = st.session_state.X_test.iloc[idx]
                st.write("Instance Data:", selected_instance.to_dict())
                
                # LIME Helper
                # We need to pass the prediction function that accepts raw data (since pipeline handles preprocessing)
                
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data=st.session_state.X_train.values, # LIME likes numpy
                    feature_names=st.session_state.X_train.columns.tolist(),
                    class_names=[str(c) for c in st.session_state.pipeline.classes_] if problem_type=="Classification" else None,
                    mode=problem_type.lower()
                )
                
                # Create a wrapper for prediction that uses the pipeline
                predict_fn = st.session_state.pipeline.predict_proba if problem_type=="Classification" else st.session_state.pipeline.predict
                
                try:
                    # Pass the instance as numpy
                    explanation = explainer.explain_instance(
                        selected_instance.values, 
                        predict_fn, 
                        num_features=10
                    )
                    # Enable raw HTML for LIME
                    st.components.v1.html(explanation.as_html(), height=400, scrolling=True)
                except Exception as e:
                    st.error(f"LIME Error: {e}")
            else:
                st.warning("LIME not installed. Check requirements.")
                
            st.markdown("---")
            
            if SHAP_AVAILABLE:
                st.subheader("SHAP (Global Importance)")
                try:
                    # Using KernelExplainer is generic but slow. TreeExplainer is better but requires accessing the inner model.
                    # Simplified: Use KernelExplainer on smaller sample
                    st.write("Computing SHAP values... (this involves complex math, please wait)")
                    
                    model = st.session_state.pipeline.named_steps['model']
                    preprocessor = st.session_state.pipeline.named_steps['preprocessor']
                    
                    X_train_trans = preprocessor.transform(st.session_state.X_train)
                    X_test_trans = preprocessor.transform(st.session_state.X_test)
                    
                    # Subsample for speed
                    X_subs = shap.kmeans(X_train_trans, 10) 
                    
                    explainer = shap.KernelExplainer(model.predict, X_subs)
                    shap_values = explainer.shap_values(X_test_trans[0:10]) # Only explain first 10 for demo speed
                    
                    st.pyplot(shap.summary_plot(shap_values, X_test_trans[0:10], show=False))
                    
                except Exception as e:
                    st.warning(f"SHAP Error (Complex Pipeline): {e}")

    # --- TAB 4: REPORT ---
    with tab_report:
        st.header("📝 Executive Summary")
        
        if st.session_state.results is not None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            report_md = f"""
# AI Data Analysis Report
**Generated on:** {timestamp}

## 1. Objective
Analyze the dataset to predict **{target}** using **{problem_type}** models.

## 2. Data Insights
- **Rows:** {len(df)}
- **Columns:** {len(df.columns)}
- **Missing Values:** {df.isnull().sum().sum()}

## 3. Model Performance
The best performing model was **{st.session_state.best_model}**.

### Leaderboard
{st.session_state.results.to_markdown(index=False)}

## 4. Key Predictors
(Refer to the Feature Importance section in the Modeling tab for visual details.)

## 5. Conclusion
Based on the analysis, the {st.session_state.best_model} offers the most reliable predictions.
            """
            st.markdown(report_md)
            
            st.download_button(
                label="📥 Download Full Report",
                data=report_md,
                file_name="orbit_ai_executive_report.md",
                mime="text/markdown"
            )
        else:
            st.info("Run the modeling pipeline to generate the report.")
