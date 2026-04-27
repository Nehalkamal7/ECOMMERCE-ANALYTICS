# pages/predictive_models.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           roc_auc_score, r2_score, mean_squared_error, mean_absolute_error)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

def render():
    st.markdown('<h2 class="section-header">Advanced Predictive Models</h2>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("Please upload data in the Data Overview page first.")
    else:
        df = st.session_state.df
        
        st.subheader("Select Prediction Task")
        prediction_task = st.selectbox(
            "Choose what to predict:",
            ["Purchase Likelihood", "Customer Lifetime Value", "Churn Prediction", 
             "Product Rating Prediction", "High-Value Customer", "Next Purchase Amount",
             "Customer Segmentation", "Browsing Behavior"]
        )
        
        # Advanced model selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            with st.expander("⚙️ Advanced Model Configuration", expanded=True):
                # Model selection
                model_type = st.selectbox(
                    "Select Model Type:",
                    ["Random Forest", "XGBoost", "LightGBM"]
                )
                
                # Advanced options
                use_cross_validation = st.checkbox("Use Cross-Validation", value=True)
                handle_imbalance = st.checkbox("Handle Class Imbalance", value=True)
                feature_selection = st.checkbox("Auto Feature Selection", value=False)
                
                # Hyperparameter tuning
                tune_hyperparams = st.checkbox("Tune Hyperparameters", value=False)
                if tune_hyperparams:
                    tuning_method = st.radio(
                        "Tuning Method:",
                        ["Grid Search", "Random Search", "Bayesian Optimization"]
                    )
                
                # Feature selection
                available_features = [col for col in df.select_dtypes(include=[np.number]).columns 
                                    if col != 'Customer ID']
                selected_features = st.multiselect(
                    "Select Features:",
                    available_features,
                    default=available_features[:min(10, len(available_features))]
                )
        
        with col2:
            st.subheader("🎯 Training Settings")
            
            test_size = st.slider("Test Set Size (%)", 10, 40, 20)
            random_state = st.number_input("Random State", value=42)
            
            if use_cross_validation:
                cv_folds = st.slider("CV Folds", 3, 10, 5)
            
            if st.button("🚀 Train Advanced Model", type="primary", use_container_width=True):
                if len(selected_features) < 2:
                    st.error("Please select at least 2 features.")
                else:
                    with st.spinner("Training advanced model..."):
                        # Prepare data based on prediction task
                        if prediction_task == "Purchase Likelihood":
                            if 'Made_Purchase' not in df.columns:
                                if 'Total_Purchases' in df.columns:
                                    df['Made_Purchase'] = (df['Total_Purchases'] > 0).astype(int)
                                else:
                                    st.error("Cannot create target variable.")
                                    st.stop()
                            target = 'Made_Purchase'
                            problem_type = 'classification'
                        
                        elif prediction_task == "Customer Lifetime Value":
                            if 'Total_Spent' not in df.columns:
                                st.error("'Total_Spent' column required.")
                                st.stop()
                            target = 'Total_Spent'
                            problem_type = 'regression'
                        
                        # Prepare X and y
                        X = df[selected_features].fillna(0)
                        y = df[target]
                        
                        # Handle class imbalance for classification
                        if problem_type == 'classification' and handle_imbalance:
                            smote = SMOTE(random_state=42)
                            X, y = smote.fit_resample(X, y)
                            st.info(f"Applied SMOTE. New dataset shape: {X.shape}")
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size/100, random_state=random_state,
                            stratify=y if problem_type == 'classification' else None
                        )
                        
                        # Scale features
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        # Train model
                        if model_type == "Random Forest":
                            if problem_type == 'classification':
                                model = RandomForestClassifier(n_estimators=100, random_state=random_state)
                            else:
                                model = RandomForestRegressor(n_estimators=100, random_state=random_state)
                        
                        elif model_type == "XGBoost":
                            if problem_type == 'classification':
                                model = xgb.XGBClassifier(random_state=random_state, use_label_encoder=False)
                            else:
                                model = xgb.XGBRegressor(random_state=random_state)
                        
                        elif model_type == "LightGBM":
                            if problem_type == 'classification':
                                model = lgb.LGBMClassifier(random_state=random_state)
                            else:
                                model = lgb.LGBMRegressor(random_state=random_state)
                        
                        # Cross-validation
                        if use_cross_validation:
                            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds,
                                                      scoring='accuracy' if problem_type == 'classification' else 'r2')
                            st.metric(f"CV Score ({cv_folds}-fold)", f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                        
                        # Train final model
                        model.fit(X_train_scaled, y_train)
                        
                        # Evaluate
                        y_pred = model.predict(X_test_scaled)
                        
                        if problem_type == 'classification':
                            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                            
                            # Display metrics
                            col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                            with col_met1:
                                accuracy = accuracy_score(y_test, y_pred)
                                st.metric("Accuracy", f"{accuracy:.3f}")
                            with col_met2:
                                auc = roc_auc_score(y_test, y_pred_proba)
                                st.metric("AUC-ROC", f"{auc:.3f}")
                            with col_met3:
                                precision = precision_score(y_test, y_pred, zero_division=0)
                                st.metric("Precision", f"{precision:.3f}")
                            with col_met4:
                                recall = recall_score(y_test, y_pred, zero_division=0)
                                st.metric("Recall", f"{recall:.3f}")
                        
                        else:  # Regression
                            col_met1, col_met2, col_met3 = st.columns(3)
                            with col_met1:
                                r2 = r2_score(y_test, y_pred)
                                st.metric("R² Score", f"{r2:.3f}")
                            with col_met2:
                                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                                st.metric("RMSE", f"{rmse:.3f}")
                            with col_met3:
                                mae = mean_absolute_error(y_test, y_pred)
                                st.metric("MAE", f"{mae:.3f}")