# # pages/model_interpretability.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import shap

# def render():
#     st.markdown('<h2 class="section-header">Model Interpretability & SHAP Analysis</h2>', unsafe_allow_html=True)
    
#     if 'models_trained' not in st.session_state or not st.session_state.models_trained:
#         st.warning("No trained models found. Please train a model first.")
#     else:
#         st.info("Understand model predictions with SHAP values and feature importance.")
        
#         # Select model
#         model_names = list(st.session_state.models_trained.keys())
#         selected_model = st.selectbox("Select Model:", model_names)
        
#         if selected_model:
#             model_data = st.session_state.models_trained[selected_model]
            
#             # Get model and data
#             model = model_data.get('model')
#             features = model_data.get('features', [])
#             target = model_data.get('target', 'Unknown')
            
#             if model and features:
#                 st.subheader(f"📋 Model: {selected_model}")
#                 st.write(f"**Target:** {target}")
#                 st.write(f"**Features used:** {len(features)} features")
                
#                 # Get data for SHAP
#                 df = st.session_state.df
#                 X = df[features].fillna(0)
                
#                 # Scale if needed
#                 if 'scaler' in model_data:
#                     scaler = model_data['scaler']
#                     X_scaled = scaler.transform(X)
#                 else:
#                     X_scaled = X.values
                
#                 # SHAP analysis
#                 if st.button("🔍 Run SHAP Analysis", type="primary"):
#                     with st.spinner("Calculating SHAP values..."):
#                         try:
#                             # Create explainer
#                             try:
#                                 if hasattr(model, 'predict_proba') and hasattr(model, 'estimators_'):
#                                     # Tree-based models
#                                     explainer = shap.TreeExplainer(model)
#                                     shap_values = explainer.shap_values(X_scaled)
#                                 else:
#                                     # For other models, use KernelExplainer
#                                     # Ensure data is 2D for SHAP
#                                     if X_scaled.ndim == 1:
#                                         X_scaled = X_scaled.reshape(-1, 1)
                                    
#                                     # Use smaller sample for speed
#                                     sample_size = min(100, len(X_scaled))
#                                     X_sample = X_scaled[:sample_size] if len(X_scaled) > 100 else X_scaled
                                    
#                                     explainer = shap.KernelExplainer(model.predict, X_sample)
#                                     shap_values = explainer.shap_values(X_sample)
#                             except Exception as e:
#                                 st.warning(f"SHAP TreeExplainer failed: {e}")
#                                 # Fallback to simpler method
#                                 if X_scaled.ndim == 1:
#                                     X_scaled = X_scaled.reshape(-1, 1)
                                
#                                 sample_size = min(50, len(X_scaled))
#                                 X_sample = X_scaled[:sample_size] if len(X_scaled) > 50 else X_scaled
                                
#                                 explainer = shap.KernelExplainer(model.predict, X_sample)
#                                 shap_values = explainer.shap_values(X_sample)

#                             # Store SHAP values
#                             st.session_state.shap_values[selected_model] = {
#                                 'explainer': explainer,
#                                 'shap_values': shap_values,
#                                 'features': features
#                             }
                            
#                             # Create visualizations
#                             create_shap_visualizations(shap_values, X_scaled, features, X, explainer, model)
                            
#                             st.success("✅ SHAP analysis completed!")
                            
#                         except Exception as e:
#                             st.error(f"SHAP analysis failed: {e}")
#                             st.info("Some models may not be compatible with SHAP. Try a tree-based model like Random Forest.")

# def create_shap_visualizations(shap_values, X_scaled, features, X, explainer, model):
#     """Helper function to create SHAP visualizations"""
#     try:
#         # Ensure shap_values is in correct format
#         if isinstance(shap_values, list):
#             # If it's a list, take the first element (usually for classification)
#             shap_values_array = shap_values[0] if len(shap_values) > 0 else shap_values
#         else:
#             shap_values_array = shap_values
        
#         # Ensure data is 2D
#         if X_scaled.ndim == 1:
#             X_scaled = X_scaled.reshape(-1, 1)
        
#         # 1. Summary plot
#         st.subheader("📊 Feature Importance Summary")
#         try:
#             fig, ax = plt.subplots(figsize=(10, 6))
#             shap.summary_plot(shap_values_array, X_scaled, feature_names=features, show=False)
#             plt.tight_layout()
#             st.pyplot(fig)
#         except Exception as e:
#             st.warning(f"Could not create summary plot: {e}")
        
#         # 2. Bar plot of mean absolute SHAP values
#         st.subheader("📈 Mean Absolute SHAP Values")
#         try:
#             # Calculate mean absolute SHAP values
#             if shap_values_array.ndim == 2:
#                 shap_abs = np.abs(shap_values_array).mean(axis=0)
#             elif shap_values_array.ndim == 1:
#                 shap_abs = np.abs(shap_values_array)
#             else:
#                 st.warning("Unexpected SHAP values shape")
#                 return
            
#             # Ensure we have the right number of features
#             if len(shap_abs) != len(features):
#                 # If dimensions don't match, take mean across all except features dimension
#                 if shap_values_array.ndim > 1:
#                     shap_abs = np.abs(shap_values_array).mean(axis=tuple(range(shap_values_array.ndim-1)))
            
#             importance_df = pd.DataFrame({
#                 'Feature': features[:len(shap_abs)],
#                 'Importance': shap_abs[:len(features)]
#             }).sort_values('Importance', ascending=False)
            
#             fig, ax = plt.subplots(figsize=(10, 6))
#             top_n = min(15, len(importance_df))
#             ax.barh(importance_df['Feature'][:top_n], importance_df['Importance'][:top_n])
#             ax.set_xlabel('Mean |SHAP value|')
#             ax.set_title(f'Top {top_n} Features by Importance')
#             ax.invert_yaxis()
#             plt.tight_layout()
#             st.pyplot(fig)
#         except Exception as e:
#             st.warning(f"Could not create bar plot: {e}")
        
#         # 3. Dependence plots for top features
#         st.subheader("🔗 Feature Dependence Plots")
#         try:
#             # Get top features from importance
#             if 'importance_df' in locals() and len(importance_df) > 0:
#                 top_features = importance_df['Feature'].head(min(3, len(importance_df))).tolist()
                
#                 for i, feature in enumerate(top_features):
#                     if feature in features:
#                         col1, col2 = st.columns(2)
#                         with col1:
#                             try:
#                                 fig, ax = plt.subplots(figsize=(8, 4))
#                                 shap.dependence_plot(
#                                     feature, shap_values_array, X_scaled, 
#                                     feature_names=features, ax=ax, show=False
#                                 )
#                                 ax.set_title(f'Dependence Plot: {feature}')
#                                 plt.tight_layout()
#                                 st.pyplot(fig)
#                             except:
#                                 st.write(f"Could not create dependence plot for {feature}")
                        
#                         with col2:
#                             try:
#                                 # Show feature distribution
#                                 fig, ax = plt.subplots(figsize=(8, 4))
#                                 if feature in X.columns:
#                                     ax.hist(X[feature].dropna(), bins=30, alpha=0.7)
#                                     ax.set_title(f'Distribution: {feature}')
#                                     ax.set_xlabel(feature)
#                                     ax.set_ylabel('Frequency')
#                                     plt.tight_layout()
#                                     st.pyplot(fig)
#                             except:
#                                 st.write(f"Could not show distribution for {feature}")
#         except Exception as e:
#             st.warning(f"Could not create dependence plots: {e}")
        
#         # 4. Waterfall plot for individual predictions
#         st.subheader("💧 Individual Prediction Explanations")
#         try:
#             sample_idx = st.slider("Select sample index", 0, min(50, len(X)-1), 0)
            
#             col_pred1, col_pred2 = st.columns(2)
            
#             with col_pred1:
#                 try:
#                     # Waterfall plot
#                     fig, ax = plt.subplots(figsize=(10, 6))
#                     shap.waterfall_plot(
#                         shap.Explanation(
#                             values=shap_values_array[sample_idx] if hasattr(shap_values_array, '__len__') and len(shap_values_array) > sample_idx else shap_values_array,
#                             base_values=explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
#                             data=X_scaled[sample_idx] if len(X_scaled) > sample_idx else X_scaled[0],
#                             feature_names=features
#                         ),
#                         show=False
#                     )
#                     plt.tight_layout()
#                     st.pyplot(fig)
#                 except Exception as e:
#                     st.write(f"Could not create waterfall plot: {e}")
            
#             with col_pred2:
#                 # Show actual feature values
#                 st.write("**Feature Values for Sample:**")
#                 if len(X) > sample_idx:
#                     sample_data = X.iloc[sample_idx]
#                     for feature in features[:5]:  # Show only top 5 features
#                         if feature in sample_data.index:
#                             st.write(f"• **{feature}:** {sample_data[feature]:.4f}")
                
#                 # Show prediction
#                 try:
#                     if len(X_scaled) > sample_idx:
#                         sample_input = X_scaled[sample_idx:sample_idx+1]
#                         if hasattr(model, 'predict_proba'):
#                             pred_proba = model.predict_proba(sample_input)
#                             if len(pred_proba[0]) > 1:
#                                 st.write(f"**Predicted Probability (Class 1):** {pred_proba[0][1]:.3f}")
#                             else:
#                                 st.write(f"**Predicted Probability:** {pred_proba[0][0]:.3f}")
#                         else:
#                             pred = model.predict(sample_input)
#                             st.write(f"**Prediction:** {pred[0]:.3f}")
#                 except:
#                     st.write("Could not get prediction for sample")
                    
#         except Exception as e:
#             st.warning(f"Could not create individual explanations: {e}")
            
#     except Exception as e:
#         st.error(f"Error creating SHAP visualizations: {e}")

# pages/model_interpretability.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

def render():
    st.markdown('<h2 class="section-header">Model Interpretability & SHAP Analysis</h2>', unsafe_allow_html=True)
    
    if 'models_trained' not in st.session_state or not st.session_state.models_trained:
        st.warning("No trained models found. Please train a model first.")
    else:
        st.info("Understand model predictions with SHAP values and feature importance.")
        
        # Select model
        model_names = list(st.session_state.models_trained.keys())
        selected_model = st.selectbox("Select Model:", model_names)
        
        if selected_model:
            model_data = st.session_state.models_trained[selected_model]
            model = model_data.get('model')
            features = model_data.get('features', [])
            target = model_data.get('target', 'Unknown')
            
            if model and features:
                st.subheader(f"📋 Model: {selected_model}")
                st.write(f"**Target:** {target}")
                st.write(f"**Features used:** {len(features)} features")
                
                # Get data for SHAP
                df = st.session_state.df
                X = df[features].fillna(0)
                
                # Scale if needed
                if 'scaler' in model_data:
                    scaler = model_data['scaler']
                    X_scaled = scaler.transform(X)
                else:
                    X_scaled = X.values
                
                # SHAP analysis
                if st.button("🔍 Run SHAP Analysis", type="primary"):
                    with st.spinner("Calculating SHAP values..."):
                        try:
                            # Create explainer based on model type
                            explainer, shap_values = create_shap_explainer(model, X_scaled, features)
                            
                            if explainer and shap_values is not None:
                                # Store SHAP values
                                st.session_state.shap_values[selected_model] = {
                                    'explainer': explainer,
                                    'shap_values': shap_values,
                                    'features': features
                                }
                                
                                # Create visualizations
                                create_shap_visualizations(shap_values, X_scaled, features, X, explainer, model)
                                
                                st.success("✅ SHAP analysis completed!")
                            else:
                                st.error("Could not create SHAP explainer for this model type.")
                                
                        except Exception as e:
                            st.error(f"SHAP analysis failed: {e}")
                            st.info("This model might not be compatible with SHAP. Try a tree-based model like Random Forest.")

def create_shap_explainer(model, X_scaled, features):
    """Create appropriate SHAP explainer based on model type"""
    try:
        # Check model type
        model_type = type(model).__name__
        st.info(f"Model type: {model_type}")
        
        # Limit sample size for performance
        sample_size = min(100, len(X_scaled))
        X_sample = X_scaled[:sample_size]
        
        # Handle different model types
        if hasattr(model, 'estimators_'):  # Tree-based ensemble
            st.write("Using TreeExplainer for tree-based model...")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
        elif 'XGB' in model_type or 'XGBoost' in model_type:
            st.write("Using TreeExplainer for XGBoost...")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
        elif 'LGBM' in model_type or 'LightGBM' in model_type:
            st.write("Using TreeExplainer for LightGBM...")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
        elif 'RandomForest' in model_type or 'DecisionTree' in model_type:
            st.write("Using TreeExplainer for tree model...")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
        else:
            # For other models, use KernelExplainer
            st.write("Using KernelExplainer (slower)...")
            
            # Define prediction function
            if hasattr(model, 'predict_proba'):
                # For classification
                predict_func = lambda x: model.predict_proba(x)[:, 1] if hasattr(model.predict_proba(x), 'shape') and model.predict_proba(x).shape[1] > 1 else model.predict_proba(x)
            else:
                # For regression
                predict_func = model.predict
            
            # Use background data
            background = shap.sample(X_sample, 10) if len(X_sample) > 10 else X_sample
            
            explainer = shap.KernelExplainer(predict_func, background)
            shap_values = explainer.shap_values(X_sample, nsamples=50)  # Limit samples for speed
        
        return explainer, shap_values
        
    except Exception as e:
        st.warning(f"Could not create SHAP explainer: {e}")
        return None, None

def create_shap_visualizations(shap_values, X_scaled, features, X, explainer, model):
    """Create SHAP visualizations with robust error handling"""
    
    # Debug info
    st.write("**Debug Information:**")
    st.write(f"SHAP values type: {type(shap_values)}")
    if hasattr(shap_values, 'shape'):
        st.write(f"SHAP values shape: {shap_values.shape}")
    elif isinstance(shap_values, list):
        st.write(f"SHAP values list length: {len(shap_values)}")
        for i, val in enumerate(shap_values[:3]):
            if hasattr(val, 'shape'):
                st.write(f"  List item {i} shape: {val.shape}")
    
    # Prepare SHAP values
    shap_values_processed = prepare_shap_values(shap_values)
    
    if shap_values_processed is None:
        st.error("Could not process SHAP values")
        return
    
    # 1. Feature Importance Bar Chart (Always works)
    create_feature_importance_bar(shap_values_processed, features)
    
    # 2. Summary Plot (if we have valid data)
    if shap_values_processed.ndim == 2 and shap_values_processed.shape[1] == len(features):
        try:
            create_summary_plot(shap_values_processed, X_scaled, features)
        except:
            st.warning("Could not create summary plot")
    
    # 3. Individual Prediction Explanation
    try:
        create_individual_explanation(shap_values_processed, X_scaled, features, X, explainer, model)
    except:
        st.warning("Could not create individual explanation")

def prepare_shap_values(shap_values):
    """Process SHAP values into standard format"""
    if shap_values is None:
        return None
    
    # If it's a list (common for classification)
    if isinstance(shap_values, list):
        if len(shap_values) == 2:  # Binary classification
            # Usually second element is for class 1
            return shap_values[1]
        elif len(shap_values) == 1:  # Single output
            return shap_values[0]
        else:  # Multi-class, take mean across classes
            return np.mean(shap_values, axis=0)
    
    # If already numpy array
    if isinstance(shap_values, np.ndarray):
        # If 3D (samples, features, classes), average across classes
        if shap_values.ndim == 3:
            return np.mean(shap_values, axis=2)
        return shap_values
    
    return None

def create_feature_importance_bar(shap_values, features):
    """Create bar chart of feature importance"""
    st.subheader("📈 Feature Importance")
    
    try:
        # Calculate mean absolute SHAP values
        if shap_values.ndim == 2:  # 2D array
            shap_abs = np.abs(shap_values).mean(axis=0)
        elif shap_values.ndim == 1:  # 1D array
            shap_abs = np.abs(shap_values)
        else:
            st.error(f"Unexpected SHAP values dimension: {shap_values.ndim}")
            return
        
        # Ensure we have correct number of features
        n_features = min(len(shap_abs), len(features))
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'Feature': features[:n_features],
            'Importance': shap_abs[:n_features]
        }).sort_values('Importance', ascending=False)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        top_n = min(15, len(importance_df))
        
        if top_n > 0:
            bars = ax.barh(range(top_n), importance_df['Importance'][:top_n])
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(importance_df['Feature'][:top_n])
            ax.set_xlabel('Mean |SHAP value|')
            ax.set_title(f'Top {top_n} Most Important Features')
            ax.invert_yaxis()
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, importance_df['Importance'][:top_n])):
                ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{val:.4f}', va='center')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show table
            st.write("**Feature Importance Scores:**")
            st.dataframe(importance_df.head(10))
        else:
            st.warning("No features to display")
            
    except Exception as e:
        st.error(f"Error creating feature importance chart: {e}")

def create_summary_plot(shap_values, X_scaled, features):
    """Create SHAP summary plot"""
    st.subheader("📊 SHAP Summary Plot")
    
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_scaled, feature_names=features, show=False)
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not create summary plot: {e}")

def create_individual_explanation(shap_values, X_scaled, features, X, explainer, model):
    """Create individual prediction explanation"""
    st.subheader("💧 Individual Prediction Analysis")
    
    # Select sample
    max_idx = min(50, len(X) - 1)
    if max_idx > 0:
        sample_idx = st.slider("Select sample index", 0, max_idx, 0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Feature Values:**")
            if len(X) > sample_idx:
                sample_data = X.iloc[sample_idx]
                for feature in features[:8]:  # Show top 8 features
                    if feature in sample_data.index:
                        st.write(f"• **{feature}:** {sample_data[feature]:.4f}")
        
        with col2:
            st.write("**Prediction Info:**")
            try:
                # Get prediction
                if len(X_scaled) > sample_idx:
                    sample_input = X_scaled[sample_idx:sample_idx+1]
                    
                    if hasattr(model, 'predict_proba'):
                        pred_proba = model.predict_proba(sample_input)
                        if len(pred_proba[0]) > 1:
                            st.write(f"**Probability (Class 1):** {pred_proba[0][1]:.3f}")
                            st.write(f"**Probability (Class 0):** {pred_proba[0][0]:.3f}")
                        else:
                            st.write(f"**Probability:** {pred_proba[0][0]:.3f}")
                    else:
                        pred = model.predict(sample_input)
                        st.write(f"**Prediction:** {pred[0]:.3f}")
            except Exception as e:
                st.write(f"Could not get prediction: {e}")
        
        # Try to create force plot if we have valid data
        try:
            if shap_values.ndim == 2 and len(shap_values) > sample_idx:
                st.write("**SHAP Force Plot:**")
                
                # Create simple horizontal bar chart for SHAP values
                fig, ax = plt.subplots(figsize=(10, 4))
                
                # Get SHAP values for this sample
                sample_shap = shap_values[sample_idx]
                
                # Sort by absolute value
                idx_sorted = np.argsort(np.abs(sample_shap))[::-1]
                top_idx = idx_sorted[:10]  # Top 10 features
                
                colors = ['red' if val < 0 else 'green' for val in sample_shap[top_idx]]
                
                ax.barh(range(len(top_idx)), sample_shap[top_idx], color=colors)
                ax.set_yticks(range(len(top_idx)))
                ax.set_yticklabels([features[i] for i in top_idx])
                ax.set_xlabel('SHAP Value (Impact on Prediction)')
                ax.set_title(f'Top Features Impact for Sample {sample_idx}')
                ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
                
                plt.tight_layout()
                st.pyplot(fig)
                
        except Exception as e:
            st.info("Simple force plot visualization")
    else:
        st.warning("Not enough samples for individual analysis")