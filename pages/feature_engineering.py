# pages/feature_engineering.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from utils import extract_purchase_features, extract_browsing_features, extract_text_features

def render():
    st.markdown('<h2 class="section-header">Advanced Feature Engineering</h2>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("Please upload data in the Data Overview page first.")
    else:
        df = st.session_state.df
        
        st.info("Extract advanced features using machine learning techniques.")
        
        # Use different variable names to avoid conflicts
        left_col, right_col = st.columns([3, 1])
        
        with left_col:
            with st.expander("🎯 Advanced Feature Extraction", expanded=True):
                feature_options = st.multiselect(
                    "Select features to extract:",
                    ["Purchase History Features", "Browsing History Features", 
                     "Text Analysis Features", "RFM Analysis", "Time Series Features",
                     "Interaction Features", "Polynomial Features"],
                    default=["Purchase History Features", "RFM Analysis"]
                )
                
                # Advanced options
                st.subheader("⚙️ Advanced Options")
                
                # Initialize with default values
                poly_degree = 2
                create_interactions = True
                normalize_features = True
                
                advanced_options = st.checkbox("Enable advanced feature engineering")
                if advanced_options:
                    poly_degree = st.slider("Polynomial Degree", 2, 4, 2)
                    create_interactions = st.checkbox("Create feature interactions", value=True)
                    normalize_features = st.checkbox("Normalize features", value=True)
                
                if st.button("🚀 Extract Advanced Features", type="primary", use_container_width=True):
                    with st.spinner("Extracting advanced features..."):
                        progress_bar = st.progress(0)
                        
                        # Extract purchase features
                        if "Purchase History Features" in feature_options:
                            if 'Purchase History' in df.columns:
                                progress_bar.progress(10)
                                purchase_features = df['Purchase History'].apply(extract_purchase_features)
                                df = pd.concat([df, purchase_features], axis=1)
                                st.success("✅ Advanced purchase features extracted!")
                            else:
                                st.warning("'Purchase History' column not found.")
                        
                        # Extract browsing features
                        if "Browsing History Features" in feature_options:
                            if 'Browsing History' in df.columns:
                                progress_bar.progress(20)
                                browsing_features = df['Browsing History'].apply(extract_browsing_features)
                                df = pd.concat([df, browsing_features], axis=1)
                                st.success("✅ Browsing behavior features extracted!")
                        
                        # Text analysis
                        if "Text Analysis Features" in feature_options:
                            if 'Product Reviews' in df.columns:
                                progress_bar.progress(30)
                                text_features = df['Product Reviews'].apply(extract_text_features)
                                df = pd.concat([df, text_features], axis=1)
                                st.success("✅ Text analysis features extracted!")
                        
                        # RFM Analysis - SIMPLER VERSION
                        if "RFM Analysis" in feature_options:
                            progress_bar.progress(40)
                            # Simple RFM scoring without qcut
                            rfm_cols = ['Total_Spent', 'Total_Purchases']
                            rfm_present = [col for col in rfm_cols if col in df.columns]
                            
                            if len(rfm_present) >= 2:
                                for col in rfm_present:
                                    # Use simple percentile ranking instead of qcut
                                    df[f'{col}_Percentile'] = df[col].rank(pct=True)
                                    # Convert to 1-5 scale
                                    df[f'{col}_Quantile'] = pd.cut(df[f'{col}_Percentile'], 
                                                                 bins=[0, 0.2, 0.4, 0.6, 0.8, 1],
                                                                 labels=[1, 2, 3, 4, 5], include_lowest=True)
                                
                                # Composite RFM score
                                df['RFM_Advanced'] = df[[f'{col}_Quantile' for col in rfm_present]].astype(float).mean(axis=1)
                                st.success("✅ Advanced RFM analysis completed!")
                        
                        # Time series features
                        if "Time Series Features" in feature_options:
                            progress_bar.progress(50)
                            # Create time-based features if date columns exist
                            date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                            for date_col in date_columns[:3]:  # Limit to first 3 date columns
                                try:
                                    df[f'{date_col}_parsed'] = pd.to_datetime(df[date_col], errors='coerce')
                                    df[f'{date_col}_day'] = df[f'{date_col}_parsed'].dt.day
                                    df[f'{date_col}_month'] = df[f'{date_col}_parsed'].dt.month
                                    df[f'{date_col}_year'] = df[f'{date_col}_parsed'].dt.year
                                    df[f'{date_col}_dayofweek'] = df[f'{date_col}_parsed'].dt.dayofweek
                                except:
                                    pass
                        
                        # Interaction features
                        if "Interaction Features" in feature_options and create_interactions:
                            progress_bar.progress(60)
                            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()[:5]
                            if len(numeric_columns) >= 2:
                                for i in range(len(numeric_columns)):
                                    for j in range(i+1, len(numeric_columns)):
                                        col1, col2_val = numeric_columns[i], numeric_columns[j]  # Changed variable name
                                        df[f'{col1}_x_{col2_val}'] = df[col1] * df[col2_val]
                                        df[f'{col1}_div_{col2_val}'] = df[col1] / (df[col2_val].replace(0, 1))
                        
                        # Polynomial features
                        if "Polynomial Features" in feature_options:
                            progress_bar.progress(70)
                            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()[:3]
                            if len(numeric_columns) >= 2:
                                # Ensure all values are finite
                                numeric_data = df[numeric_columns].fillna(0)
                                # Replace infinite values column by column
                                for col in numeric_columns:
                                    numeric_data[col] = numeric_data[col].replace([np.inf, -np.inf], 0)
                                
                                poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
                                poly_features = poly.fit_transform(numeric_data)
                                poly_df = pd.DataFrame(poly_features, 
                                                     columns=poly.get_feature_names_out(numeric_columns))
                                df = pd.concat([df, poly_df], axis=1)
                        
                        # Normalize features
                        if normalize_features:
                            progress_bar.progress(80)
                            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                            if len(numeric_columns) > 0:
                                # Process infinite values
                                for col in numeric_columns:
                                    if isinstance(col, str):
                                        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                                        df[col] = df[col].fillna(0)
                                
                                # Scale features
                                scaler = StandardScaler()
                                scaled_data = scaler.fit_transform(df[numeric_columns])
                                df[numeric_columns] = scaled_data
                        
                        progress_bar.progress(100)
                        st.session_state.df = df
                        st.session_state.features_extracted = True
                        st.success("🎉 All advanced features extracted successfully!")
        
        with right_col:  # Changed from col2 to right_col
            st.subheader("📊 Feature Summary")
            if st.session_state.features_extracted:
                # Show feature statistics
                numeric_features = [col for col in df.select_dtypes(include=[np.number]).columns.tolist() if isinstance(col, str)]
                categorical_features = [col for col in df.select_dtypes(include=['object']).columns.tolist() if isinstance(col, str)]
                
                st.metric("Total Features", len(df.columns))
                st.metric("Numeric Features", len(numeric_features))
                st.metric("Categorical Features", len(categorical_features))
                
                with st.expander("📋 Feature Details"):
                    # Show most important features (based on variance)
                    if len(numeric_features) > 0:
                        # Use try-except to handle any variance calculation issues
                        try:
                            variances = df[numeric_features].var().sort_values(ascending=False)
                            top_features = variances.head(10)
                            
                            st.write("**Top 10 Features by Variance:**")
                            for feature, variance in top_features.items():
                                st.write(f"• {feature}: {variance:.4f}")
                        except:
                            st.write("Could not calculate feature variances")
            
            # Feature visualization
            if st.session_state.features_extracted and len(numeric_features) > 0:
                st.subheader("📈 Feature Visualization")
                selected_feature = st.selectbox("Select feature:", numeric_features[:20])
                
                if selected_feature:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    # Histogram
                    try:
                        ax1.hist(df[selected_feature].dropna(), bins=30, edgecolor='black', alpha=0.7)
                        ax1.set_title(f'Distribution of {selected_feature}')
                        ax1.set_xlabel(selected_feature)
                        ax1.set_ylabel('Frequency')
                    except:
                        ax1.text(0.5, 0.5, f'Could not plot {selected_feature}', 
                                ha='center', va='center', transform=ax1.transAxes)
                    
                    # Box plot
                    try:
                        ax2.boxplot(df[selected_feature].dropna())
                        ax2.set_title(f'Box Plot of {selected_feature}')
                        ax2.set_ylabel(selected_feature)
                    except:
                        ax2.text(0.5, 0.5, f'Could not plot {selected_feature}', 
                                ha='center', va='center', transform=ax2.transAxes)
                    
                    plt.tight_layout()
                    st.pyplot(fig)