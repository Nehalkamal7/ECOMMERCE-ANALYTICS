# pages/insights_patterns.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

def render():
    st.markdown('<h2 class="section-header">Insights & Patterns</h2>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("Please upload data in the Data Overview page first.")
    else:
        df = st.session_state.df.copy()  # Use copy to avoid modifying original
        
        insight_type = st.selectbox(
            "Select Insight Type:",
            ["Correlation Analysis", "Trend Analysis", "Pattern Detection", 
             "Anomaly Detection", "Demographic Insights"]
        )
        
        if insight_type == "Correlation Analysis":
            st.subheader("Feature Correlations")
            
            numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns.tolist() 
                          if isinstance(col, str)]
            
            if len(numeric_cols) > 1:
                selected_cols = st.multiselect(
                    "Select columns for correlation:",
                    numeric_cols,
                    default=numeric_cols[:min(8, len(numeric_cols))]
                )
                
                if len(selected_cols) >= 2:
                    # Filter out columns with all NaN or constant values
                    valid_cols = []
                    for col in selected_cols:
                        if col in df.columns:
                            col_data = df[col]
                            # Check if column has variance and not all NaN
                            if col_data.notna().any() and col_data.nunique() > 1:
                                valid_cols.append(col)
                    
                    if len(valid_cols) >= 2:
                        try:
                            # Fill NaN with column mean for correlation calculation
                            corr_data = df[valid_cols].copy()
                            for col in valid_cols:
                                if corr_data[col].isna().any():
                                    corr_data[col] = corr_data[col].fillna(corr_data[col].mean())
                            
                            corr_matrix = corr_data.corr()
                            
                            fig, ax = plt.subplots(figsize=(10, 8))
                            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                                       center=0, square=True, ax=ax, vmin=-1, vmax=1)
                            ax.set_title('Correlation Heatmap')
                            st.pyplot(fig)
                            
                            # Strong correlations
                            strong_corrs = []
                            for i in range(len(valid_cols)):
                                for j in range(i+1, len(valid_cols)):
                                    try:
                                        corr = corr_matrix.iloc[i, j]
                                        if abs(corr) > 0.7:
                                            strong_corrs.append({
                                                'Feature 1': valid_cols[i],
                                                'Feature 2': valid_cols[j],
                                                'Correlation': f"{corr:.3f}"
                                            })
                                    except:
                                        continue
                            
                            if strong_corrs:
                                st.subheader("Strong Correlations (|r| > 0.7)")
                                st.dataframe(pd.DataFrame(strong_corrs))
                            else:
                                st.info("No strong correlations found (|r| ≤ 0.7)")
                                
                        except Exception as e:
                            st.error(f"Could not calculate correlations: {e}")
                    else:
                        st.warning("Need at least 2 valid numeric columns for correlation analysis")
        
        elif insight_type == "Trend Analysis":
            st.subheader("Trend Analysis")
            
            # Create columns for layout
            col1, col2 = st.columns(2)
            
            with col1:
                # Purchases vs Spending analysis
                if 'Total_Spent' in df.columns and 'Total_Purchases' in df.columns:
                    # Filter out NaN values
                    trend_data = df[['Total_Purchases', 'Total_Spent']].dropna()
                    
                    if len(trend_data) > 1:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.scatter(trend_data['Total_Purchases'], trend_data['Total_Spent'], alpha=0.5)
                        ax.set_xlabel('Total Purchases')
                        ax.set_ylabel('Total Spent')
                        ax.set_title('Purchases vs Spending')
                        ax.grid(True, alpha=0.3)
                        
                        # Add trend line only if we have enough data points
                        if len(trend_data) > 2:
                            try:
                                x_vals = trend_data['Total_Purchases'].values
                                y_vals = trend_data['Total_Spent'].values
                                z = np.polyfit(x_vals, y_vals, 1)
                                p = np.poly1d(z)
                                ax.plot(x_vals, p(x_vals), "r--", alpha=0.8, label=f"Trend: y = {z[0]:.2f}x + {z[1]:.2f}")
                                ax.legend()
                            except:
                                pass  # Skip trend line if polyfit fails
                        
                        st.pyplot(fig)
                    else:
                        st.warning("Not enough data for purchases vs spending analysis")
                else:
                    st.info("'Total_Spent' and 'Total_Purchases' columns not found")
            
            with col2:
                # Age vs Income analysis
                age_col = None
                income_col = None
                
                # Find age and income columns (case-insensitive)
                for col in df.columns:
                    if isinstance(col, str):
                        col_lower = col.lower()
                        if 'age' in col_lower and age_col is None:
                            age_col = col
                        elif ('income' in col_lower or 'salary' in col_lower or 'annual' in col_lower) and income_col is None:
                            income_col = col
                
                if age_col and income_col:
                    # Filter out NaN values
                    trend_data = df[[age_col, income_col]].dropna()
                    
                    if len(trend_data) > 1:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.scatter(trend_data[age_col], trend_data[income_col], alpha=0.5)
                        ax.set_xlabel(age_col)
                        ax.set_ylabel(income_col)
                        ax.set_title(f'{age_col} vs {income_col}')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    else:
                        st.warning(f"Not enough data for {age_col} vs {income_col} analysis")
                else:
                    st.info("Age and income columns not found for analysis")
        
        elif insight_type == "Pattern Detection":
            st.subheader("Pattern Detection")
            
            # Pattern detection options
            pattern_option = st.selectbox(
                "Select Pattern Type:",
                ["Purchase Patterns", "Customer Behavior Patterns"]
            )
            
            if pattern_option == "Purchase Patterns":
                if 'Total_Spent' in df.columns and 'Total_Purchases' in df.columns:
                    # Create purchase frequency categories
                    df['Purchase_Frequency_Category'] = pd.cut(
                        df['Total_Purchases'],
                        bins=[0, 1, 3, 10, float('inf')],
                        labels=['One-time', 'Occasional', 'Regular', 'Frequent']
                    )
                    
                    # Analyze spending by frequency
                    freq_stats = df.groupby('Purchase_Frequency_Category')['Total_Spent'].agg(['mean', 'count']).round(2)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        freq_stats['mean'].plot(kind='bar', ax=ax, color='skyblue')
                        ax.set_xlabel('Purchase Frequency')
                        ax.set_ylabel('Average Total Spent')
                        ax.set_title('Average Spending by Purchase Frequency')
                        ax.tick_params(axis='x', rotation=45)
                        st.pyplot(fig)
                    
                    with col2:
                        st.write("**Purchase Frequency Statistics:**")
                        st.dataframe(freq_stats)
            
            elif pattern_option == "Customer Behavior Patterns":
                # Check for behavioral columns
                behavior_cols = [col for col in df.columns if any(word in str(col).lower() 
                                 for word in ['frequency', 'regularity', 'engagement', 'activity', 'session'])]
                
                if behavior_cols:
                    selected_behavior = st.selectbox("Select behavior metric:", behavior_cols[:5])
                    
                    if selected_behavior in df.columns:
                        # Create histogram of behavior patterns
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.hist(df[selected_behavior].dropna(), bins=30, alpha=0.7, edgecolor='black')
                        ax.set_xlabel(selected_behavior)
                        ax.set_ylabel('Frequency')
                        ax.set_title(f'Distribution of {selected_behavior}')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
        
        elif insight_type == "Anomaly Detection":
            st.subheader("Anomaly Detection")
            
            numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns.tolist() 
                          if isinstance(col, str) and df[col].nunique() > 1]
            
            if len(numeric_cols) >= 2:
                selected_features = st.multiselect(
                    "Select features for anomaly detection:",
                    numeric_cols,
                    default=['Total_Spent', 'Total_Purchases'][:min(2, len(numeric_cols))]
                )
                
                if len(selected_features) >= 2:
                    # Ensure selected features exist
                    selected_features = [col for col in selected_features if col in df.columns]
                    
                    if len(selected_features) >= 2:
                        if st.button("Detect Anomalies"):
                            with st.spinner("Detecting anomalies..."):
                                try:
                                    # Prepare data
                                    X = df[selected_features].copy()
                                    
                                    # Handle missing values
                                    for col in selected_features:
                                        if X[col].isna().any():
                                            X[col] = X[col].fillna(X[col].median())
                                    
                                    scaler = StandardScaler()
                                    X_scaled = scaler.fit_transform(X)
                                    
                                    # Use Isolation Forest
                                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                                    anomalies = iso_forest.fit_predict(X_scaled)
                                    df['Is_Anomaly'] = (anomalies == -1).astype(int)
                                    
                                    n_anomalies = df['Is_Anomaly'].sum()
                                    st.success(f"✅ Detected {n_anomalies} anomalies ({n_anomalies/len(df)*100:.1f}% of data)")
                                    
                                    # Visualize anomalies
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                                                       c=df['Is_Anomaly'], cmap='coolwarm',
                                                       alpha=0.6, s=30)
                                    ax.set_xlabel(selected_features[0])
                                    ax.set_ylabel(selected_features[1])
                                    ax.set_title('Anomaly Detection Results')
                                    ax.grid(True, alpha=0.3)
                                    plt.colorbar(scatter, ax=ax, label='Is Anomaly (1=Yes)')
                                    st.pyplot(fig)
                                    
                                    # Show anomaly details
                                    if n_anomalies > 0:
                                        with st.expander("View Anomalous Records"):
                                            anomalies_df = df[df['Is_Anomaly'] == 1][selected_features + ['Is_Anomaly']]
                                            st.dataframe(anomalies_df.head(20))
                                    else:
                                        st.info("No anomalies detected with current settings")
                                        
                                except Exception as e:
                                    st.error(f"Anomaly detection failed: {e}")
                    else:
                        st.warning("Please select at least 2 valid features")
                else:
                    st.warning("Please select at least 2 features")
            else:
                st.warning("Need at least 2 numeric columns with variance for anomaly detection")
        
        elif insight_type == "Demographic Insights":
            st.subheader("Demographic Insights")
            
            # Find demographic columns
            demo_cols = []
            for col in df.columns:
                if isinstance(col, str):
                    col_lower = col.lower()
                    if any(word in col_lower for word in ['age', 'gender', 'income', 'location', 'city', 'country', 'region']):
                        demo_cols.append(col)
            
            if demo_cols:
                selected_demo = st.selectbox("Select demographic feature:", demo_cols[:10])
                
                if selected_demo in df.columns:
                    # Check if it's categorical or numeric
                    if df[selected_demo].dtype == 'object' or df[selected_demo].nunique() < 10:
                        # Categorical - show distribution
                        demo_counts = df[selected_demo].value_counts().head(10)
                        
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                        
                        # Bar chart
                        demo_counts.plot(kind='bar', ax=ax1, color='lightblue')
                        ax1.set_xlabel(selected_demo)
                        ax1.set_ylabel('Count')
                        ax1.set_title(f'Distribution of {selected_demo}')
                        ax1.tick_params(axis='x', rotation=45)
                        
                        # Pie chart (top categories only)
                        top_categories = demo_counts.head(5)
                        ax2.pie(top_categories.values, labels=top_categories.index, autopct='%1.1f%%')
                        ax2.set_title(f'Top 5 {selected_demo} Categories')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        # Numeric - show histogram
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.hist(df[selected_demo].dropna(), bins=30, alpha=0.7, edgecolor='black')
                        ax.set_xlabel(selected_demo)
                        ax.set_ylabel('Frequency')
                        ax.set_title(f'Distribution of {selected_demo}')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
            else:
                st.info("No demographic columns found in the dataset")