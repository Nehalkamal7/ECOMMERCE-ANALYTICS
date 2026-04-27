# pages/data_overview.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data

def render():
    st.markdown('<h2 class="section-header">Data Overview & Upload</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload E-commerce CSV", type=['csv'])
        
        if uploaded_file is not None:
            with st.spinner("Loading data..."):
                df = load_data(uploaded_file)
                st.session_state.df = df
                
            st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
            
            # Enhanced data analysis
            st.subheader("📈 Advanced Data Analysis")
            
            tab1, tab2, tab3 = st.tabs(["Basic Info", "Data Quality", "Advanced Stats"])
            
            with tab1:
                col_info1, col_info2, col_info3, col_info4 = st.columns(4)
                with col_info1:
                    st.metric("Total Rows", df.shape[0])
                with col_info2:
                    st.metric("Total Columns", df.shape[1])
                with col_info3:
                    numeric_cols = len(df.select_dtypes(include=['number']).columns)
                    st.metric("Numeric Columns", numeric_cols)
                with col_info4:
                    cat_cols = len(df.select_dtypes(include=['object']).columns)
                    st.metric("Categorical Columns", cat_cols)
            
            with tab2:
                # Missing values analysis
                missing_values = df.isnull().sum().sort_values(ascending=False)
                missing_percent = (missing_values / len(df) * 100).round(2)
                missing_df = pd.DataFrame({
                    'Missing Count': missing_values,
                    'Missing %': missing_percent
                })
                
                col_miss1, col_miss2 = st.columns([2, 1])
                with col_miss1:
                    st.dataframe(missing_df[missing_df['Missing Count'] > 0])
                with col_miss2:
                    total_missing = missing_values.sum()
                    st.metric("Total Missing Values", total_missing)
            
            with tab3:
                # Data distribution visualization
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    selected_col = st.selectbox("Select column for distribution:", numeric_cols[:10])
                    
                    col_dist1, col_dist2 = st.columns(2)
                    with col_dist1:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.hist(df[selected_col].dropna(), bins=30, edgecolor='black', alpha=0.7)
                        ax.set_title(f'Distribution of {selected_col}')
                        ax.set_xlabel(selected_col)
                        ax.set_ylabel('Frequency')
                        st.pyplot(fig)
                    
                    with col_dist2:
                        stats = df[selected_col].describe()
                        stats_df = pd.DataFrame({
                            'Statistic': stats.index,
                            'Value': stats.values
                        })
                        st.dataframe(stats_df)
            
            # Data preview with more options
            st.subheader("🔍 Data Preview")
            preview_tab1, preview_tab2, preview_tab3 = st.tabs(["Head", "Tail", "Sample"])
            
            with preview_tab1:
                st.dataframe(df.head(10), use_container_width=True)
            with preview_tab2:
                st.dataframe(df.tail(10), use_container_width=True)
            with preview_tab3:
                sample_size = st.slider("Sample size", 5, 50, 10)
                st.dataframe(df.sample(sample_size), use_container_width=True)
    
    with col2:
        if st.session_state.df is not None:
            st.subheader("⚡ Quick Actions")
            
            if st.button("📊 Generate Full Report", use_container_width=True):
                with st.expander("Comprehensive Report", expanded=True):
                    # Column information
                    col_types = pd.DataFrame({
                        'Column': df.columns,
                        'Data Type': df.dtypes.values,
                        'Non-Null': df.notnull().sum().values,
                        'Null': df.isnull().sum().values,
                        'Unique': df.nunique().values
                    })
                    st.dataframe(col_types, use_container_width=True)
            
            if st.button("🔍 Detect Data Issues", use_container_width=True):
                issues = []
                
                # Check for duplicate rows
                duplicates = df.duplicated().sum()
                if duplicates > 0:
                    issues.append(f"⚠️ {duplicates} duplicate rows found")
                
                # Check for constant columns
                for col in df.columns:
                    if df[col].nunique() == 1:
                        issues.append(f"⚠️ Constant column: {col}")
                
                # Check for high cardinality
                for col in df.select_dtypes(include=['object']).columns:
                    if df[col].nunique() > 100:
                        issues.append(f"⚠️ High cardinality: {col} ({df[col].nunique()} unique values)")
                
                if issues:
                    st.warning("Data Issues Detected:")
                    for issue in issues:
                        st.write(issue)
                else:
                    st.success("✅ No major data issues detected")
            
            if st.button("📈 Visualize Structure", use_container_width=True):
                # Create correlation matrix for numeric columns
                numeric_df = df.select_dtypes(include=['number'])
                if len(numeric_df.columns) > 1:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    corr = numeric_df.corr()
                    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                               center=0, square=True, ax=ax)
                    ax.set_title('Correlation Matrix')
                    st.pyplot(fig)