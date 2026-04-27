# pages/export_results.py
import streamlit as st
import pandas as pd
import io

def render():
    st.markdown('<h2 class="section-header">Export Results</h2>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("Please upload data first.")
    else:
        df = st.session_state.df
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Export Options")
            
            export_format = st.selectbox(
                "Export Format:",
                ["CSV", "Excel", "JSON"]
            )
            
            if 'Cluster' in df.columns or 'Segment_Name' in df.columns:
                include_segments = st.checkbox("Include segmentation results", value=True)
            
            if 'Is_Anomaly' in df.columns:
                include_anomalies = st.checkbox("Include anomaly detection", value=True)
            
            if st.button("📥 Download Results", type="primary", use_container_width=True):
                # Prepare data for export
                export_df = df.copy()
                
                # Convert datetime columns to string
                datetime_cols = export_df.select_dtypes(include=['datetime64']).columns
                for col in datetime_cols:
                    export_df[col] = export_df[col].astype(str)
                
                if export_format == "CSV":
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="ecommerce_analysis_results.csv",
                        mime="text/csv"
                    )
                
                elif export_format == "Excel":
                    # Requires openpyxl
                    try:
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            export_df.to_excel(writer, index=False, sheet_name='Analysis Results')
                            
                            # Add summary sheet if needed
                            if 'Cluster' in export_df.columns:
                                cluster_summary = export_df.groupby('Cluster').agg({
                                    'Total_Spent': 'mean',
                                    'Total_Purchases': 'mean',
                                    'Age': 'mean'
                                }).round(2)
                                cluster_summary.to_excel(writer, sheet_name='Cluster Summary')
                        
                        st.download_button(
                            label="Download Excel",
                            data=buffer.getvalue(),
                            file_name="ecommerce_analysis_results.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    except ImportError:
                        st.error("Please install openpyxl: pip install openpyxl")
                
                elif export_format == "JSON":
                    json_str = export_df.to_json(orient='records', indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name="ecommerce_analysis_results.json",
                        mime="application/json"
                    )
        
        with col2:
            st.subheader("Summary Report")
            
            if st.button("📋 Generate Summary Report", use_container_width=True):
                with st.expander("View Summary Report", expanded=True):
                    st.write("### Dataset Summary")
                    st.write(f"**Total Records:** {len(df)}")
                    st.write(f"**Total Features:** {len(df.columns)}")
                    
                    st.write("### Key Metrics")
                    
                    if 'Total_Spent' in df.columns:
                        st.write(f"**Total Revenue:** ${df['Total_Spent'].sum():,.2f}")
                        st.write(f"**Average Spend per Customer:** ${df['Total_Spent'].mean():,.2f}")
                    
                    if 'Total_Purchases' in df.columns:
                        st.write(f"**Total Purchases:** {df['Total_Purchases'].sum():,.0f}")
                        st.write(f"**Average Purchases per Customer:** {df['Total_Purchases'].mean():.2f}")
                    
                    if 'Made_Purchase' in df.columns:
                        purchase_rate = df['Made_Purchase'].mean() * 100
                        st.write(f"**Purchase Rate:** {purchase_rate:.1f}%")
                    
                    if 'Cluster' in df.columns:
                        st.write(f"**Number of Segments:** {df['Cluster'].nunique()}")
                    
                    if 'Is_Anomaly' in df.columns:
                        anomaly_rate = df['Is_Anomaly'].mean() * 100
                        st.write(f"**Anomaly Rate:** {anomaly_rate:.1f}%")