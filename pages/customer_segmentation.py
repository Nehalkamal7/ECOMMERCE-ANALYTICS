# pages/customer_segmentation.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score

def render():
    st.markdown('<h2 class="section-header">Advanced Customer Segmentation</h2>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("Please upload data in the Data Overview page first.")
    else:
        df = st.session_state.df.copy()  # Use copy to avoid modifying original
        
        # Advanced segmentation methods
        segmentation_method = st.selectbox(
            "Segmentation Method:",
            ["K-Means Clustering", "Gaussian Mixture Models", "Hierarchical Clustering",
             "DBSCAN", "Spectral Clustering"]
        )
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Default features - check which ones exist
        default_features = []
        for col in ['Age', 'Annual Income', 'Total_Spent', 'Total_Purchases', 'RFM_Sum', 'Engagement_Score']:
            if col in numeric_cols:
                default_features.append(col)
            if len(default_features) >= 6:
                break
        
        # Feature selection for clustering
        features = st.multiselect(
            "Select features for segmentation:",
            numeric_cols,
            default=default_features[:min(6, len(default_features))]
        )
        
        if len(features) >= 2:
            # Advanced options
            with st.expander("⚙️ Advanced Options"):
                n_clusters = st.slider("Number of clusters", 2, 10, 4)
                use_pca = st.checkbox("Use PCA for visualization", value=True)
                auto_clusters = st.checkbox("Auto-determine optimal clusters", value=False)
            
            if st.button("🔍 Perform Advanced Segmentation", type="primary"):
                with st.spinner("Performing segmentation..."):
                    # Handle missing values properly
                    X = df[features].copy()
                    
                    # Fill NaN values appropriately
                    for col in features:
                        if X[col].isna().any():
                            # Fill with median for numeric columns
                            X[col] = X[col].fillna(X[col].median())
                    
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Auto-determine clusters using elbow method
                    if auto_clusters:
                        try:
                            wcss = []
                            max_clusters = min(10, len(X_scaled) - 1)  # Ensure k <= n_samples
                            for k in range(1, max_clusters + 1):
                                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                                kmeans.fit(X_scaled)
                                wcss.append(kmeans.inertia_)
                            
                            fig, ax = plt.subplots(figsize=(8, 4))
                            ax.plot(range(1, max_clusters + 1), wcss, 'bo-')
                            ax.set_xlabel('Number of clusters')
                            ax.set_ylabel('WCSS')
                            ax.set_title('Elbow Method')
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                            
                            # Choose optimal k (simple heuristic)
                            if len(wcss) > 1:
                                diffs = np.diff(wcss)
                                optimal_k = np.argmin(diffs) + 2 if len(diffs) > 0 else 2
                                n_clusters = min(optimal_k, max_clusters)
                                st.info(f"Optimal clusters suggested: {optimal_k}")
                            else:
                                st.warning("Not enough data for elbow method")
                        except Exception as e:
                            st.warning(f"Could not compute optimal clusters: {e}")
                    
                    # Perform clustering
                    labels = None
                    try:
                        if segmentation_method == "K-Means Clustering":
                            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                            labels = clusterer.fit_predict(X_scaled)
                            
                        elif segmentation_method == "Gaussian Mixture Models":
                            clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
                            labels = clusterer.fit_predict(X_scaled)
                            
                        elif segmentation_method == "Hierarchical Clustering":
                            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
                            labels = clusterer.fit_predict(X_scaled)
                            
                        elif segmentation_method == "DBSCAN":
                            clusterer = DBSCAN(eps=0.5, min_samples=5)
                            labels = clusterer.fit_predict(X_scaled)
                            # DBSCAN may return -1 for outliers
                            if len(set(labels)) <= 1:
                                st.warning("DBSCAN found only 1 cluster or mostly noise. Try adjusting parameters.")
                                
                        elif segmentation_method == "Spectral Clustering":
                            from sklearn.cluster import SpectralClustering
                            clusterer = SpectralClustering(n_clusters=n_clusters, random_state=42)
                            labels = clusterer.fit_predict(X_scaled)
                    
                    except Exception as e:
                        st.error(f"Clustering failed: {e}")
                        st.stop()
                    
                    if labels is not None:
                        # Update dataframe with segment labels
                        df['Segment'] = labels
                        
                        # Calculate cluster metrics (only if we have multiple clusters)
                        unique_labels = set(labels)
                        if len(unique_labels) > 1:
                            try:
                                sil_score = silhouette_score(X_scaled, labels)
                                calinski = calinski_harabasz_score(X_scaled, labels)
                                
                                col_met1, col_met2 = st.columns(2)
                                with col_met1:
                                    st.metric("Silhouette Score", f"{sil_score:.3f}")
                                with col_met2:
                                    st.metric("Calinski-Harabasz", f"{calinski:.0f}")
                            except Exception as e:
                                st.warning(f"Could not calculate cluster metrics: {e}")
                        else:
                            st.warning("Only one cluster found. Metrics not available.")
                        
                        # Visualize clusters
                        if use_pca and len(unique_labels) > 1:
                            try:
                                pca = PCA(n_components=2)
                                X_pca = pca.fit_transform(X_scaled)
                                
                                fig, ax = plt.subplots(figsize=(10, 6))
                                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, 
                                                   cmap='tab10', alpha=0.6, s=50)
                                ax.set_xlabel('PCA Component 1')
                                ax.set_ylabel('PCA Component 2')
                                ax.set_title(f'{segmentation_method} - Customer Segments')
                                plt.colorbar(scatter, ax=ax, label='Segment')
                                ax.grid(True, alpha=0.3)
                                st.pyplot(fig)
                            except Exception as e:
                                st.warning(f"Could not create PCA visualization: {e}")
                        
                        # Segment profiles
                        st.subheader("📊 Segment Profiles")
                        
                        # Calculate segment statistics
                        try:
                            segment_stats = df.groupby('Segment')[features].agg(['mean', 'std', 'count']).round(3)
                            
                            # Display in tabs
                            tab1, tab2 = st.tabs(["Summary", "Detailed"])
                            
                            with tab1:
                                # Mean values heatmap
                                try:
                                    mean_df = segment_stats.xs('mean', axis=1, level=1)
                                    
                                    fig, ax = plt.subplots(figsize=(12, 6))
                                    sns.heatmap(mean_df.T, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
                                    ax.set_title('Segment Mean Values')
                                    st.pyplot(fig)
                                except Exception as e:
                                    st.warning(f"Could not create heatmap: {e}")
                                    st.dataframe(segment_stats.xs('mean', axis=1, level=1))
                            
                            with tab2:
                                # Detailed statistics
                                st.dataframe(segment_stats)
                                
                        except Exception as e:
                            st.warning(f"Could not calculate segment statistics: {e}")
                        
                        # Segment naming based on characteristics
                        st.subheader("🏷️ Segment Interpretation")
                        
                        # Auto-generate segment names based on characteristics
                        for seg in sorted(df['Segment'].unique()):
                            seg_data = df[df['Segment'] == seg]
                            
                            # Determine segment characteristics
                            spending = ""
                            frequency = ""
                            
                            # Check if feature exists and has data
                            if 'Total_Spent' in features and 'Total_Spent' in df.columns:
                                try:
                                    avg_spent = seg_data['Total_Spent'].mean()
                                    overall_quantiles = df['Total_Spent'].quantile([0.25, 0.75])
                                    
                                    if pd.notna(avg_spent):
                                        if avg_spent > overall_quantiles.iloc[1]:
                                            spending = "High Spenders"
                                        elif avg_spent > overall_quantiles.iloc[0]:
                                            spending = "Medium Spenders"
                                        else:
                                            spending = "Low Spenders"
                                except:
                                    spending = ""
                            
                            if 'Total_Purchases' in features and 'Total_Purchases' in df.columns:
                                try:
                                    avg_freq = seg_data['Total_Purchases'].mean()
                                    overall_quantiles = df['Total_Purchases'].quantile([0.25, 0.75])
                                    
                                    if pd.notna(avg_freq):
                                        if avg_freq > overall_quantiles.iloc[1]:
                                            frequency = "Frequent"
                                        elif avg_freq > overall_quantiles.iloc[0]:
                                            frequency = "Regular"
                                        else:
                                            frequency = "Occasional"
                                except:
                                    frequency = ""
                            
                            # Combine for segment name
                            if spending and frequency:
                                segment_name = f"{frequency} {spending}"
                            elif spending:
                                segment_name = f"{spending}"
                            elif frequency:
                                segment_name = f"{frequency} Buyers"
                            else:
                                segment_name = f"Segment {seg}"
                            
                            # Display segment info
                            col_seg1, col_seg2 = st.columns([3, 1])
                            with col_seg1:
                                st.markdown(f"**{segment_name}** (Segment {seg})")
                                st.write(f"Size: {len(seg_data)} customers ({len(seg_data)/len(df)*100:.1f}%)")
                            
                            with col_seg2:
                                if 'Total_Spent' in features and 'Total_Spent' in seg_data.columns:
                                    avg_spent_val = seg_data['Total_Spent'].mean()
                                    if pd.notna(avg_spent_val):
                                        st.metric("Avg Spend", f"${avg_spent_val:.0f}")