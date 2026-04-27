# pages/pattern_recognition.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score, calinski_harabasz_score

def render():
    st.markdown('<h2 class="section-header">Advanced Pattern Recognition</h2>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("Please upload data in the Data Overview page first.")
    else:
        df = st.session_state.df
        
        st.info("Discover hidden patterns and relationships in your data.")
        
        # Pattern recognition methods
        pattern_method = st.selectbox(
            "Select Pattern Recognition Method:",
            ["Dimensionality Reduction", "Cluster Analysis", 
             "Anomaly Detection"]
        )
        
        if pattern_method == "Dimensionality Reduction":
            st.subheader("📊 Dimensionality Reduction")
            
            # Select features
            features = st.multiselect(
                "Select features for dimensionality reduction:",
                df.select_dtypes(include=[np.number]).columns.tolist(),
                default=df.select_dtypes(include=[np.number]).columns.tolist()[:8]
            )
            
            if len(features) >= 2:
                # Method selection
                dr_method = st.radio(
                    "Dimensionality Reduction Method:",
                    ["PCA", "t-SNE"]
                )
                
                if st.button("🔍 Apply Dimensionality Reduction", type="primary"):
                    with st.spinner("Analyzing patterns..."):
                        X = df[features].fillna(0)
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        
                        if dr_method == "PCA":
                            # PCA with variance explained
                            pca = PCA()
                            X_reduced = pca.fit_transform(X_scaled)
                            
                            # Plot explained variance
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                            
                            ax1.plot(range(1, len(pca.explained_variance_ratio_) + 1),
                                   np.cumsum(pca.explained_variance_ratio_), 'bo-')
                            ax1.axhline(y=0.95, color='r', linestyle='--', alpha=0.5)
                            ax1.set_xlabel('Number of Components')
                            ax1.set_ylabel('Cumulative Explained Variance')
                            ax1.set_title('PCA: Explained Variance')
                            ax1.grid(True, alpha=0.3)
                            
                            # Scatter plot of first two components
                            ax2.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.5)
                            ax2.set_xlabel('Principal Component 1')
                            ax2.set_ylabel('Principal Component 2')
                            ax2.set_title('PCA: First Two Components')
                            ax2.grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Show component loadings
                            st.subheader("📋 Component Loadings")
                            loadings = pd.DataFrame(
                                pca.components_[:3].T,
                                columns=[f'PC{i+1}' for i in range(3)],
                                index=features
                            )
                            st.dataframe(loadings.style.background_gradient(cmap='coolwarm', axis=0))
                        
                        elif dr_method == "t-SNE":
                            # t-SNE visualization
                            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
                            X_tsne = tsne.fit_transform(X_scaled)
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6, s=30)
                            ax.set_xlabel('t-SNE Component 1')
                            ax.set_ylabel('t-SNE Component 2')
                            ax.set_title('t-SNE Visualization')
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
        
        elif pattern_method == "Cluster Analysis":
            st.subheader("👥 Advanced Cluster Analysis")
            
            # Select features
            features = st.multiselect(
                "Select features for clustering:",
                df.select_dtypes(include=[np.number]).columns.tolist(),
                default=df.select_dtypes(include=[np.number]).columns.tolist()[:6]
            )
            
            if len(features) >= 2:
                # Clustering method
                cluster_method = st.selectbox(
                    "Clustering Algorithm:",
                    ["K-Means"]
                )
                
                if cluster_method == "K-Means":
                    n_clusters = st.slider("Number of clusters", 2, 10, 4)
                
                if st.button("🔍 Perform Cluster Analysis", type="primary"):
                    with st.spinner("Clustering data..."):
                        X = df[features].fillna(0)
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        
                        # Perform clustering
                        if cluster_method == "K-Means":
                            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                            labels = kmeans.fit_predict(X_scaled)
                            df['Cluster'] = labels
                        
                        # Calculate metrics
                        if len(set(labels)) > 1:
                            sil_score = silhouette_score(X_scaled, labels)
                            calinski = calinski_harabasz_score(X_scaled, labels)
                            
                            col_met1, col_met2 = st.columns(2)
                            with col_met1:
                                st.metric("Silhouette Score", f"{sil_score:.3f}")
                            with col_met2:
                                st.metric("Calinski-Harabasz", f"{calinski:.0f}")
                        
                        # Visualize clusters
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X_scaled)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, 
                                           cmap='tab10', alpha=0.6, s=50)
                        ax.set_xlabel('PCA Component 1')
                        ax.set_ylabel('PCA Component 2')
                        ax.set_title(f'{cluster_method} Clusters')
                        plt.colorbar(scatter, ax=ax, label='Cluster')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        
                        # Cluster statistics
                        st.subheader("📊 Cluster Statistics")
                        cluster_stats = df.groupby('Cluster')[features].mean().round(3)
                        st.dataframe(cluster_stats)
        
        elif pattern_method == "Anomaly Detection":
            st.subheader("🚨 Advanced Anomaly Detection")
            
            # Select features
            features = st.multiselect(
                "Select features for anomaly detection:",
                df.select_dtypes(include=[np.number]).columns.tolist(),
                default=df.select_dtypes(include=[np.number]).columns.tolist()[:5]
            )
            
            if len(features) >= 2:
                # Anomaly detection method
                anomaly_method = st.selectbox(
                    "Anomaly Detection Algorithm:",
                    ["Isolation Forest", "One-Class SVM", "Elliptic Envelope"]
                )
                
                contamination = st.slider("Expected contamination rate", 0.01, 0.5, 0.1, 0.01)
                
                if st.button("🔍 Detect Anomalies", type="primary"):
                    with st.spinner("Detecting anomalies..."):
                        X = df[features].fillna(0)
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        
                        # Detect anomalies
                        if anomaly_method == "Isolation Forest":
                            detector = IsolationForest(contamination=contamination, random_state=42)
                        elif anomaly_method == "Local Outlier Factor":
                            detector = LocalOutlierFactor(contamination=contamination, novelty=True)
                        elif anomaly_method == "One-Class SVM":
                            detector = OneClassSVM(nu=contamination)
                        else:  # Elliptic Envelope
                            detector = EllipticEnvelope(contamination=contamination, random_state=42)
                        
                        anomalies = detector.fit_predict(X_scaled)
                        df['Is_Anomaly'] = (anomalies == -1).astype(int)
                        
                        # Results
                        n_anomalies = df['Is_Anomaly'].sum()
                        st.metric("Anomalies Detected", n_anomalies)
                        
                        # Visualize anomalies
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X_scaled)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                                           c=df['Is_Anomaly'], cmap='coolwarm',
                                           alpha=0.6, s=30)
                        ax.set_xlabel('PCA Component 1')
                        ax.set_ylabel('PCA Component 2')
                        ax.set_title(f'{anomaly_method}: Anomaly Detection')
                        plt.colorbar(scatter, ax=ax, label='Is Anomaly (1=Yes)')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        
                        # Show anomalies
                        if n_anomalies > 0:
                            with st.expander("📋 View Anomalous Records"):
                                anomalies_df = df[df['Is_Anomaly'] == 1][features + ['Is_Anomaly']]
                                st.dataframe(anomalies_df.head(20))