# pages/neural_networks.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, mean_squared_error, r2_score, confusion_matrix)
from models.neural_networks import create_simple_nn, create_deep_nn, create_lstm_model
import warnings
warnings.filterwarnings('ignore')

# def render():
#     st.markdown('<h2 class="section-header">Neural Network Models</h2>', unsafe_allow_html=True)
    
#     # First check if data is loaded
#     if st.session_state.df is None:
#         st.warning("Please upload data in the Data Overview page first.")
#         st.stop()
    
#     df = st.session_state.df
    
#     # Then check if ANY deep learning library is available
#     if not st.session_state.get('DL_AVAILABLE', False):
#         st.error("""
#         ⚠️ TensorFlow/Keras is not installed. Neural network features are disabled.
        
#         To enable neural networks, install either:
        
#         **TensorFlow (recommended):**
#         ```bash
#         pip install tensorflow
#         ```
        
#         For CPU-only version:
#         ```bash
#         pip install tensorflow-cpu
#         ```
        
#         **Or standalone Keras:**
#         ```bash
#         pip install keras
#         ```
#         """)

#         if st.button("Install TensorFlow (CPU)"):
#             st.code("pip install tensorflow-cpu", language="bash")
#             st.info("Run this command in your terminal, then restart the app.")
        
#         # Show what would be available with TensorFlow
#         st.info("""
#         **With TensorFlow installed, you could:**
#         - Train feedforward neural networks
#         - Build deep neural networks with multiple layers
#         - Create LSTM models for sequential data
#         - Customize architecture, activation functions, and training parameters
#         - Visualize training history and model performance
#         """)
        
#         # Show a sample of what features would be available
#         with st.expander("📋 Sample Neural Network Configuration"):
#             st.write("**Model Type:** Feedforward Neural Network")
#             st.write("**Task:** Classification")
#             st.write("**Architecture:** Input → Dense(64) → Dropout → Dense(32) → Output")
#             st.write("**Training:** 50 epochs, batch size 32, validation split 0.2")
        
#         st.stop()  # Stop execution here
    

# pages/neural_networks.py
def render():
    st.markdown('<h2 class="section-header">Neural Network Models</h2>', unsafe_allow_html=True)
    
    # First check if data is loaded
    if st.session_state.df is None:
        st.warning("Please upload data in the Data Overview page first.")
        st.stop()
    
    df = st.session_state.df
    
    # Then check if ANY deep learning library is available
    # Get from session state or default to False
    DL_AVAILABLE = st.session_state.get('DL_AVAILABLE', False)
    
    if not DL_AVAILABLE:
        # Try to detect TensorFlow/Keras directly
        try:
            import tensorflow as tf
            st.session_state['DL_AVAILABLE'] = True
            st.session_state['TF_AVAILABLE'] = True
            DL_AVAILABLE = True
        except ImportError:
            try:
                import keras
                st.session_state['DL_AVAILABLE'] = True
                st.session_state['KERAS_AVAILABLE'] = True
                DL_AVAILABLE = True
            except ImportError:
                pass
    
    if not DL_AVAILABLE:
        st.error("""
        ⚠️ TensorFlow/Keras is not installed. Neural network features are disabled.
        
        To enable neural networks, install either:
        
        **TensorFlow (recommended):**
        ```bash
        pip install tensorflow
        ```
        
        For CPU-only version:
        ```bash
        pip install tensorflow-cpu
        ```
        
        **Or standalone Keras:**
        ```bash
        pip install keras
        ```
        """)

        # Check if TensorFlow is actually installed
        st.write("**Debug Info:**")
        st.code("""
        # To check what's installed:
        pip list | findstr tensorflow
        pip list | findstr keras
        """)
        
        if st.button("Test TensorFlow Installation"):
            try:
                import tensorflow as tf
                st.success(f"✅ TensorFlow {tf.__version__} is installed!")
                st.info("The app needs to be restarted to detect TensorFlow.")
            except ImportError as e:
                st.error(f"TensorFlow import failed: {e}")
        
        st.stop()
    
    # Normal flow when TensorFlow IS available
    st.info("Train and evaluate neural network models for various prediction tasks.")
    
    # Model selection
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.subheader("⚙️ Model Configuration")
        
        nn_type = st.selectbox(
            "Neural Network Type:",
            ["Feedforward NN", "Deep NN"]
        )
        
        task_type = st.selectbox(
            "Task Type:",
            ["Regression"]
        )
        
        if task_type == "Classification":
            target_col = st.selectbox(
                "Target Variable:",
                [col for col in df.columns if df[col].nunique() <= 10]
            )
        else:
            target_col = st.selectbox(
                "Target Variable:",
                df.select_dtypes(include=[np.number]).columns.tolist()
            )
    
    with col2:
        st.subheader("📊 Architecture Settings")
        
        if nn_type in ["Feedforward NN", "Deep NN"]:
            hidden_layers = st.multiselect(
                "Hidden Layer Sizes:",
                [16, 32, 64, 128, 256, 512],
                default=[64, 32, 16] if nn_type == "Deep NN" else [64, 32]
            )
            
            activation = st.selectbox(
                "Activation Function:",
                ["relu", "tanh", "sigmoid", "elu"]
            )
            
            dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
            learning_rate = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, 0.0001, format="%.4f")
        
        elif nn_type == "LSTM (Sequential)":
            lstm_units = st.multiselect(
                "LSTM Units:",
                [32, 64, 128, 256],
                default=[64, 32]
            )
            sequence_length = st.slider("Sequence Length", 5, 50, 10)
        
        # Training settings
        st.subheader("🎯 Training Settings")
        
        epochs = st.slider("Epochs", 10, 200, 50)
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
        validation_split = st.slider("Validation Split", 0.1, 0.4, 0.2, 0.05)
    
    with col3:
        st.subheader("📈 Feature Selection")
        
        feature_cols = st.multiselect(
            "Select Features:",
            df.select_dtypes(include=[np.number]).columns.tolist(),
            default=df.select_dtypes(include=[np.number]).columns.tolist()[:5]
        )
    
    # Train button
    if st.button("🚀 Train Neural Network", type="primary", use_container_width=True):
        if len(feature_cols) < 1:
            st.error("Please select at least one feature.")
        elif target_col not in df.columns:
            st.error("Target column not found in dataset.")
        else:
            with st.spinner("Training neural network..."):
                # Prepare data
                X = df[feature_cols].fillna(0).values
                y = df[target_col].values
                
                # For classification, encode labels
                if task_type == "Classification":
                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y)
                    output_dim = len(np.unique(y_encoded))
                    
                    # For binary classification
                    if output_dim == 1:
                        y = y_encoded.reshape(-1, 1)  # Ensure 2D shape
                    elif output_dim == 2:
                        # Binary classification
                        y = y_encoded.reshape(-1, 1)
                    else:
                        # Multi-class classification: one-hot encode
                        try:
                            from tensorflow.keras.utils import to_categorical
                            y = to_categorical(y_encoded)
                        except ImportError:
                            from keras.utils import to_categorical
                            y = to_categorical(y_encoded)
                else:
                    output_dim = 1
                    y = y.reshape(-1, 1)  # Ensure 2D for regression
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42,
                    stratify=y_encoded if task_type == "Classification" else None
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Create and train model
                if nn_type == "Feedforward NN":
                    model = create_simple_nn(
                        input_dim=X_train_scaled.shape[1],
                        output_dim=output_dim,
                        task=task_type.lower()
                    )
                elif nn_type == "Deep NN":
                    model = create_deep_nn(
                        input_dim=X_train_scaled.shape[1],
                        output_dim=output_dim,
                        task=task_type.lower()
                    )
                elif nn_type == "LSTM (Sequential)":
                    # Reshape for LSTM
                    X_train_reshaped = X_train_scaled.reshape(-1, sequence_length, X_train_scaled.shape[1] // sequence_length)
                    X_test_reshaped = X_test_scaled.reshape(-1, sequence_length, X_test_scaled.shape[1] // sequence_length)
                    
                    model = create_lstm_model(
                        input_shape=X_train_reshaped.shape[2],
                        sequence_length=sequence_length
                    )
                
                if model:
                    # Get callbacks
                    try:
                        from tensorflow.keras import callbacks as tf_callbacks
                        callbacks_module = tf_callbacks
                    except ImportError:
                        from keras import callbacks as keras_callbacks
                        callbacks_module = keras_callbacks
                    
                    early_stopping = callbacks_module.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    )
                    
                    reduce_lr = callbacks_module.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=5,
                        min_lr=0.00001
                    )
                    
                    # Train model
                    history = model.fit(
                        X_train_scaled if nn_type != "LSTM (Sequential)" else X_train_reshaped,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=validation_split,
                        callbacks=[early_stopping, reduce_lr],
                        verbose=0
                    )
                    
                    # Evaluate model
                    if nn_type != "LSTM (Sequential)":
                        y_pred = model.predict(X_test_scaled)
                        X_eval = X_test_scaled
                    else:
                        y_pred = model.predict(X_test_reshaped)
                        X_eval = X_test_reshaped
                    
                    # Display results
                    st.subheader("📊 Training Results")
                    
                    # Plot training history
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    ax1.plot(history.history['loss'], label='Training Loss')
                    ax1.plot(history.history['val_loss'], label='Validation Loss')
                    ax1.set_title('Model Loss')
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Loss')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    if 'accuracy' in history.history:
                        ax2.plot(history.history['accuracy'], label='Training Accuracy')
                        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
                        ax2.set_title('Model Accuracy')
                        ax2.set_xlabel('Epoch')
                        ax2.set_ylabel('Accuracy')
                    else:
                        ax2.plot(history.history['mae'], label='Training MAE')
                        ax2.plot(history.history['val_mae'], label='Validation MAE')
                        ax2.set_title('Model MAE')
                        ax2.set_xlabel('Epoch')
                        ax2.set_ylabel('MAE')
                    
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Metrics
                    if task_type == "Classification":
                        y_pred_class = (y_pred > 0.5).astype(int) if output_dim == 1 else np.argmax(y_pred, axis=1)
                        
                        col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                        with col_met1:
                            accuracy = accuracy_score(y_test, y_pred_class)
                            st.metric("Accuracy", f"{accuracy:.4f}")
                        with col_met2:
                            precision = precision_score(y_test, y_pred_class, average='weighted', zero_division=0)
                            st.metric("Precision", f"{precision:.4f}")
                        with col_met3:
                            recall = recall_score(y_test, y_pred_class, average='weighted', zero_division=0)
                            st.metric("Recall", f"{recall:.4f}")
                        with col_met4:
                            f1 = f1_score(y_test, y_pred_class, average='weighted', zero_division=0)
                            st.metric("F1-Score", f"{f1:.4f}")
                        
                        # Confusion matrix
                        if output_dim <= 10:  # Limit for visualization
                            cm = confusion_matrix(y_test, y_pred_class)
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                            ax.set_xlabel('Predicted')
                            ax.set_ylabel('Actual')
                            ax.set_title('Confusion Matrix')
                            st.pyplot(fig)
                    
                    else:  # Regression
                        col_met1, col_met2, col_met3 = st.columns(3)
                        with col_met1:
                            mse = mean_squared_error(y_test, y_pred)
                            st.metric("MSE", f"{mse:.4f}")
                        with col_met2:
                            rmse = np.sqrt(mse)
                            st.metric("RMSE", f"{rmse:.4f}")
                        with col_met3:
                            r2 = r2_score(y_test, y_pred)
                            st.metric("R² Score", f"{r2:.4f}")
                        
                        # Scatter plot
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.scatter(y_test, y_pred, alpha=0.5)
                        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                        ax.set_xlabel('Actual')
                        ax.set_ylabel('Predicted')
                        ax.set_title('Actual vs Predicted')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    
                    # Save model to session state
                    st.session_state.models_trained['neural_network'] = {
                        'model': model,
                        'history': history.history,
                        'scaler': scaler,
                        'features': feature_cols,
                        'target': target_col,
                        'type': nn_type
                    }
                    
                    st.success("✅ Neural network trained successfully!")