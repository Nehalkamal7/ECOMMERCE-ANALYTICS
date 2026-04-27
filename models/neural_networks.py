# models/neural_networks.py
import streamlit as st

def create_simple_nn(input_dim, output_dim=1, task='classification'):
    """Create a simple neural network"""
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        try:
            import keras
            from keras import layers
        except ImportError:
            return None
    
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
    ])
    
    if task == 'classification':
        if output_dim == 1:
            # Binary classification
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_cross-entropy',
                         metrics=['accuracy', keras.metrics.AUC()])
        elif output_dim == 2:
            # Also binary classification (2 classes)
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_cross-entropy',
                         metrics=['accuracy', keras.metrics.AUC()])
        else:
            # Multi-class classification
            model.add(layers.Dense(output_dim, activation='softmax'))
            model.compile(optimizer='adam', loss='categorical_cross-entropy',
                         metrics=['accuracy'])
    else:  # regression
        model.add(layers.Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
    
    return model

def create_deep_nn(input_dim, output_dim=1, task='classification'):
    """Create a deep neural network with batch normalization"""
    try:
        from tensorflow import keras
        from tensorflow.keras import layers, regularizers
        from tensorflow.keras.optimizers import Adam
    except ImportError:
        try:
            import keras
            from keras import layers, regularizers
            from keras.optimizers import Adam
        except ImportError:
            return None
    
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
    ])
    
    if task == 'classification':
        if output_dim == 1 or output_dim == 2:
            # Binary classification (1 or 2 classes)
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(optimizer=Adam(learning_rate=0.001), 
                         loss='binary_cross-entropy',
                         metrics=['accuracy', keras.metrics.AUC(name='auc'),
                                 keras.metrics.Precision(name='precision'),
                                 keras.metrics.Recall(name='recall')])
        else:
            # Multi-class classification
            model.add(layers.Dense(output_dim, activation='softmax'))
            model.compile(optimizer=Adam(learning_rate=0.001), 
                         loss='categorical_cross-entropy',
                         metrics=['accuracy'])
    else:
        model.add(layers.Dense(1, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mse', 
                     metrics=['mae', 'mse'])
    
    return model

def create_lstm_model(input_shape, sequence_length=10):
    """Create LSTM model for sequential data"""
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        try:
            import keras
            from keras import layers
        except ImportError:
            return None
    
    model = keras.Sequential([
        layers.Input(shape=(sequence_length, input_shape)),
        layers.LSTM(64, return_sequences=True, dropout=0.2),
        layers.LSTM(32, dropout=0.2),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_cross-entropy',
                 metrics=['accuracy', keras.metrics.AUC()])
    return model