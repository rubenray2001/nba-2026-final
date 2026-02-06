"""
Deep Learning Model Definition (LSTM)
Architecture: Bidirectional LSTM -> Dense -> Output
"""
import tensorflow as tf
import tensorflow as tf
import tensorflow as tf
import os
import joblib
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam

class NBALSTMModel:
    def __init__(self, input_shape):
        """
        Initialize LSTM model
        input_shape: (lookback_steps, num_features)
        """
        self.input_shape = input_shape
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential([
            # 1. Input + Bidirectional LSTM Layer based on time-series
            Bidirectional(LSTM(64, return_sequences=True), input_shape=self.input_shape),
            BatchNormalization(),
            Dropout(0.3),
            
            # 2. Second LSTM Layer for deeper pattern recognition
            Bidirectional(LSTM(32)),
            BatchNormalization(),
            Dropout(0.3),
            
            # 3. Dense Layers for regression/classification
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            
            # 4. Output Layer (Home Score, Visitor Score) -> or Winner Prob
            # Let's target Home Spread (Home - Visitor) and Total, or plain Winner Prob?
            # User wants "Elite Accuracy" -> predicting Winner Probability is direct.
            Dense(1, activation='sigmoid') 
        ])
        
        optimizer = Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        print("Training LSTM Model...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        return history
    
    def save(self, path):
        self.model.save(path)
        print(f"Model saved to {path}")

    @staticmethod
    def load(path):
        return tf.keras.models.load_model(path)

if __name__ == "__main__":
    # Dummy test
    import numpy as np
    model = NBALSTMModel((10, 25)) # 10 games, 25 features
    model.model.summary()
