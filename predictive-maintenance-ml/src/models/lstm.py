from __future__ import annotations

try:
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Dense, LSTM
    from tensorflow.keras.optimizers import Adam
except ImportError:  # pragma: no cover
    Model = None
    LSTM = None
    Dense = None
    Adam = None


def build_lstm_model(input_shape: tuple[int, int], hidden_units: int = 64, dropout_rate: float = 0.2, learning_rate: float = 1e-3):
    """Build an LSTM regressor with optional TensorFlow dependency.

    When TensorFlow is not installed, the function raises a clear ImportError instead of failing
    silently. This keeps the project modular and explicit about runtime requirements.
    """
    if Model is None or LSTM is None or Dense is None or Adam is None:
        raise ImportError('TensorFlow is required to build the LSTM model. Install the TensorFlow runtime for the full deep-learning stack.')

    inputs = __import__('tensorflow').keras.Input(shape=input_shape)
    x = LSTM(hidden_units, return_sequences=False)(inputs)
    x = __import__('tensorflow').keras.layers.Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='linear')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model
