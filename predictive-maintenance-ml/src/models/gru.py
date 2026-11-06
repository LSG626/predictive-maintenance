from __future__ import annotations

try:
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Dense, GRU
    from tensorflow.keras.optimizers import Adam
except ImportError:  # pragma: no cover
    Model = None
    GRU = None
    Dense = None
    Adam = None


def build_gru_model(input_shape: tuple[int, int], hidden_units: int = 64, dropout_rate: float = 0.2, learning_rate: float = 1e-3):
    if Model is None or GRU is None or Dense is None or Adam is None:
        raise ImportError('TensorFlow is required to build the GRU model. Install the TensorFlow runtime for the full deep-learning stack.')

    inputs = __import__('tensorflow').keras.Input(shape=input_shape)
    x = GRU(hidden_units, return_sequences=False)(inputs)
    x = __import__('tensorflow').keras.layers.Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='linear')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model
