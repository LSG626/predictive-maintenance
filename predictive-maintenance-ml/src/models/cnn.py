from __future__ import annotations

try:
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D
    from tensorflow.keras.optimizers import Adam
except ImportError:  # pragma: no cover
    Model = None
    Conv1D = None
    Dense = None
    Dropout = None
    Flatten = None
    MaxPooling1D = None
    Adam = None


def build_cnn_model(input_shape: tuple[int, int], filters: int = 32, kernel_size: int = 3, learning_rate: float = 1e-3):
    if Model is None or Conv1D is None or Dense is None or Flatten is None or MaxPooling1D is None or Dropout is None or Adam is None:
        raise ImportError('TensorFlow is required to build the CNN model. Install the TensorFlow runtime for the full deep-learning stack.')

    inputs = __import__('tensorflow').keras.Input(shape=input_shape)
    x = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    outputs = Dense(1, activation='linear')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model
