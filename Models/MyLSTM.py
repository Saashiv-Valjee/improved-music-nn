import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd

key_order=['pitch', 'step', 'duration']

def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
    mse = (y_true - y_pred) ** 2
    positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
    return tf.reduce_mean(mse + positive_pressure)


def create_LSTM(seq_length, learning_rate):
    input_shape = (seq_length, 3)

    inputs = Input(input_shape)
    x = LSTM(128)(inputs)

    outputs = {'pitch': Dense(128, name='pitch')(x),
               'step': Dense(1, name='step')(x),
               'duration': Dense(1, name='duration')(x),
              }

    model = tf.keras.models.Model(inputs, outputs)

    loss = {'pitch': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            'step': mse_with_positive_pressure,
            'duration': mse_with_positive_pressure,
           }

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(loss=loss, optimizer=optimizer)

    return model

def train_LSTM(model, train_ds, val_ds, epochs):
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath='.data/training_checkpoints/ckpt_{epoch}', save_weights_only=True),
    ]

    loss = {
        'pitch': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        'step': mse_with_positive_pressure,
        'duration': mse_with_positive_pressure,
    }

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(loss=loss, 
                  loss_weights={'pitch': 0.1, 'step': 1.0, 'duration': 1.0},
                  optimizer=optimizer)

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    model.save(r'data/path_to_save_model')

    return history

def predict_next_note(notes: np.ndarray, model: tf.keras.Model, 
                      temperature: float = 1.0) -> int:
    """Generates a note IDs using a trained sequence model."""

    assert temperature > 0

    # Add batch dimension
    inputs = tf.expand_dims(notes, 0)

    predictions = model.predict(inputs)
    pitch_logits = predictions['pitch']
    step = predictions['step']
    duration = predictions['duration']

    pitch_logits /= temperature
    pitch = tf.random.categorical(pitch_logits, num_samples=1)
    pitch = tf.squeeze(pitch, axis=-1)
    duration = tf.squeeze(duration, axis=-1)
    step = tf.squeeze(step, axis=-1)

    # `step` and `duration` values should be non-negative
    step = tf.maximum(0, step)
    duration = tf.maximum(0, duration)

    return int(pitch), float(step), float(duration)

def Gen_LSTM(model,sample_notes,temperature=10,num_predictions=128,seq_length=25,vocab_size=128):

    # The initial sequence of notes while the pitch is normalized similar to training sequences
    input_notes = (sample_notes[:seq_length] / np.array([vocab_size, 1, 1]))

    generated_notes = []
    prev_start = 0

    for _ in range(num_predictions):
        pitch, step, duration = predict_next_note(input_notes, model, temperature)
        start = prev_start + step
        end = start + duration
        input_note = (pitch, step, duration)
        generated_notes.append((*input_note, start, end))
        input_notes = np.delete(input_notes, 0, axis=0)
        input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
        prev_start = start

    generated_notes = pd.DataFrame(generated_notes, columns=(*key_order, 'start', 'end'))

    generated_notes.head(30)

    return generated_notes