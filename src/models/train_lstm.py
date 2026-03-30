"""
models/train_lstm.py
────────────────────
Trains an LSTM Autoencoder for temporal anomaly detection on network flows.

The model learns to reconstruct sequences of NORMAL traffic. At inference
time, high reconstruction error signals an anomaly.

Architecture
────────────
    Input (SEQ_LEN × N_FEATURES)
      → LSTM encoder (64 units) → RepeatVector
      → LSTM decoder (64 units) → Dense(N_FEATURES)
    Output: reconstructed sequence

Usage
─────
    # Train on the NF-UQ-NIDS dataset (extracts benign traffic automatically):
    python -m models.train_lstm --csv-path data/NF-UQ-NIDS-v2.csv

    # Train on a smaller sample for quick testing:
    python -m models.train_lstm --csv-path data/NF-UQ-NIDS-v2.csv --sample-frac 0.01

    # Convert and export to TFLite for Pi deployment:
    python -m models.train_lstm --csv-path data/NF-UQ-NIDS-v2.csv --export-tflite
"""

import argparse
import logging
import sys
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("shield.train_lstm")

# ── Constants ─────────────────────────────────────────────────────────────────

SEQ_LEN = 10          # number of consecutive flows per sequence
N_FEATURES = 39       # matches features/extractor.py output
LATENT_DIM = 64       # LSTM hidden units
BATCH_SIZE = 256
EPOCHS = 30
VALIDATION_SPLIT = 0.1

MODEL_DIR = Path("models/artefacts")
KERAS_PATH = MODEL_DIR / "lstm_autoencoder.keras"
TFLITE_PATH = MODEL_DIR / "lstm_autoencoder.tflite"
THRESHOLD_PATH = MODEL_DIR / "lstm_threshold.npz"


# ── Dataset Loading ───────────────────────────────────────────────────────────

def load_benign_features(csv_path: str, sample_frac: float = 0.05,
                          chunk_size: int = 250_000) -> np.ndarray:
    """
    Stream the NF-UQ-NIDS CSV, keep only benign traffic (Label==0),
    drop identifiers, and return a float32 feature matrix.
    """
    logger.info("Streaming dataset from %s (sample_frac=%.3f)...", csv_path, sample_frac)
    
    columns_to_drop = [
        'IPV4_SRC_ADDR', 'L4_SRC_PORT',
        'IPV4_DST_ADDR', 'L4_DST_PORT',
        'Attack', 'Label', 'Dataset'
    ]
    
    sampled_chunks = []
    total = 0
    
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        total += len(chunk)
        
        # Keep only benign
        benign = chunk[chunk['Label'] == 0]
        if len(benign) == 0:
            continue
            
        sampled = benign.sample(frac=min(sample_frac, 1.0), random_state=42)
        
        # Drop identifier columns
        cols_in = [c for c in columns_to_drop if c in sampled.columns]
        sampled = sampled.drop(columns=cols_in)
        
        sampled_chunks.append(sampled)
        sys.stdout.write(f"\rProcessed {total:,} rows, kept {sum(len(c) for c in sampled_chunks):,} benign samples...")
        sys.stdout.flush()
    
    print()
    
    if not sampled_chunks:
        raise ValueError("No benign samples found in dataset!")
    
    df = pd.concat(sampled_chunks, ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Clip to float32 range
    f32_max = np.finfo(np.float32).max
    df = df.clip(lower=-f32_max, upper=f32_max)
    
    X = df.values.astype(np.float32)
    logger.info("Final benign feature matrix: %s (from %d total rows)", X.shape, total)
    return X


# ── Sequence Builder ──────────────────────────────────────────────────────────

def build_sequences(X: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """
    Slide a window of `seq_len` over the feature matrix to create
    overlapping sequences for the autoencoder.
    
    Returns shape: (n_sequences, seq_len, n_features)
    """
    n_samples, n_features = X.shape
    n_sequences = n_samples - seq_len + 1
    
    if n_sequences <= 0:
        raise ValueError(f"Need at least {seq_len} samples, got {n_samples}")
    
    sequences = np.zeros((n_sequences, seq_len, n_features), dtype=np.float32)
    for i in range(n_sequences):
        sequences[i] = X[i:i + seq_len]
    
    logger.info("Built %d sequences of shape (%d, %d)", n_sequences, seq_len, n_features)
    return sequences


# ── Model Architecture ────────────────────────────────────────────────────────

def build_autoencoder(seq_len: int = SEQ_LEN, n_features: int = N_FEATURES,
                       latent_dim: int = LATENT_DIM):
    """
    LSTM Autoencoder:
        Encoder: LSTM(latent_dim) compresses the sequence into a fixed vector.
        Decoder: RepeatVector + LSTM(latent_dim) reconstructs the sequence.
        
    The model is trained to minimize reconstruction error (MSE).
    Normal traffic → low error. Anomalous traffic → high error.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    # Encoder
    inputs = keras.Input(shape=(seq_len, n_features), name="input_sequence")
    
    # First LSTM layer with return_sequences for deeper encoding
    encoded = layers.LSTM(latent_dim * 2, activation='tanh',
                          return_sequences=True, name="encoder_1")(inputs)
    encoded = layers.Dropout(0.2)(encoded)
    
    # Bottleneck — compress to single latent vector
    encoded = layers.LSTM(latent_dim, activation='tanh',
                          return_sequences=False, name="encoder_bottleneck")(encoded)
    
    # Decoder — expand latent vector back to sequence
    decoded = layers.RepeatVector(seq_len, name="repeat")(encoded)
    decoded = layers.LSTM(latent_dim, activation='tanh',
                          return_sequences=True, name="decoder_1")(decoded)
    decoded = layers.Dropout(0.2)(decoded)
    decoded = layers.LSTM(latent_dim * 2, activation='tanh',
                          return_sequences=True, name="decoder_2")(decoded)
    
    # Output — reconstruct each timestep
    outputs = layers.TimeDistributed(
        layers.Dense(n_features), name="output"
    )(decoded)
    
    model = keras.Model(inputs, outputs, name="lstm_autoencoder")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train(csv_path: str, sample_frac: float = 0.05, epochs: int = EPOCHS,
          export_tflite: bool = True):
    """Full training pipeline."""
    import tensorflow as tf
    from tensorflow import keras
    
    # 1. Load data
    X = load_benign_features(csv_path, sample_frac=sample_frac)
    
    # 2. Normalize (per-feature StandardScaler)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X_norm = (X - mean) / std
    
    # 3. Build sequences
    sequences = build_sequences(X_norm)
    
    # Shuffle
    indices = np.random.permutation(len(sequences))
    sequences = sequences[indices]
    
    # 4. Split train/val
    split = int(len(sequences) * (1 - VALIDATION_SPLIT))
    X_train = sequences[:split]
    X_val = sequences[split:]
    
    logger.info("Train: %d sequences, Val: %d sequences", len(X_train), len(X_val))
    
    # 5. Build model
    actual_features = X.shape[1]
    model = build_autoencoder(
        seq_len=SEQ_LEN,
        n_features=actual_features,
        latent_dim=LATENT_DIM,
    )
    model.summary(print_fn=logger.info)
    
    # 6. Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=3, min_lr=1e-6, verbose=1
        ),
    ]
    
    # 7. Train (autoencoder: input == target)
    history = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    
    # 8. Compute anomaly threshold from validation set
    val_pred = model.predict(X_val, batch_size=BATCH_SIZE)
    val_mse = np.mean((X_val - val_pred) ** 2, axis=(1, 2))
    
    threshold = np.percentile(val_mse, 95)  # 95th percentile as threshold
    logger.info("Anomaly threshold (95th percentile): %.6f", threshold)
    logger.info("Val MSE — mean: %.6f, std: %.6f, max: %.6f",
                val_mse.mean(), val_mse.std(), val_mse.max())
    
    # 9. Save Keras model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(KERAS_PATH)
    logger.info("Keras model saved to %s", KERAS_PATH)
    
    # Save normalization params and threshold
    np.savez(THRESHOLD_PATH, mean=mean, std=std, threshold=threshold)
    logger.info("Threshold & scaler saved to %s", THRESHOLD_PATH)
    
    # 10. Export to TFLite
    if export_tflite:
        export_to_tflite(model)
    
    logger.info("✓ LSTM training complete!")
    return model, history


def export_to_tflite(model=None):
    """Convert Keras model to TFLite for Raspberry Pi deployment."""
    import tensorflow as tf
    
    if model is None:
        from tensorflow import keras
        logger.info("Loading Keras model from %s...", KERAS_PATH)
        model = keras.models.load_model(KERAS_PATH)
    
    logger.info("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimize for Pi
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    
    TFLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
    
    size_mb = len(tflite_model) / (1024 * 1024)
    logger.info("TFLite model saved to %s (%.2f MB)", TFLITE_PATH, size_mb)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train LSTM Autoencoder for SHIELD IDS anomaly detection"
    )
    parser.add_argument(
        "--csv-path", default="data/NF-UQ-NIDS-v2.csv",
        help="Path to the NF-UQ-NIDS CSV dataset"
    )
    parser.add_argument(
        "--sample-frac", type=float, default=0.05,
        help="Fraction of benign traffic to sample (default: 0.05)"
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help=f"Training epochs (default: {EPOCHS})"
    )
    parser.add_argument(
        "--export-tflite", action="store_true", default=True,
        help="Export to TFLite after training (default: True)"
    )
    parser.add_argument(
        "--convert-only", action="store_true",
        help="Only convert existing Keras model to TFLite (no training)"
    )
    
    args = parser.parse_args()
    
    if args.convert_only:
        export_to_tflite()
    else:
        train(
            csv_path=args.csv_path,
            sample_frac=args.sample_frac,
            epochs=args.epochs,
            export_tflite=args.export_tflite,
        )


if __name__ == "__main__":
    main()
