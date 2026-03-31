import argparse
import logging
import sys
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("shield.train_transformer")

# ── Constants ─────────────────────────────────────────────────────────────────

SEQ_LEN = 10          # number of consecutive flows per sequence
N_FEATURES = 39       # matches features/extractor.py output
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 4
BATCH_SIZE = 512
EPOCHS = 50
VALIDATION_SPLIT = 0.1

MODEL_DIR = Path("models/artefacts")
PT_PATH = MODEL_DIR / "transformer_autoencoder.pt"
THRESHOLD_PATH = MODEL_DIR / "transformer_threshold.npz"


# ── Dataset Loading ───────────────────────────────────────────────────────────

def load_benign_features(csv_path: str, sample_frac: float = 0.01,
                          chunk_size: int = 1000000) -> np.ndarray:
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

class TransformerAutoencoder(nn.Module):
    def __init__(self, n_features: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_projection = nn.Linear(n_features, d_model)
        
        # Positional encoding
        max_len = 5000
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # shape: (1, max_len, d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.output_projection = nn.Linear(d_model, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = self.input_projection(x)
        # Add positional encoding
        x = x + self.pe[:, :seq_len, :]
        encoded = self.transformer_encoder(x)
        out = self.output_projection(encoded)
        return out


# ── Training ──────────────────────────────────────────────────────────────────

def train(csv_path: str, sample_frac: float = 0.05, epochs: int = EPOCHS):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Using device: %s", device)
    
    # 1. Load data
    X = load_benign_features(csv_path, sample_frac=sample_frac)
    
    # 2. Normalize (per-feature StandardScaler)
    # Compute mean and std using float64 to prevent overflow across 1M+ rows
    mean = np.mean(X, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(X, axis=0, dtype=np.float64).astype(np.float32) + 1e-8
    X_norm = (X - mean) / std
    X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)
    
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
    
    train_dataset = TensorDataset(torch.tensor(X_train))
    val_dataset = TensorDataset(torch.tensor(X_val))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # 5. Build model
    actual_features = X.shape[1]
    model = TransformerAutoencoder(
        n_features=actual_features,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
    
    # 6. Train loop
    best_val_loss = float('inf')
    early_stop_patience = 5
    patience_counter = 0
    best_model_state = None
    
    model.train()
    for epoch in range(epochs):
        train_loss = 0.0
        for i, (batch_seq,) in enumerate(train_loader):
            batch_seq = batch_seq.to(device)
            optimizer.zero_grad()
            
            reconstruction = model(batch_seq)
            loss = criterion(reconstruction, batch_seq)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_seq.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_val_mses = []
        with torch.no_grad():
            for (batch_seq,) in val_loader:
                batch_seq = batch_seq.to(device)
                reconstruction = model(batch_seq)
                loss = criterion(reconstruction, batch_seq)
                val_loss += loss.item() * batch_seq.size(0)
                
                # For computing threshold later
                batch_mse = torch.mean((batch_seq - reconstruction) ** 2, dim=(1, 2)).cpu().numpy()
                all_val_mses.extend(batch_mse)
                
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        
        logger.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        model.train()
        
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 7. Compute anomaly threshold from validation set
    all_val_mses = np.array(all_val_mses)
    threshold = np.percentile(all_val_mses, 95)
    logger.info("Anomaly threshold (95th percentile): %.6f", threshold)
    logger.info("Val MSE — mean: %.6f, std: %.6f, max: %.6f",
                all_val_mses.mean(), all_val_mses.std(), all_val_mses.max())
    
    # 8. Save Model and Params
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Export to TorchScript
    model.eval()
    dummy_input = torch.randn(1, SEQ_LEN, actual_features, device=device)
    scripted_model = torch.jit.script(model)
    scripted_model.save(str(PT_PATH))
    logger.info("PyTorch model exported to %s", PT_PATH)
    
    # Save normalization params and threshold
    np.savez(THRESHOLD_PATH, mean=mean, std=std, threshold=threshold)
    logger.info("Threshold & scaler saved to %s", THRESHOLD_PATH)
    
    logger.info("✓ Transformer training complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Train Transformer Autoencoder for SHIELD IDS anomaly detection"
    )
    parser.add_argument(
        "--csv-path", default="models/data/NF-UQ-NIDS-v2.csv",
        help="Path to the NF-UQ-NIDS CSV dataset"
    )
    parser.add_argument(
        "--sample-frac", type=float, default=0.01,
        help="Fraction of benign traffic to sample (default: 0.05)"
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help=f"Training epochs (default: {EPOCHS})"
    )
    
    args = parser.parse_args()
    
    train(
        csv_path=args.csv_path,
        sample_frac=args.sample_frac,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
