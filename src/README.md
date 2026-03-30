# SHIELD — AI-Powered Intrusion Detection

Privacy-preserving IDS using Zeek + ML ensemble + TLS fingerprinting. No packet decryption.

## Features

- **4-Model Ensemble**: Isolation Forest + Random Forest + LSTM Autoencoder + Statistical Detector
- **TLS Fingerprinting**: JA3/JA3S hash analysis, threat-intel matching against known malware signatures
- **Real-time Dashboard**: WebSocket-powered live traffic view, alert queue, risk gauges, SHAP explainability
- **Automated Response**: iptables auto-blocking with configurable thresholds and auto-expiry
- **Privacy-Preserving**: Analyzes metadata (flow stats, TLS fingerprints) — never decrypts packet payloads

## Quick start

### 1. Install Zeek
```bash
# Debian / Raspberry Pi OS
sudo apt install zeek

# Start capturing (run in a separate terminal)
sudo zeek -i eth0 policy/protocols/ssl/ja3.zeek LogAscii::use_json=T
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
# On Raspberry Pi, replace tensorflow with:
pip install tflite-runtime
```

### 3. Train models (first time)
```bash
# Download NF-UQ-NIDS-v2 dataset
# Train Random Forest + Isolation Forest + Statistical baseline:
python main.py --mode train \
    --csv-path data/NF-UQ-NIDS-v2.csv

# Train LSTM Autoencoder separately:
python -m models.train_lstm \
    --csv-path data/NF-UQ-NIDS-v2.csv \
    --epochs 30 \
    --export-tflite
```

### 4. Run live detection
```bash
# Terminal 1: Zeek (already running from step 1)

# Terminal 2: SHIELD pipeline + Dashboard (all-in-one)
python main.py --mode live
# Dashboard available at http://localhost:8000
```

### 5. Test with a saved log
```bash
python main.py --mode replay \
    --conn-log tests/sample_conn.log \
    --ssl-log  tests/sample_ssl.log
```

## Project structure

```
shield/
├── config/
│   └── settings.py              # all tuneable parameters
├── capture/
│   └── zeek_reader.py           # zat-based multi-log tailer + uid joiner
├── features/
│   ├── extractor.py             # 39-feature vector extraction (NF-UQ-NIDS schema)
│   └── tls_fingerprint.py       # JA3/JA3S fingerprinting + threat-intel engine
├── models/
│   ├── detectors.py             # IsoForest, RF, LSTM, Statistical detectors
│   ├── train_lstm.py            # LSTM Autoencoder training + TFLite export
│   └── artefacts/               # saved .joblib / .tflite / .npz files
├── scoring/
│   └── risk_scorer.py           # ensemble fusion + TLS analysis + SHAP + alerts
├── response/
│   └── auto_block.py            # iptables block with auto-expiry
├── dashboard/
│   ├── api.py                   # FastAPI REST + WebSocket + TLS stats
│   └── index.html               # real-time dashboard UI
├── main.py                      # entry point (live / train / replay modes)
└── requirements.txt
```

## TLS Fingerprinting

SHIELD performs deep TLS metadata analysis on every SSL/TLS flow:

| Check | Description |
|-------|-------------|
| **JA3 Threat Intel** | Matches client TLS fingerprints against known malware (Emotet, CobaltStrike, TrickBot, etc.) |
| **JA3S Analysis** | Server-side fingerprint matching for C2 infrastructure |
| **TLS Version** | Flags deprecated SSL/TLS versions (SSLv2, SSLv3, TLS 1.0/1.1) |
| **Cipher Strength** | Detects weak ciphers (RC4, DES, NULL, EXPORT) |
| **SNI Anomaly** | DGA detection via entropy analysis, suspicious TLD flagging |
| **Frequency** | Tracks first-seen fingerprints for zero-day detection |

## LSTM Training

Train the temporal anomaly detector separately for best results:

```bash
# Full training (recommended)
python -m models.train_lstm --csv-path data/NF-UQ-NIDS-v2.csv --sample-frac 0.1

# Quick test
python -m models.train_lstm --csv-path data/NF-UQ-NIDS-v2.csv --sample-frac 0.01 --epochs 5

# Convert existing model to TFLite only
python -m models.train_lstm --convert-only
```

The LSTM autoencoder learns to reconstruct normal traffic sequences. At inference time, high reconstruction error signals a temporal anomaly.

## Tuning

Edit `config/settings.py`:
- `ALERT_THRESHOLD` — lower = more sensitive (more false positives)
- `BLOCK_THRESHOLD` — score above which auto-block fires
- `ENSEMBLE_WEIGHTS` — rebalance detector contributions
- `AUTO_BLOCK_ENABLED` — set True to enable iptables auto-blocking
- `JA3_BLOCKLIST` — add known-malicious JA3 hashes

## Dashboard API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve dashboard HTML |
| `/alerts` | GET | Recent alerts (JSON) |
| `/stats` | GET | Live system metrics |
| `/traffic` | GET | Recent network flows |
| `/traffic/stats` | GET | Aggregated traffic statistics |
| `/blocked` | GET | Current IP block list |
| `/tls/stats` | GET | TLS fingerprinting statistics |
| `/block/{ip}` | POST | Manually block an IP |
| `/unblock/{ip}` | POST | Manually unblock an IP |
| `/test-trigger` | POST | Generate test alert (demo) |
| `/ws/alerts` | WS | Live WebSocket feed |
