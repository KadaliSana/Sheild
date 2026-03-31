# SHIELD - AI-Powered Intrusion Detection System

SHIELD is a next-generation Intrusion Detection System (IDS) that combines deep learning, machine learning, and real-time network traffic analysis to detect and prevent cyber threats. It provides a comprehensive security solution with automated threat intelligence, TLS fingerprinting, and a modern web-based dashboard.

## Features

- **Multi-Model Detection**: Uses a hybrid approach with:
  - **Transformer Autoencoder**: Detects anomalies in network traffic patterns
  - **Random Forest**: Classifies 21 different types of cyberattacks
  - **Isolation Forest**: Unsupervised anomaly detection
  - **Statistical Analysis**: Real-time baseline deviation detection
- **Real-Time Monitoring**: Continuously monitors network traffic using Zeek
- **TLS Fingerprinting**: Identifies malicious TLS connections using JA3 hashes
- **Automated Response**: Automatically blocks detected threats
- **Live Dashboard**: Modern web interface with real-time analytics
- **Threat Intelligence**: Integrates with threat intelligence feeds

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          SHIELD - IDS SYSTEM                             │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  ZEEK Network Monitor                                              │  │
│  │  - Captures conn.log, ssl.log, dns.log, weird.log                   │  │
│  │  - Real-time log streaming                                         │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  Feature Extraction Engine                                         │  │
│  │  - Flow-based feature engineering                                  │  │
│  │  - TLS JA3 fingerprint extraction                                  │  │
│  │  - Statistical feature extraction                                  │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  AI Detection Models                                               │  │
│  │  ┌──────────────────────────────────────────────────────────────┐  │
│  │  │  Transformer Autoencoder (Deep Learning)                     │  │
│  │  │  - Anomaly detection                                           │  │
│  │  │  - Sequence-based analysis                                     │  │
│  │  └──────────────────────────────────────────────────────────────┘  │
│  │  ┌──────────────────────────────────────────────────────────────┐  │
│  │  │  Random Forest Classifier (Supervised)                       │  │
│  │  │  - 21 attack categories                                        │  │
│  │  │  - Attack classification                                       │  │
│  │  └──────────────────────────────────────────────────────────────┘  │
│  │  ┌──────────────────────────────────────────────────────────────┐  │
│  │  │  Isolation Forest (Unsupervised)                             │  │
│  │  │  - Anomaly detection                                           │  │
│  │  │  - Outlier detection                                           │  │
│  │  └──────────────────────────────────────────────────────────────┘  │
│  │  ┌──────────────────────────────────────────────────────────────┐  │
│  │  │  Statistical Baseline                                          │  │
│  │  │  - Real-time deviation detection                               │  │
│  │  │  - Threshold-based alerting                                    │  │
│  │  └──────────────────────────────────────────────────────────────┘  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  Risk Scoring & Decision Engine                                    │  │
│  │  - Ensemble scoring algorithm                                      │  │
│  │  - Weighted threat assessment                                      │  │
│  │  - Automated blocking                                              │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  Dashboard & API                                                   │  │
│  │  - Real-time analytics                                             │  │
│  │  - Threat visualization                                            │  │
│  │  - WebSocket notifications                                         │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Python 3.8+
- Zeek Network Security Monitor (installed and configured)
- Required Python packages:
  ```bash
  pip install -r requirements.txt
  ```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd IDS
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Zeek**
   Ensure Zeek is installed and configured to generate the following logs:
   - `conn.log`
   - `ssl.log`
   - `dns.log`
   - `weird.log`

   Update the log paths in `src/config/settings.py` if needed:
   ```python
   ZEEK_LOG_DIR = Path("/path/to/zeek/logs")
   ```

## Usage

### Start Monitoring (Live Mode)

Run the system in live monitoring mode:

```bash
python3 main.py --mode live
```

This will:
1. Start Zeek (if not already running)
2. Load trained models
3. Monitor network traffic in real-time
4. Start the web dashboard on port 8000

Access the dashboard at: [http://localhost:8000](http://localhost:8000)

### Train Models

To train the machine learning models:

```bash
python3 main.py --mode train
```

This will train:
- Random Forest classifier
- Isolation Forest
- Statistical baseline

```bash
python3 train_transformer.py
```


## Configuration

Edit `src/config/settings.py` to customize:

- **Network Interface**: `CAPTURE_IFACE` (default: `eth0`)
- **Flow Window**: `FLOW_WINDOW_SEC` (default: 30 seconds)
- **Thresholds**: `ALERT_THRESHOLD`, `BLOCK_THRESHOLD` (0-100)
- **Model Paths**: Locations of trained models
- **JA3 Blocklist**: Malicious TLS fingerprints

## Model Details

### Transformer Autoencoder
- **Architecture**: 4 encoder layers, 4 decoder layers, 4 attention heads
- **Input**: 20-dimensional feature vector
- **Output**: Reconstructed feature vector
- **Detection**: High reconstruction error indicates anomaly

### Random Forest Classifier
- **Classes**: 21 attack categories (including benign)
- **Features**: 20 network flow features
- **Depth**: 15
- **Estimators**: 200

### Isolation Forest
- **Estimators**: 100
- **Max Samples**: 256
- **Contamination**: 0.01 (1%)

## Threat Intelligence

The system includes a built-in JA3 fingerprint blocklist for known-malicious TLS connections. This list is automatically checked during TLS analysis.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
