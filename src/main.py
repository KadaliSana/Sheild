
import argparse
import asyncio
import uvicorn
from dashboard.api import app
import pandas as pd
import logging
import signal
import sys
import time
import threading
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from capture.zeek_reader import ZeekFlowReader, load_conn_dataframe, load_ssl_dataframe, merge_conn_ssl
from features.extractor import FeatureExtractor, extract_dataframe
from scoring.risk_scorer import RiskScorer
from response.auto_block import handle_alert
from dashboard.api import ingest_alert, ingest_flow, increment_flow_counter, broadcast_alert, broadcast_flow, update_tls_stats
from config.settings import ALERT_THRESHOLD

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("shield.main")


# ── pipeline ──────────────────────────────────────────────────────────────────

class SHIELDPipeline:
    def __init__(self):
        self.extractor = FeatureExtractor()
        self.scorer    = RiskScorer()
        self._loop     = None     # asyncio event loop (set in run_live)

    def setup(self):
        logger.info("Loading models…")
        self.scorer.load_models()
        logger.info("Models ready.")

    def on_flow(self, flow: dict):
        """
        Callback invoked by ZeekFlowReader for every complete flow.
        Runs in a background thread — keep it fast.
        """
        try:
            increment_flow_counter()

            # extract features
            feature_vec = self.extractor.extract(flow)

            # score
            alert = self.scorer.evaluate(flow, feature_vec)

            if alert is None:
                return

            alert_dict = alert.to_dict()

            # Build flow record for the traffic table
            is_malicious = alert.attack_type != "Benign" and alert.risk_score >= ALERT_THRESHOLD
            dur_val = flow.get("duration") or 0
            if hasattr(dur_val, "total_seconds"):
                dur_float = dur_val.total_seconds()
            else:
                dur_float = float(dur_val)
            dur_ms = round(dur_float * 1000, 1)

            flow_record = {
                "uid":         alert_dict.get("uid", ""),
                "timestamp":   alert_dict.get("timestamp", time.time()),
                "src":         alert_dict.get("src", "?"),
                "dst":         alert_dict.get("dst", "?"),
                "proto":       alert_dict.get("proto", ""),
                "bytes_in":    int(flow.get("orig_ip_bytes") or flow.get("orig_bytes") or 0),
                "bytes_out":   int(flow.get("resp_ip_bytes") or flow.get("resp_bytes") or 0),
                "packets_in":  int(flow.get("orig_pkts") or 0),
                "packets_out": int(flow.get("resp_pkts") or 0),
                "duration_ms": dur_ms,
                "attack_type": alert.attack_type,
                "risk_score":  alert.risk_score,
                "severity":    alert.severity,
                "is_malicious": is_malicious,
            }
            ingest_flow(flow_record)
            if self._loop and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    broadcast_flow(flow_record), self._loop
                )

            # update TLS fingerprinting stats for dashboard
            update_tls_stats(self.scorer.get_tls_stats())
            
            if alert.risk_score < ALERT_THRESHOLD:
                time.sleep(0.02)  # Process at ~50 flows per second for UI animation
                return

            handle_alert(alert)
            ingest_alert(alert_dict)
            if self._loop and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    broadcast_alert(alert_dict), self._loop
                )
            
            time.sleep(0.02)  # Process at ~50 flows per second for UI animation

        except Exception as exc:
            logger.error("Pipeline error: %s", exc, exc_info=True)

    # ── live mode ─────────────────────────────────────────────────────────────

    def run_live(self,conn_log: Path, ssl_log: Path):
        logger.info("Starting live monitoring…")
        reader = ZeekFlowReader(
            on_flow=self.on_flow,
            conn_log=str(conn_log),
            ssl_log=str(ssl_log)
        )
        reader.start()
        # set up asyncio loop for WebSocket broadcasts
        self._loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="asyncio-loop"
        )
        loop_thread.start()

        # graceful shutdown
        def _shutdown(sig, frame):
            logger.info("Shutting down…")
            reader.stop()
            self.scorer.save_models()
            self._loop.call_soon_threadsafe(self._loop.stop)
            sys.exit(0)

        signal.signal(signal.SIGINT,  _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        logger.info("SHIELD is running. Press Ctrl-C to stop.")
        logger.info("Dashboard API: http://localhost:8000")

        # keep main thread alive by running the dashboard API

        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

    # ── training mode ─────────────────────────────────────────────────────────

    def run_train(self, csv_path: Path):
            """
            Offline training using Chunked Processing to prevent RAM exhaustion.
            """

            logger.info(f"Streaming NetFlow dataset from {csv_path} in chunks...")

            CHUNK_SIZE = 250_000 
            SAMPLE_FRACTION = 0.05   # Keep 5% of each chunk (Adjust this up/down based on your RAM)
            
            sampled_chunks = []
            total_rows_processed = 0

            # 1. Read the massive CSV piece-by-piece
            # This acts like a conveyor belt, never overloading your memory.
            for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE):
                total_rows_processed += len(chunk)
                
                # Keep a random, highly representative slice of this specific chunk
                sampled_chunk = chunk.sample(frac=SAMPLE_FRACTION, random_state=42)
                sampled_chunks.append(sampled_chunk)
                
                # Print a live updating progress bar
                sys.stdout.write(f"\rProcessed {total_rows_processed:,} rows...")
                sys.stdout.flush()

            print() # Add a newline when the loop finishes
            logger.info("Finished streaming file. Assembling training data...")

            # 2. Combine all the tiny samples into one highly-dense, manageable dataframe
            df = pd.concat(sampled_chunks, ignore_index=True)
            
            logger.info(f"Original dataset: {total_rows_processed:,} rows.")
            logger.info(f"Actual Training RAM footprint: {len(df):,} rows.")

            # 3. Multi-class attack labels from the 'Attack' column
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_attack = le.fit_transform(df['Attack'].values)  # string → int
            attack_label_map = {int(i): name for i, name in enumerate(le.classes_)}
            logger.info(f"Attack categories ({len(attack_label_map)}): {list(le.classes_)}")

            # Binary labels for unsupervised detectors (IsoForest, Statistical)
            y_binary = df['Label'].values.astype(int)

            # 4. Drop Identifiers and Metadata
            columns_to_drop = [
                'IPV4_SRC_ADDR', 'L4_SRC_PORT', 
                'IPV4_DST_ADDR', 'L4_DST_PORT', 
                'Attack', 'Label', 'Dataset'
            ]
            
            # Safely drop columns
            cols_in_df = [col for col in columns_to_drop if col in df.columns]
            X_df = df.drop(columns=cols_in_df)
            
            X_df = X_df.replace([np.inf, -np.inf], np.nan)
            X_df = X_df.fillna(0)

            # Clip values to the absolute max/min of float32 to prevent "value too large" errors
            f32_max = np.finfo(np.float32).max
            f32_min = np.finfo(np.float32).min
            X_df = X_df.clip(lower=f32_min, upper=f32_max)

            # Now convert to float32
            X = X_df.values.astype(np.float32)

            # 6. Split into Train/Test
            X_train, X_test, y_attack_train, y_attack_test = train_test_split(
                X, y_attack, test_size=0.2, random_state=42
            )
            # Also split binary labels in the same order for unsupervised models
            y_binary_train = y_binary[X_train.shape[0] * 0 : len(y_binary)][:len(y_attack_train)]
            # Recompute properly using same split
            _, _, y_binary_train, _ = train_test_split(
                X, y_binary, test_size=0.2, random_state=42
            )

            logger.info(f"Training models on {len(X_train)} samples with {X_train.shape[1]} features...")

            # 7. Train the models (multi-class RF + binary unsupervised)
            self.scorer.fit_all(X_train, y_attack_train, y_binary_train, attack_label_map)
            
            # 8. Save the .joblib files
            self.scorer.save_models()
            logger.info("Training complete! Models saved. Ready for deployment.")
    # ── replay mode ───────────────────────────────────────────────────────────

    def run_replay(self, conn_log: Path, ssl_log: Path):
        """
        Score a saved log file and print all alerts.
        Useful for testing the pipeline without a live network.
        """

        logger.info("Replay mode: loading %s", conn_log)
        conn_df = load_conn_dataframe(conn_log)
        ssl_df  = load_ssl_dataframe(ssl_log)
        merged  = merge_conn_ssl(conn_df, ssl_df)

        logger.info("Scoring %d flows…", len(merged))
        alerts_found = 0
        for _, row in merged.iterrows():
            flow = row.to_dict()
            vec  = self.extractor.extract(flow)
            alert = self.scorer.evaluate(flow, vec)
            if alert:
                alerts_found += 1
                print(f"\n{'='*60}")
                import json
                print(json.dumps(alert.to_dict(), indent=2))

        logger.info("Replay complete. %d / %d flows triggered alerts.",
                    alerts_found, len(merged))

def main():
    parser = argparse.ArgumentParser(description="SHIELD IDS")
    parser.add_argument("--mode", choices=["live", "train", "replay"],
                        default="live")
    
    parser.add_argument("--csv-path", default="data/NF-UQ-NIDS-v2.csv",
                        help="Path to the NetFlow CSV dataset for training")

    parser.add_argument("--conn-log",   default="models/data/conn.log")
    parser.add_argument("--ssl-log",    default="models/data/ssl.log")
    
    args = parser.parse_args()

    pipeline = SHIELDPipeline()
    pipeline.setup()

    if args.mode == "live":
       pipeline.run_live(Path(args.conn_log), Path(args.ssl_log))

    elif args.mode == "train":
        pipeline.run_train(Path(args.csv_path))

    elif args.mode == "replay":
        pipeline.run_replay(
            Path(args.conn_log),
            Path(args.ssl_log),
        )

if __name__ == "__main__":
    main()
