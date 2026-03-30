"""
scoring/risk_scorer.py
──────────────────────
Fuses the four detector scores into a single 0–100 risk score,
runs SHAP for explainability, performs TLS fingerprint analysis,
and produces a structured Alert.
Updated for NF-UQ-NIDS NetFlow architecture.
"""

import time
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from config.settings import (
    ENSEMBLE_WEIGHTS, ALERT_THRESHOLD,
    BLOCK_THRESHOLD, JA3_BLOCKLIST,
)
from models.detectors import (
    IsolationForestDetector,
    RandomForestDetector,
    TransformerDetector,
    StatisticalDetector,
)
from features.tls_fingerprint import TLSFingerprintEngine

logger = logging.getLogger(__name__)

# ── Multi-class Attack Labels (from NF-UQ-NIDS-v2 'Attack' column) ────────────
# These are populated at training time via LabelEncoder. The static fallback
# below covers the known categories in alphabetical order (sklearn default).
ATTACK_LABELS = {
    0: "Analysis",
    1: "Backdoor",
    2: "Benign",
    3: "Bot",
    4: "Brute Force",
    5: "DDoS",
    6: "DoS",
    7: "Exploits",
    8: "Fuzzers",
    9: "Generic",
    10: "Infilteration",
    11: "Reconnaissance",
    12: "Shellcode",
    13: "Theft",
    14: "Worms",
    15: "Injection",
    16: "Man-in-the-Middle",
    17: "Password Attack",
    18: "Ransomware",
    19: "Port Scan",
    20: "XSS",
}

# ── severity bands ────────────────────────────────────────────────────────────
def _severity(score: int) -> str:
    if score >= 85: return "critical"
    if score >= 70: return "high"
    if score >= ALERT_THRESHOLD: return "medium"
    return "low"

# ── Alert dataclass ───────────────────────────────────────────────────────────

@dataclass
class Alert:
    uid:         str
    timestamp:   float
    src_ip:      str
    dst_ip:      str
    src_port:    int
    dst_port:    int
    proto:       str

    risk_score:  int                       # 0-100
    severity:    str                       # low/medium/high/critical
    attack_type: str                       # from ATTACK_LABELS
    should_block: bool

    # per-detector contributions
    scores: dict[str, float] = field(default_factory=dict)

    # explainability
    shap_values:     list[tuple[str, float]] = field(default_factory=list)  # top-5
    plain_language:  str = ""

    # TLS Threat Intel
    ja3_hash:       Optional[str] = None
    ja3s_hash:      Optional[str] = None
    ja3_blocked:    bool = False
    tls_version:    Optional[str] = None
    tls_risk_score: float = 0.0
    tls_risk_factors: list = field(default_factory=list)
    tls_threat_name: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "uid":          self.uid,
            "timestamp":    self.timestamp,
            "src":          f"{self.src_ip}:{self.src_port}",
            "dst":          f"{self.dst_ip}:{self.dst_port}",
            "proto":        self.proto,
            "risk_score":   self.risk_score,
            "severity":     self.severity,
            "attack_type":  self.attack_type,
            "should_block": self.should_block,
            "detector_scores": self.scores,
            "top_features": self.shap_values,
            "explanation":  self.plain_language,
            "ja3":          self.ja3_hash,
            "ja3s":         self.ja3s_hash,
            "ja3_blocked":  self.ja3_blocked,
            "tls_version":  self.tls_version,
            "tls_risk_score": round(self.tls_risk_score, 3),
            "tls_risk_factors": self.tls_risk_factors,
            "tls_threat":   self.tls_threat_name,
        }

# ── plain-language templates ──────────────────────────────────────────────────

def _plain_language(alert: Alert) -> str:
    if alert.ja3_blocked and alert.tls_threat_name:
        return (
            f"CRITICAL: Device {alert.src_ip} used a TLS fingerprint (JA3) "
            f"matching known malware '{alert.tls_threat_name}'. "
            f"Connection to {alert.dst_ip} was flagged for immediate blocking."
        )
    
    if alert.ja3_blocked:
        return (
            f"CRITICAL: Device {alert.src_ip} used an encrypted connection "
            f"(JA3) identical to known malware. Blocked immediately."
        )
    
    if alert.tls_risk_factors:
        tls_detail = "; ".join(alert.tls_risk_factors[:2])
        if alert.risk_score > 85:
            return (
                f"High-confidence threat from {alert.src_ip} → {alert.dst_ip}. "
                f"TLS analysis: {tls_detail}. "
                f"Network behavior strongly indicates an automated attack."
            )
        return (
            f"Suspicious activity from {alert.src_ip} → {alert.dst_ip} "
            f"(Risk: {alert.risk_score}). TLS indicators: {tls_detail}."
        )
    
    if alert.risk_score > 85:
        return (
            f"High-confidence anomaly detected from {alert.src_ip} to "
            f"{alert.dst_ip}. Traffic volume and timing strongly indicate "
            f"an automated attack."
        )
        
    return (
        f"Suspicious network behavior detected from {alert.src_ip} "
        f"to {alert.dst_ip} (Risk Score: {alert.risk_score})."
    )

# ── main scorer ───────────────────────────────────────────────────────────────

class RiskScorer:
    """
    Orchestrates the ML detectors and TLS fingerprint engine,
    then produces Alert objects with full explainability.
    """

    def __init__(self):
        self._iso  = IsolationForestDetector()
        self._rf   = RandomForestDetector()
        self._transformer = TransformerDetector()
        self._stat = StatisticalDetector()
        
        # Use weights from config (it's already a dict)
        self._weights = ENSEMBLE_WEIGHTS
        self._shap_explainer = None
        
        # Multi-class label map (populated from training or model load)
        self._attack_label_map: dict[int, str] = {}
        
        # TLS Fingerprinting Engine
        self._tls_engine = TLSFingerprintEngine()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def load_models(self):
        for det in (self._iso, self._rf, self._transformer, self._stat):
            try:
                det.load()
            except Exception as exc:
                logger.warning("Could not load %s: %s", det.name, exc)
        # Restore the attack label map from the RF model bundle
        if self._rf._attack_label_map:
            self._attack_label_map = self._rf._attack_label_map
            logger.info("Loaded %d attack categories from RF model", len(self._attack_label_map))

    def save_models(self):
        for det in (self._iso, self._rf, self._transformer, self._stat):
            try:
                det.save()
            except Exception as exc:
                logger.warning("Could not save %s: %s", det.name, exc)

    def fit_all(self, X: np.ndarray, y_attack: np.ndarray, 
                y_binary: np.ndarray = None, attack_label_map: dict = None):
        """
        Train all models.
        - y_attack: multi-class integer labels (from LabelEncoder on 'Attack' column)
        - y_binary: binary 0/1 labels for unsupervised anomaly detectors
        - attack_label_map: {int: str} mapping class index → attack name
        """
        if y_binary is None:
            # Fallback: treat 0 class as benign
            y_binary = (y_attack != 0).astype(int)
        
        if attack_label_map:
            self._attack_label_map = attack_label_map
            self._rf._attack_label_map = attack_label_map  # persist with the model
        
        logger.info("Splitting dataset for Unsupervised vs Supervised training...")
        
        # Isolate Benign traffic (binary Label == 0) for the anomaly detectors
        normal_mask = (y_binary == 0)
        X_normal = X[normal_mask]

        logger.info(f"Training Anomaly Detectors on {len(X_normal)} benign samples...")
        self._iso.fit(X_normal)
        self._stat.fit(X_normal)
        
        logger.info(f"Training Random Forest (multi-class) on {len(X)} samples, {len(set(y_attack))} classes...")
        self._rf.fit(X, y_attack)
        
        logger.info("All models trained successfully.")

    # ── evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, flow: dict, feature_vec: np.ndarray) -> Optional[Alert]:
        
        # 1. Individual detector scores (0.0–1.0)
        scores = {
            "isolation_forest": self._iso.score(feature_vec),
            "random_forest":    self._rf.score(feature_vec),
            "transformer":      self._transformer.score(feature_vec),
            "statistical":      self._stat.score(feature_vec),
        }

        # 2. Weighted fusion
        fused = sum(
            self._weights.get(name, 0) * val
            for name, val in scores.items()
        )
        risk_score = int(np.clip(fused * 100, 0, 100))

        # 3. TLS Fingerprint Analysis
        tls_fp = self._tls_engine.analyze(flow)
        
        ja3_blocked = tls_fp.is_malicious
        if tls_fp.tls_risk_score > 0:
            # Blend TLS risk into the overall score (max 30 point boost)
            tls_boost = int(tls_fp.tls_risk_score * 30)
            risk_score = min(100, risk_score + tls_boost)
        
        if ja3_blocked:
            risk_score = 100  # Instant max risk for known malware JA3

        # Also check the legacy JA3 field for backward compat
        legacy_ja3 = flow.get("JA3_HASH", "")
        if legacy_ja3 and legacy_ja3 in JA3_BLOCKLIST and not ja3_blocked:
            ja3_blocked = True
            risk_score = 100

        # 4. Attack Classification (multi-class)
        atk_class = self._rf.predict_class(feature_vec)
        # Prefer the dynamically loaded label map, fall back to static dict
        label_map = self._attack_label_map if self._attack_label_map else ATTACK_LABELS
        attack_type = label_map.get(atk_class, "Unknown Anomaly")
        
        # If RF says Benign but ensemble score is high, use the next-best class
        # This handles domain-shift between training data and live Zeek features
        if attack_type == "Benign" and risk_score >= ALERT_THRESHOLD:
            alt_class = self._rf.predict_top_nonbenign(feature_vec)
            attack_type = label_map.get(alt_class, "Anomalous Flow")

        # 5. SHAP Explainability
        shap_values = self._rf.top_features(n=3)

        # 6. Build the Alert (Mapped to NetFlow Keys)
        ts_val = flow.get("ts", time.time())
        if hasattr(ts_val, "timestamp"):
            timestamp = float(ts_val.timestamp())
        else:
            timestamp = float(ts_val)

        alert = Alert(
            uid         = str(flow.get("uid", time.time())),
            timestamp   = timestamp,
            src_ip      = str(flow.get("IPV4_SRC_ADDR") or flow.get("id.orig_h", "Unknown")),
            dst_ip      = str(flow.get("IPV4_DST_ADDR") or flow.get("id.resp_h", "Unknown")),
            src_port    = int(flow.get("L4_SRC_PORT") or flow.get("id.orig_p", 0)),
            dst_port    = int(flow.get("L4_DST_PORT") or flow.get("id.resp_p", 0)),
            proto       = str(flow.get("PROTOCOL") or flow.get("proto", "")),
            risk_score  = risk_score,
            severity    = _severity(risk_score),
            attack_type = attack_type,
            should_block= risk_score >= BLOCK_THRESHOLD,
            scores      = {k: round(v, 3) for k, v in scores.items()},
            shap_values = shap_values,
            ja3_hash    = tls_fp.ja3_hash or (str(legacy_ja3) if legacy_ja3 else None),
            ja3s_hash   = tls_fp.ja3s_hash,
            ja3_blocked = ja3_blocked,
            tls_version = tls_fp.tls_version,
            tls_risk_score = tls_fp.tls_risk_score,
            tls_risk_factors = tls_fp.risk_factors,
            tls_threat_name = tls_fp.threat_name,
        )
        alert.plain_language = _plain_language(alert)
        return alert

    def get_tls_stats(self) -> dict:
        """Expose TLS fingerprinting statistics for the dashboard."""
        return self._tls_engine.get_stats()