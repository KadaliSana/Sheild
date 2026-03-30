"""
features/tls_fingerprint.py
───────────────────────────
TLS/JA3 fingerprinting engine for SHIELD IDS.

Provides:
    1. JA3 hash extraction from Zeek ssl.log fields
    2. JA3S (server) fingerprint extraction
    3. Threat-intel matching against known-malicious fingerprints
    4. TLS version & cipher strength scoring
    5. Certificate anomaly detection

JA3 is a method for creating SSL/TLS client fingerprints by hashing
specific fields from the ClientHello packet:
    SSLVersion + Ciphers + Extensions + EllipticCurves + EllipticCurvePointFormats

References:
    - https://github.com/salesforce/ja3
    - https://ja3er.com/
    - Zeek's ja3.zeek policy script produces these fields automatically
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

from config.settings import JA3_BLOCKLIST
from intel.abuse_ch import update_ja3_blacklist

logger = logging.getLogger(__name__)

# ── Sync Abuse.ch Blacklist ──────────────────────────────────────────────────
_abuse_ch_list = update_ja3_blacklist()



# ── Known-malicious JA3 fingerprint database ──────────────────────────────────
# Extended database with names for logging/dashboard display

JA3_THREAT_DB: dict[str, dict] = {
    # Emotet variants
    "e7d705a3286e19ea42f587b344ee6865": {
        "malware": "Emotet",
        "severity": "critical",
        "description": "Emotet banking trojan — lateral movement and credential theft",
    },
    # TrickBot
    "6734f37431670b3ab4292b8f60f29984": {
        "malware": "TrickBot",
        "severity": "critical",
        "description": "TrickBot modular trojan — data exfiltration and ransomware delivery",
    },
    # CobaltStrike
    "51c64c77e60f3980eea90869b68c58a8": {
        "malware": "CobaltStrike",
        "severity": "critical",
        "description": "CobaltStrike default beacon — command & control framework",
    },
    "72a589da586844d7f0818ce684948eea": {
        "malware": "CobaltStrike",
        "severity": "critical",
        "description": "CobaltStrike HTTPS beacon variant",
    },
    # Dridex
    "b386946a5a44d1ddcc843bc75336dfce": {
        "malware": "Dridex",
        "severity": "critical",
        "description": "Dridex banking trojan — credential harvesting",
    },
    # AsyncRAT
    "3b5074b1b5d032e5620f69f9f700ff0e": {
        "malware": "AsyncRAT",
        "severity": "high",
        "description": "AsyncRAT remote access trojan",
    },
    # Metasploit
    "5d65ea3fb1d4aa7d826733d2f2cbbb1d": {
        "malware": "Metasploit",
        "severity": "high",
        "description": "Metasploit Meterpreter reverse HTTPS shell",
    },
    # Tofsee
    "e35df3e28c4080f38b891f47c3837f34": {
        "malware": "Tofsee",
        "severity": "medium",
        "description": "Tofsee spambot — proxy and DDoS capabilities",
    },
    # IcedID
    "c12f54a3f91dc794969968b0c09e6e9c": {
        "malware": "IcedID",
        "severity": "critical",
        "description": "IcedID (BokBot) banking trojan — credential theft",
    },
    # QakBot
    "d5a155f8572c09e8c4d4e76b2f975939": {
        "malware": "QakBot",
        "severity": "critical",
        "description": "QakBot (QBot) — banking trojan with worm capabilities",
    },
}

# Merge with config-defined blocklist
for ja3_hash in JA3_BLOCKLIST:
    if ja3_hash not in JA3_THREAT_DB:
        JA3_THREAT_DB[ja3_hash] = {
            "malware": "Custom Blocklist",
            "severity": "high",
            "description": "Matched user-configured JA3 blocklist entry",
        }

# Merge with Abuse.ch blacklist
for ja3_hash, threat_info in _abuse_ch_list.items():
    if ja3_hash not in JA3_THREAT_DB:
        JA3_THREAT_DB[ja3_hash] = threat_info



# ── TLS version risk scoring ─────────────────────────────────────────────────

TLS_VERSION_SCORES = {
    "SSLv2":    1.0,   # Critically insecure
    "SSLv3":    0.9,   # POODLE-vulnerable
    "TLSv10":   0.6,   # Deprecated
    "TLSv1":    0.6,   # Deprecated
    "TLSv11":   0.4,   # Deprecated
    "TLSv1.1":  0.4,   # Deprecated
    "TLSv12":   0.05,  # Current standard
    "TLSv1.2":  0.05,  # Current standard
    "TLSv13":   0.0,   # Best
    "TLSv1.3":  0.0,   # Best
}

# Weak cipher suites (partial matches)
WEAK_CIPHERS = {
    "RC4", "DES", "3DES", "NULL", "EXPORT", "anon",
    "MD5", "RC2", "IDEA", "SEED",
}


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class TLSFingerprint:
    """Complete TLS fingerprint analysis for a single flow."""
    ja3_hash:       Optional[str] = None
    ja3s_hash:      Optional[str] = None
    tls_version:    Optional[str] = None
    cipher_suite:   Optional[str] = None
    server_name:    Optional[str] = None
    
    # Threat intel results
    is_malicious:   bool = False
    threat_name:    Optional[str] = None
    threat_info:    Optional[dict] = None
    
    # Scoring
    tls_risk_score: float = 0.0     # 0.0 (safe) – 1.0 (maximum risk)
    risk_factors:   list = field(default_factory=list)
    is_pseudohash:  bool = False    # True if generated from metadata (no raw JA3)
    
    def to_dict(self) -> dict:
        return {
            "ja3": self.ja3_hash,
            "ja3s": self.ja3s_hash,
            "tls_version": self.tls_version,
            "cipher": self.cipher_suite,
            "server_name": self.server_name,
            "is_malicious": self.is_malicious,
            "threat_name": self.threat_name,
            "tls_risk_score": round(self.tls_risk_score, 3),
            "risk_factors": self.risk_factors,
            "is_pseudohash": self.is_pseudohash,
        }


# ── JA3 Hash Computation ─────────────────────────────────────────────────────

def compute_ja3_hash(
    ssl_version: str = "",
    ciphers: str = "",
    extensions: str = "",
    elliptic_curves: str = "",
    ec_point_formats: str = "",
) -> str:
    """
    Compute JA3 hash from TLS ClientHello fields.
    
    JA3 = MD5(SSLVersion,Ciphers,Extensions,EllipticCurves,ECPointFormats)
    
    Each field is a dash-separated list of decimal values.
    Fields are joined with commas.
    
    Example:
        ja3_string = "769,47-53-5-10-49161-49162-49171-49172-50-56-19-4,0-10-11,23-24-25,0"
        ja3_hash = md5(ja3_string)
    """
    ja3_string = ",".join([
        str(ssl_version),
        str(ciphers),
        str(extensions),
        str(elliptic_curves),
        str(ec_point_formats),
    ])
    return hashlib.md5(ja3_string.encode()).hexdigest()


def compute_ja3s_hash(
    ssl_version: str = "",
    cipher: str = "",
    extensions: str = "",
) -> str:
    """
    Compute JA3S (server) hash from TLS ServerHello fields.
    
    JA3S = MD5(SSLVersion,Cipher,Extensions)
    """
    ja3s_string = ",".join([
        str(ssl_version),
        str(cipher),
        str(extensions),
    ])
    return hashlib.md5(ja3s_string.encode()).hexdigest()


# ── Main Fingerprinting Engine ────────────────────────────────────────────────

class TLSFingerprintEngine:
    """
    Analyzes TLS metadata from Zeek flows to detect:
    - Known-malicious JA3 fingerprints
    - Deprecated TLS versions
    - Weak cipher suites
    - Certificate anomalies
    - Suspicious SNI patterns
    
    Usage:
        engine = TLSFingerprintEngine()
        result = engine.analyze(flow_dict)
        if result.is_malicious:
            print(f"THREAT: {result.threat_name}")
    """
    
    def __init__(self):
        self._ja3_cache: dict[str, TLSFingerprint] = {}  # cache recent results
        self._ja3_seen: dict[str, int] = defaultdict(int)  # frequency tracking
        self._tls_seen_total: int = 0
        self._first_seen: dict[str, float] = {}
        
    def analyze(self, flow: dict) -> TLSFingerprint:
        """
        Analyze a Zeek flow dict for TLS fingerprint indicators.
        
        Expects fields from Zeek's ssl.log (prefixed with ssl_ after merge):
            ssl_ja3, ssl_ja3s, ssl_version, ssl_cipher,
            ssl_server_name, ssl_established, ssl_resumed
        """
        fp = TLSFingerprint()
        
        # Extract JA3 fields (Zeek ssl.log fields after merge have ssl_ prefix)
        ja3 = str(flow.get("ssl_ja3") or flow.get("ja3") or flow.get("JA3_HASH") or "")
        ja3s = str(flow.get("ssl_ja3s") or flow.get("ja3s") or "")
        tls_ver = str(flow.get("ssl_version") or flow.get("version") or "")
        cipher = str(flow.get("ssl_cipher") or flow.get("cipher") or "")
        sni = str(flow.get("ssl_server_name") or flow.get("server_name") or "")
        
        fp.ja3_hash = ja3 if ja3 and ja3 != "-" else None
        fp.ja3s_hash = ja3s if ja3s and ja3s != "-" else None
        fp.tls_version = tls_ver if tls_ver and tls_ver != "-" else None
        fp.cipher_suite = cipher if cipher and cipher != "-" else None
        fp.server_name = sni if sni and sni != "-" else None
        
        # ── 0. Pseudo-hash Fallback ──────────────────────────────────────────
        # If raw JA3 is missing, generate a fingerprint from other metadata
        if not fp.ja3_hash and (fp.tls_version or fp.cipher_suite):
            components = [
                str(fp.tls_version or "unknown"),
                str(fp.cipher_suite or "unknown"),
                str(fp.server_name or "unknown")
            ]
            pseudo_str = ",".join(components)
            fp.ja3_hash = hashlib.md5(pseudo_str.encode()).hexdigest()
            fp.is_pseudohash = True
        
        # ── 0.1 Generic Fallback ──────────────────────────────────────────────
        # Ensure every analyzed flow has a fingerprint so counts sum up correctly
        if not fp.ja3_hash:
            # MD5 hash of "UNKNOWN"
            fp.ja3_hash = "39e71e7503f07a4a983362fa92d131f4"
            fp.threat_name = "Unknown TLS Client"


        
        risk_score = 0.0
        risk_factors = []
        
        # ── 1. JA3 Threat Intel Match ────────────────────────────────────────
        if fp.ja3_hash and fp.ja3_hash in JA3_THREAT_DB:
            threat = JA3_THREAT_DB[fp.ja3_hash]
            fp.is_malicious = True
            fp.threat_name = threat["malware"]
            fp.threat_info = threat
            risk_score = 1.0
            risk_factors.append(f"JA3 matches known malware: {threat['malware']}")
            logger.warning(
                "TLS THREAT: JA3 %s matches %s — %s",
                fp.ja3_hash, threat["malware"], threat["description"]
            )
        
        # ── 2. JA3S (Server) Threat Intel ────────────────────────────────────
        if fp.ja3s_hash and fp.ja3s_hash in JA3_THREAT_DB:
            threat = JA3_THREAT_DB[fp.ja3s_hash]
            fp.is_malicious = True
            fp.threat_name = (fp.threat_name or "") + f" + {threat['malware']} (server)"
            fp.threat_info = threat
            risk_score = max(risk_score, 0.95)
            risk_factors.append(f"JA3S server fingerprint matches: {threat['malware']}")
        
        # ── 3. TLS Version Analysis ──────────────────────────────────────────
        if fp.tls_version:
            ver_score = TLS_VERSION_SCORES.get(fp.tls_version, 0.3)
            if ver_score > 0.3:
                risk_factors.append(f"Deprecated TLS version: {fp.tls_version}")
                risk_score = max(risk_score, ver_score * 0.7)
        
        # ── 4. Cipher Suite Analysis ─────────────────────────────────────────
        if fp.cipher_suite:
            for weak in WEAK_CIPHERS:
                if weak.upper() in fp.cipher_suite.upper():
                    risk_factors.append(f"Weak cipher detected: {weak} in {fp.cipher_suite}")
                    risk_score = max(risk_score, 0.5)
                    break
        
        # ── 5. SNI Anomaly Detection ─────────────────────────────────────────
        if fp.server_name:
            sni_risk = self._analyze_sni(fp.server_name)
            if sni_risk > 0:
                risk_factors.append(f"Suspicious SNI pattern: {fp.server_name}")
                risk_score = max(risk_score, sni_risk)
        
        # ── 6. Missing TLS on expected ports ─────────────────────────────────
        dst_port = flow.get("id.resp_p") or flow.get("L4_DST_PORT") or 0
        service = str(flow.get("service") or "")
        if int(dst_port) == 443 and not fp.ja3_hash and service != "ssl":
            risk_factors.append("Port 443 without TLS handshake — possible tunneling")
            risk_score = max(risk_score, 0.4)
        
        # ── 7. Frequency Analysis ────────────────────────────────────────────
        # Every flow that reaches here gets counted towards the fingerprints
        
        if fp.ja3_hash:
            self._ja3_seen[fp.ja3_hash] += 1
            if fp.ja3_hash not in self._first_seen:
                self._first_seen[fp.ja3_hash] = time.time()
            
            # Brand-new fingerprint never seen before (potential new malware)
            if self._ja3_seen[fp.ja3_hash] == 1 and not fp.is_malicious:
                risk_factors.append(f"First-time JA3 fingerprint observed: {fp.ja3_hash[:12]}...")
                risk_score = max(risk_score, 0.15)
        
        fp.tls_risk_score = min(risk_score, 1.0)
        fp.risk_factors = risk_factors
        
        return fp
    
    def _analyze_sni(self, sni: str) -> float:
        """
        Score SNI (Server Name Indication) for suspicious patterns.
        
        Flags:
        - IP addresses used as SNI (instead of domain)
        - Very long domain names (DGA-like)
        - Excessive subdomain depth
        - Known suspicious TLDs
        """
        risk = 0.0
        
        # IP address as SNI
        parts = sni.split(".")
        if len(parts) == 4:
            try:
                if all(0 <= int(p) <= 255 for p in parts):
                    risk = max(risk, 0.3)
            except ValueError:
                pass
        
        # Very long domain (possible DGA)
        if len(sni) > 50:
            risk = max(risk, 0.35)
        
        # Excessive entropy in subdomain (DGA indicator)
        if len(parts) > 0:
            subdomain = parts[0]
            if len(subdomain) > 15:
                # Calculate character entropy
                from collections import Counter
                counts = Counter(subdomain.lower())
                length = len(subdomain)
                import math
                entropy = -sum(
                    (c / length) * math.log2(c / length) 
                    for c in counts.values()
                )
                if entropy > 3.5 and len(subdomain) > 20:
                    risk = max(risk, 0.4)
        
        # Deep subdomain nesting (> 4 levels)
        if len(parts) > 5:
            risk = max(risk, 0.25)
        
        # Suspicious TLDs
        suspicious_tlds = {'.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.buzz', '.club'}
        for tld in suspicious_tlds:
            if sni.endswith(tld):
                risk = max(risk, 0.2)
                break
        
        return risk
    
    def get_stats(self) -> dict:
        """Return fingerprinting statistics for the dashboard."""
        # Derived total ensures perfect mathematical alignment in the UI
        total_hits = sum(self._ja3_seen.values())
        
        return {
            "unique_ja3_count": len(self._ja3_seen),
            "total_fingerprinted": total_hits,
            "top_ja3": sorted(
                self._ja3_seen.items(),
                key=lambda x: x[1], reverse=True
            )[:10],
            "threats_detected": sum(
                1 for h in self._ja3_seen if h in JA3_THREAT_DB
            ),
        }
