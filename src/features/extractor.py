import math
import logging
import numpy as np

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "PROTOCOL", "L7_PROTO", "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
    "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
    "MIN_TTL", "MAX_TTL", "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN", "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES", "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
    "ICMP_TYPE", "ICMP_IPV4_TYPE", "DNS_QUERY_ID", "DNS_QUERY_TYPE",
    "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE"
]

assert len(FEATURE_NAMES) == 39, "Feature name list must have exactly 39 entries"

# ── Mapping Dictionaries ──────────────────────────────────────────────────────
_PROTO_MAP = {"icmp": 1, "tcp": 6, "udp": 17}

# Approximate nDPI mappings for common L7 protocols found in NF datasets
_L7_MAP = {
    "http": 7.0, "ssl": 91.0, "tls": 91.0, 
    "dns": 5.0, "ssh": 22.0, "ftp": 1.0
}

def _safe_float(val, default=0.0) -> float:
    try:
        v = float(val)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default

def _parse_flags(history: str, side: str = "both") -> int:
    """
    Converts Zeek connection history into standard TCP Flag integers.
    FIN=1, SYN=2, RST=4, PSH=8, ACK=16, URG=32
    """
    if not history: return 0
    
    if side == "client":
        chars = [c for c in history if c.isupper()]
    elif side == "server":
        chars = [c for c in history if c.islower()]
    else:
        chars = [c.upper() for c in history]

    h_str = "".join(chars).upper()
    flags = 0
    if 'F' in h_str: flags += 1
    if 'S' in h_str: flags += 2
    if 'R' in h_str: flags += 4
    if 'P' in h_str or 'D' in h_str: flags += 8 # D = payload data in Zeek
    if 'A' in h_str: flags += 16
    if 'U' in h_str: flags += 32
    return flags


# ── Main Extractor ────────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Stateless converter: Zeek flow dict → numpy float32 vector of length 39.
    """

    def extract(self, flow: dict) -> np.ndarray:
        vec = np.zeros(39, dtype=np.float32)

        try:
            # 1. Protocols
            proto_str = str(flow.get("proto", "")).lower()
            service_str = str(flow.get("service") or "").lower()
            
            vec[0] = _PROTO_MAP.get(proto_str, 0)                   # PROTOCOL
            vec[1] = _L7_MAP.get(service_str, 0.0)                  # L7_PROTO

            # 2. Volume (Bytes and Packets)
            in_bytes = _safe_float(flow.get("orig_ip_bytes") or flow.get("orig_bytes"))
            in_pkts = _safe_float(flow.get("orig_pkts"))
            out_bytes = _safe_float(flow.get("resp_ip_bytes") or flow.get("resp_bytes"))
            out_pkts = _safe_float(flow.get("resp_pkts"))

            vec[2] = in_bytes                                       # IN_BYTES
            vec[3] = in_pkts                                        # IN_PKTS
            vec[4] = out_bytes                                      # OUT_BYTES
            vec[5] = out_pkts                                       # OUT_PKTS

            # 3. TCP Flags
            history = str(flow.get("history", ""))
            vec[6] = _parse_flags(history, "both")                  # TCP_FLAGS
            vec[7] = _parse_flags(history, "client")                # CLIENT_TCP_FLAGS
            vec[8] = _parse_flags(history, "server")                # SERVER_TCP_FLAGS

            # 4. Timing
            duration = _safe_float(flow.get("duration"), 0.0)
            dur_ms = duration * 1000.0
            
            vec[9] = dur_ms                                         # FLOW_DURATION_MILLISECONDS
            vec[10] = dur_ms / 2.0 if in_pkts > 0 else 0            # DURATION_IN (Approx)
            vec[11] = dur_ms / 2.0 if out_pkts > 0 else 0           # DURATION_OUT (Approx)

            # 5. Packet Specs (Approximated for baseline continuity)
            vec[12] = 64.0 if in_pkts > 0 else 0.0                  # MIN_TTL
            vec[13] = 64.0 if in_pkts > 0 else 0.0                  # MAX_TTL
            
            avg_in_len = in_bytes / in_pkts if in_pkts > 0 else 0
            avg_out_len = out_bytes / out_pkts if out_pkts > 0 else 0
            max_len = max(avg_in_len, avg_out_len) * 1.5            # Rough approximation
            
            vec[14] = max_len                                       # LONGEST_FLOW_PKT
            vec[15] = 40.0 if in_pkts > 0 else 0.0                  # SHORTEST_FLOW_PKT
            vec[16] = 40.0 if in_pkts > 0 else 0.0                  # MIN_IP_PKT_LEN
            vec[17] = max_len                                       # MAX_IP_PKT_LEN

            # 6. Throughput Rates
            safe_dur = max(duration, 0.001)
            vec[18] = in_bytes / safe_dur                           # SRC_TO_DST_SECOND_BYTES
            vec[19] = out_bytes / safe_dur                          # DST_TO_SRC_SECOND_BYTES
            
            vec[24] = (in_bytes * 8) / safe_dur                     # SRC_TO_DST_AVG_THROUGHPUT
            vec[25] = (out_bytes * 8) / safe_dur                    # DST_TO_SRC_AVG_THROUGHPUT

            # 7. Retransmissions (Derived from Zeek history 'c' or 't')
            retrans = history.count('c') + history.count('t') + history.count('C') + history.count('T')
            vec[21] = retrans                                       # RETRANSMITTED_IN_PKTS
            vec[23] = retrans                                       # RETRANSMITTED_OUT_PKTS

            # 8. Packet Bins (Heuristic distribution)
            total_pkts = in_pkts + out_pkts
            if avg_in_len < 128:
                vec[26] = total_pkts                                # NUM_PKTS_UP_TO_128_BYTES
            elif avg_in_len < 256:
                vec[27] = total_pkts                                # NUM_PKTS_128_TO_256_BYTES
            elif avg_in_len < 512:
                vec[28] = total_pkts                                # NUM_PKTS_256_TO_512_BYTES
            elif avg_in_len < 1024:
                vec[29] = total_pkts                                # NUM_PKTS_512_TO_1024_BYTES
            else:
                vec[30] = total_pkts                                # NUM_PKTS_1024_TO_1514_BYTES

            # The rest of the indices [31 to 38] safely default to 0.0
            # which aligns with standard NF-UQ-NIDS missing values (ICMP, DNS, FTP)

        except Exception as exc:
            logger.warning(f"Feature extraction error: {exc}")

        return vec
        
def extract_dataframe(df):
    """
    Apply feature extraction to every row of a pandas DataFrame.
    """
    import pandas as pd
    extractor = FeatureExtractor()
    rows = [extractor.extract(row.to_dict()) for _, row in df.iterrows()]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)