"""
SHIELD — configuration and constants.
Edit these to match your deployment environment.
"""

from pathlib import Path

# ── Zeek log paths ────────────────────────────────────────────────────────────
ZEEK_LOG_DIR   = Path("/home/sana/IDS/src/models/data")   # live rotating logs
CONN_LOG       = ZEEK_LOG_DIR / "conn.log"
SSL_LOG        = ZEEK_LOG_DIR / "ssl.log"
DNS_LOG        = ZEEK_LOG_DIR / "dns.log"
WEIRD_LOG      = ZEEK_LOG_DIR / "weird.log"      # Zeek's own anomaly flag

# ── Network interface (used when launching Zeek from Python) ──────────────────
CAPTURE_IFACE  = "eth0"                          # change to wlan0 for Wi-Fi

# ── Feature extraction ────────────────────────────────────────────────────────
FLOW_WINDOW_SEC   = 30        # aggregate flows over this window
UID_CACHE_TTL_SEC = 120       # how long to keep unmatched uid entries
MIN_PKTS_THRESHOLD = 3        # ignore single-packet blips

# ── Risk scoring ──────────────────────────────────────────────────────────────
ALERT_THRESHOLD   = 60        # 0-100; fire alert above this
BLOCK_THRESHOLD   = 85        # auto-block above this (if enabled)

# Ensemble weights (must sum to 1.0)
ENSEMBLE_WEIGHTS = {
    "isolation_forest": 0.25,
    "random_forest":    0.35,
    "transformer":      0.25,
    "statistical":      0.15,
}

# ── Model artefacts ───────────────────────────────────────────────────────────
MODEL_DIR            = Path("models/artefacts")
RF_MODEL_PATH        = MODEL_DIR / "rf_classifier.joblib"
ISOFOREST_MODEL_PATH = MODEL_DIR / "isolation_forest.joblib"
SCALER_PATH          = MODEL_DIR / "scaler.joblib"
TRANSFORMER_MODEL_PATH = MODEL_DIR / "transformer_autoencoder.pt"

# ── JA3 threat-intel blocklist (hashes of known-malicious TLS fingerprints) ──
# Source: https://sslbl.abuse.ch/blacklist/ja3_fingerprints.csv
# Last updated: 2026-03-31
JA3_BLOCKLIST: set[str] = {
    # Dridex
    "b386946a5a44d1ddcc843bc75336dfce",
    "cb98a24ee4b9134448ffb5714fd870ac",
    "d6f04b5a910115f4b50ecec09d40a1df",
    "51c64c77e60f3980eea90869b68c58a8",
    # TrickBot
    "1aa7bf8b97e540ca5edd75f7b8384bfa",
    "8f52d1ce303fb4a6515836aec3cc16b1",
    "c50f6a8b9173676b47ba6085bd0c6cee",
    "534ce2dbc413c68e908363b5df0ae5e0",
    "fb00055a1196aeea8d1bc609885ba953",
    "8916410db85077a5460817142dcbc8de",
    "e62a5f4d538cbf169c2af71bec2399b4",
    "f735bbc6b69723b9df7b0e7ef27872af",
    "49ed2ef3f1321e5f044f1e71b0e6fdd5",
    # Adware
    "8991a387e4cc841740f25d6f5139f92d",
    "3d89c0dfb1fa44911b8fa7523ef8dedb",
    "bc6c386f480ee97b9d9e52d472b772d8",
    "e330bca99c8a5256ae126a55c4c725c5",
    "d551fafc4f40f1dec2bb45980bfa9492",
    "83e04bc58d402f9633983cbf22724b02",
    "b8f81673c0e1d29908346f3bab892b9b",
    "9c2589e1c0e9f533a022c6205f9719e1",
    "849b04bdbd1d2b983f6e8a457e0632a8",
    "16efcf0e00504ddfedde13bfea997952",
    "550dce18de1bb143e69d6dd9413b8355",
    "098f55e27d8c4b0a590102cbdb3a5f3a",
    "29085f03f8e8a03f0b399c5c7cf0b0b8",
    "46efd49abcca8ea9baa932da68fdb529",
    "5e573c9c9f8ba720ef9b18e9fce2e2f7",
    "f6fd83a21f9f3c5f9ff7b5c63bbc179d",
    "92579701f145605e9edc0b01a901c6d5",
    "b2b61db7b9490a60d270ccb20b462826",
    "b13d01846ad7a14a70bf030a16775c78",
    "698e36219f3979420fa2581b21dac7ec",
    "93d056782d649deb51cda44ecb714bb0",
    "2092e1fffb45d7e4a19a57f9bc5e203a",
    "fb58831f892190644fe44e25bc830b45",
    "7691297bcb20a41233fd0a0baa0a3628",
    # Tofsee
    "35c0a31c481927f022a3b530255ac080",
    "70722097d1fe1d78d8c2164640ab6df4",
    "4d7a28d6f2263ed61de88ca66eb011e3",
    "590a232d04d56409fab72e752a8a2634",
    "96eba628dcb2b47607192ba74a3b55ba",
    "df5c30e670dba99f9270ed36060cf054",
    "d7150af4514b868defb854db0f62a441",
    "03e186a7f83285e93341de478334006e",
    "a50a861119aceb0ccc74902e8fddb618",
    "e7643725fcff971e3051fe0e47fc2c71",
    "7c410ce832e848a3321432c9a82e972b",
    "da949afd9bd6df820730f8f171584a71",
    "906004246f3ba5e755b043c057254a29",
    "fd80fa9c6120cdeea8520510f3c644ac",
    "b90bdbe961a648f0427db21aaa6ccb59",
    "1fe4c7a3544eb27afec2adfb3a3dbf60",
    "9f62c4f26b90d3d757bea609e82f2eaf",
    "a61299f9b501adcf680b9275d79d4ac6",
    "7dcce5b76c8b17472d024758970a406b",
    "1543a7c46633acf71e8401baccbd0568",
    "1d095e68489d3c535297cd8dffb06cb9",
    "e3b2ab1f9a56f2fb4c9248f2f41631fa",
    "dff8a0aa1c904aaea76c5bf624e88333",
    "17fd49722f8d11f3d76dce84f8e099a7",
    "911479ac8a0813ed1241b3686ccdade9",
    "c5deb9465d47232dd48772f9c4d14679",
    "f22bdd57e3a52de86cda40da2d84e83b",
    "d18a4da84af59e1108862a39bae7c9d4",
    "1aee0238942d453d679fc1e37a303387",
    "bffa4501966196d3d6e90cee1f88fc89",
    "807fca46d9d0cf63adf4e5e80e414bbe",
    "0cc1e84568e471aa1d62ad4158ade6b5",
    "8f6c918dcb585ebbea05e2cc94530e3d",
    "34f14a69ad7009ca5863379218af17f3",
    "c2b4710c6888a5d47befe865c8e6fb19",
    "08a8a4e85b25ac42e1490bc85cfdb5ce",
    "c0220cd64849a629397a9cb68f78a0ea",
    "52c7396a501e4fecbdfa99c5408334ac",
    "70a04365be5bbd4653698bebeb43ce68",
    "d81d654effb94714a4086734fa0adad9",
    "25d74b7b4b779eb1efd4b31d26d651c6",
    "fc2299d5b2964cd242c5a2c8c531a5f0",
    "32926ca3e59f0413d0b98725454594f5",
    "ffefafdb86336d057eda5fdf02b3d5ce",
    "d76ee64fb7273733cbe455ac81c292e6",
    # Quakbot
    "3cda52da4ade09f1f781ad2e82dcfa20",
    "7dd50e112cd23734a310b90f6f44a7cd",
    # Gozi
    "c201b92f8b483fa388be174d6689f534",
    "57f3642b4e37e28f5cbe3020c9331b4c",
    # JBifrost
    "51a7ad14509fd614c7bb3a50c4982b8c",
    # TorrentLocker
    "1712287800ac91b34cadd5884ce85568",
    # Ransomware
    "2d8794cb7b52b777bee2695e79c15760",
    "1be3ecebe5aa9d3654e6e703d81f6928",
    # CoinMiner
    "40adfd923eb82b89d8836ba37a19bca1",
    # AsyncRAT
    "fc54e0d16d9764783542f0146a98b300",
    # Gootkit
    "c5235d3a8b9934b7fbbd204d50bc058d",
    # Adwind
    "d2935c58fe676744fecc8614ee5356c7",
    "decfb48a53789ebe081b88aabb58ee34",
    # BitRAT
    "8515076cbbca9dce33151b798f782456",
    # Emotet
    "e7d705a3286e19ea42f587b344ee6865",
    "6734f37431670b3ab4292b8f60f29984",
    # Unknown/Generic
    "7a29c223fb122ec64d10f0a159e07996",
}

# JA3 hash → malware family name (for dashboard display)
JA3_THREAT_NAMES: dict[str, str] = {
    "b386946a5a44d1ddcc843bc75336dfce": "Dridex",
    "cb98a24ee4b9134448ffb5714fd870ac": "Dridex",
    "d6f04b5a910115f4b50ecec09d40a1df": "Dridex",
    "51c64c77e60f3980eea90869b68c58a8": "Dridex",
    "1aa7bf8b97e540ca5edd75f7b8384bfa": "TrickBot",
    "8f52d1ce303fb4a6515836aec3cc16b1": "TrickBot",
    "c50f6a8b9173676b47ba6085bd0c6cee": "TrickBot",
    "534ce2dbc413c68e908363b5df0ae5e0": "TrickBot",
    "fb00055a1196aeea8d1bc609885ba953": "TrickBot",
    "8916410db85077a5460817142dcbc8de": "TrickBot",
    "e62a5f4d538cbf169c2af71bec2399b4": "TrickBot",
    "f735bbc6b69723b9df7b0e7ef27872af": "TrickBot",
    "49ed2ef3f1321e5f044f1e71b0e6fdd5": "TrickBot",
    "e7d705a3286e19ea42f587b344ee6865": "Emotet",
    "6734f37431670b3ab4292b8f60f29984": "Emotet",
    "3cda52da4ade09f1f781ad2e82dcfa20": "Quakbot",
    "7dd50e112cd23734a310b90f6f44a7cd": "Quakbot",
    "c201b92f8b483fa388be174d6689f534": "Gozi",
    "57f3642b4e37e28f5cbe3020c9331b4c": "Gozi",
    "51a7ad14509fd614c7bb3a50c4982b8c": "JBifrost",
    "1712287800ac91b34cadd5884ce85568": "TorrentLocker",
    "2d8794cb7b52b777bee2695e79c15760": "Ransomware",
    "1be3ecebe5aa9d3654e6e703d81f6928": "Ransomware.Troldesh",
    "40adfd923eb82b89d8836ba37a19bca1": "CoinMiner",
    "fc54e0d16d9764783542f0146a98b300": "AsyncRAT",
    "c5235d3a8b9934b7fbbd204d50bc058d": "Gootkit",
    "d2935c58fe676744fecc8614ee5356c7": "Adwind",
    "decfb48a53789ebe081b88aabb58ee34": "Adwind",
    "8515076cbbca9dce33151b798f782456": "BitRAT",
}

# ── Automated response ────────────────────────────────────────────────────────
AUTO_BLOCK_ENABLED  = True    # set True to enable iptables auto-block
BLOCK_DURATION_SECS = 1800    # 30 min auto-expiry

# ── API server ────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
