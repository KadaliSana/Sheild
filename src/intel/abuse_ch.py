"""
intel/abuse_ch.py
─────────────────
Fetches and parses the JA3 fingerprint blacklist from abuse.ch (SSLBL).
Updates local JA3_THREAT_DB in features/tls_fingerprint.py.
"""

import csv
import logging
import requests
import json
from pathlib import Path

logger = logging.getLogger(__name__)

JA3_BLACKLIST_URL = "https://sslbl.abuse.ch/blacklist/ja3_fingerprints.csv"
CACHE_PATH = Path("models/intel/ja3_blacklist.json")

def update_ja3_blacklist() -> dict:
    """
    Download the latest JA3 blacklist from abuse.ch and return as a dict.
    Caches to disk for offline use.
    """
    try:
        logger.info(f"Fetching JA3 blacklist from {JA3_BLACKLIST_URL}...")
        response = requests.get(JA3_BLACKLIST_URL, timeout=15)
        response.raise_for_status()
        
        # Parse CSV
        # Format: # Firstseen,JA3_hash,Threat
        lines = response.text.splitlines()
        reader = csv.reader(line for line in lines if not line.startswith('#'))
        
        blacklist = {}
        for row in reader:
            if len(row) >= 3:
                firstseen, ja3_hash, threat = row[0], row[1], row[2]
                blacklist[ja3_hash] = {
                    "malware": threat,
                    "severity": "critical",
                    "description": f"Abuse.ch SSLBL: {threat} (First seen: {firstseen})",
                    "source": "abuse.ch"
                }
        
        # Cache to disk
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "w") as f:
            json.dump(blacklist, f, indent=2)
            
        logger.info(f"Successfully updated JA3 blacklist ({len(blacklist)} entries).")
        return blacklist

    except Exception as e:
        logger.error(f"Failed to update JA3 blacklist: {e}")
        # Try to load from cache
        if CACHE_PATH.exists():
            logger.info("Loading JA3 blacklist from cache...")
            with open(CACHE_PATH, "r") as f:
                return json.load(f)
        return {}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    update_ja3_blacklist()
