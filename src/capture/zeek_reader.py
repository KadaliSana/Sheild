import time
import threading
import logging
from collections import defaultdict
from typing import Callable, Optional
from pathlib import Path

from zat.zeek_log_reader import ZeekLogReader

from config.settings import (
    CONN_LOG, SSL_LOG, DNS_LOG, WEIRD_LOG,
    UID_CACHE_TTL_SEC, MIN_PKTS_THRESHOLD,
)

logger = logging.getLogger(__name__)

class _CacheEntry:
    """Holds partial records while waiting for the matching log type."""

    def __init__(self, record: dict, source: str):
        self.records: dict[str, dict] = {source: record}
        self.first_seen: float = time.monotonic()
        self.last_seen: float = time.monotonic()

    def add(self, record: dict, source: str):
        self.records[source] = record
        self.last_seen = time.monotonic()

    def is_stale(self, ttl: float) -> bool:
        """True if the entry has been sitting too long without any updates."""
        return (time.monotonic() - self.last_seen) > ttl

    def is_ready(self, delay: float) -> bool:
        """True if we have the primary record and waited long enough for sides."""
        return "conn" in self.records and (time.monotonic() - self.first_seen) > delay


# ── main reader class ─────────────────────────────────────────────────────────

class ZeekFlowReader:
    """
    Reads conn.log + ssl.log (+ optionally dns.log / weird.log) in
    background threads, joins on uid, and calls `on_flow` with an
    enriched dict every time a complete record is available.

    Usage
    -----
        def handle_flow(flow: dict):
            print(flow)

        reader = ZeekFlowReader(on_flow=handle_flow)
        reader.start()          # non-blocking, runs in background threads
        ...
        reader.stop()
    """

    # fields we care about from each log type
    CONN_FIELDS = {
        "uid", "ts", "id.orig_h", "id.orig_p",
        "id.resp_h", "id.resp_p", "proto", "service",
        "duration", "orig_bytes", "resp_bytes",
        "orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes",
        "conn_state", "history", "missed_bytes",
    }

    SSL_FIELDS = {
        "uid", "version", "cipher", "curve",
        "server_name", "ja3", "ja3s",
        "resumed", "established", "cert_chain_fuids",
    }

    DNS_FIELDS = {
        "uid", "query", "qtype_name", "rcode_name",
        "answers", "TTLs",
    }

    WEIRD_FIELDS = {
        "uid", "name", "addl", "notice",
    }

    def __init__(
        self,
        on_flow: Callable[[dict], None],
        conn_log: Path = CONN_LOG,
        ssl_log: Path = SSL_LOG,
        dns_log: Optional[Path] = DNS_LOG,
        weird_log: Optional[Path] = WEIRD_LOG,
        uid_cache_ttl: float = UID_CACHE_TTL_SEC,
    ):
        self._on_flow     = on_flow
        self._conn_log    = conn_log
        self._ssl_log     = ssl_log
        self._dns_log     = dns_log
        self._weird_log   = weird_log
        self._ttl         = uid_cache_ttl

        # uid → _CacheEntry  (shared across all reader threads)
        self._cache: dict[str, _CacheEntry] = {}
        self._lock  = threading.Lock()

        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()

    # ── public API ────────────────────────────────────────────────────────────

    def start(self):
        """Spawn reader threads — returns immediately."""
        sources = [
            (self._conn_log,  "conn",  self.CONN_FIELDS),
            (self._ssl_log,   "ssl",   self.SSL_FIELDS),
        ]
        if self._dns_log:
            sources.append((self._dns_log,   "dns",   self.DNS_FIELDS))
        if self._weird_log:
            sources.append((self._weird_log, "weird", self.WEIRD_FIELDS))

        for path, name, fields in sources:
            t = threading.Thread(
                target=self._tail_log,
                args=(path, name, fields),
                name=f"zeek-{name}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        # cache GC thread
        gc = threading.Thread(target=self._gc_loop, name="zeek-gc", daemon=True)
        gc.start()
        self._threads.append(gc)

        logger.info("ZeekFlowReader started — tailing %d log files", len(sources))

    def stop(self):
        """Signal all threads to exit."""
        self._stop_event.set()
        logger.info("ZeekFlowReader stopping…")

    # ── internal ──────────────────────────────────────────────────────────────

    def _tail_log(self, path: Path, source: str, fields: set[str]):
        """Continuously tail a single Zeek log file."""
        logger.info("[%s] starting tail of %s", source, path)
        
        while not self._stop_event.is_set():
            try:
                reader = ZeekLogReader(str(path), tail=True)
                for raw_row in reader.readrows():
                    if self._stop_event.is_set():
                        break

                    # filter to only the fields we need
                    row = {k: v for k, v in raw_row.items() if k in fields}
                    uid = row.get("uid")
                    if not uid:
                        continue

                    # skip noise
                    if source == "conn":
                        pkts = (row.get("orig_pkts") or 0) + (row.get("resp_pkts") or 0)
                        if pkts < MIN_PKTS_THRESHOLD:
                            continue

                    self._ingest(uid, row, source)
                
                # If reader gracefully exits (e.g. file truncated), sleep before retry
                time.sleep(2)
            except Exception as exc:
                if not getattr(self, '_notified_'+source, False):
                    logger.warning("[%s] Waiting for valid TSV log file at %s", source, path)
                    setattr(self, '_notified_'+source, True)
                time.sleep(2)

    def _ingest(self, uid: str, row: dict, source: str):
        """Add a record to the cache; emit immediately if both conn and ssl are ready."""
        flow_to_emit = None
        with self._lock:
            if uid not in self._cache:
                self._cache[uid] = _CacheEntry(row, source)
            else:
                self._cache[uid].add(row, source)

            entry = self._cache[uid]
            # Fast-path: if we have both, no need to wait
            if "conn" in entry.records and "ssl" in entry.records:
                flow_to_emit = self._build_flow(entry)
                del self._cache[uid]

        if flow_to_emit:
            self._emit(flow_to_emit)

    def _build_flow(self, entry: "_CacheEntry") -> dict:
        """Merge all partial records into a single flat flow dict."""
        flow: dict = {}
        # conn is the base
        flow.update(entry.records.get("conn", {}))

        # ssl — prefix with ssl_ to avoid key collisions
        for k, v in entry.records.get("ssl", {}).items():
            if k != "uid":
                flow[f"ssl_{k}"] = v

        # dns — prefix with dns_
        for k, v in entry.records.get("dns", {}).items():
            if k != "uid":
                flow[f"dns_{k}"] = v

        # weird — prefix with weird_
        for k, v in entry.records.get("weird", {}).items():
            if k != "uid":
                flow[f"weird_{k}"] = v

        flow["_ingested_at"] = time.time()
        return flow

    def _emit(self, flow: dict):
        try:
            self._on_flow(flow)
        except Exception as exc:
            logger.error("on_flow callback raised: %s", exc, exc_info=True)

    def _gc_loop(self):
        """
        Periodically:
        1. Evict and emit conn-only flows after a short coalescing delay (2s).
        2. Evict very stale entries (30s) that never got a conn record.
        """
        coalesce_delay = 2.0  # seconds to wait for sidecar logs
        while not self._stop_event.wait(timeout=1.0):
            flows_to_emit = []
            with self._lock:
                to_remove = []
                for uid, entry in self._cache.items():
                    # Case 1: conn is ready and we've waited enough for ssl/dns
                    if entry.is_ready(coalesce_delay):
                        flows_to_emit.append(self._build_flow(entry))
                        to_remove.append(uid)
                    # Case 2: stale (no update for 30s) - just drop/cleanup
                    elif entry.is_stale(self._ttl):
                        to_remove.append(uid)

                for uid in to_remove:
                    self._cache.pop(uid, None)

            for flow in flows_to_emit:
                self._emit(flow)

            if flows_to_emit:
                logger.debug("GC emitted %d coalesced flows", len(flows_to_emit))


# ── convenience: batch loader (for training / offline analysis) ───────────────

def load_conn_dataframe(conn_log_path: Path):
    """
    Load a *completed* conn.log into a pandas DataFrame.
    Useful for training the ML models offline.

        df = load_conn_dataframe(Path('/var/log/zeek/conn.log.gz'))
    """
    from zat.log_to_dataframe import LogToDataFrame
    loader = LogToDataFrame()
    df = loader.create_dataframe(str(conn_log_path))
    logger.info("Loaded %d conn records from %s", len(df), conn_log_path)
    return df


def load_ssl_dataframe(ssl_log_path: Path):
    """Load a completed ssl.log into a pandas DataFrame."""
    from zat.log_to_dataframe import LogToDataFrame
    loader = LogToDataFrame()
    df = loader.create_dataframe(str(ssl_log_path))
    logger.info("Loaded %d ssl records from %s", len(df), ssl_log_path)
    return df


def merge_conn_ssl(conn_df, ssl_df):
    """
    Join conn and ssl DataFrames on uid.
    Returns a merged DataFrame with ssl_ prefixed columns.
    """
    ssl_renamed = ssl_df.add_prefix("ssl_").rename(columns={"ssl_uid": "uid"})
    merged = conn_df.merge(ssl_renamed, on="uid", how="left")
    logger.info("Merged DataFrame: %d rows, %d columns", len(merged), len(merged.columns))
    return merged
