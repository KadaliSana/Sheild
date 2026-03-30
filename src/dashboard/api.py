"""
dashboard/api.py
────────────────
FastAPI server exposing:
    GET  /alerts          — recent alerts (JSON)
    GET  /stats           — live metrics
    GET  /traffic         — recent network flows (benign + malicious)
    GET  /traffic/stats   — aggregated traffic statistics
    GET  /blocked         — current block list
    POST /block/{ip}      — manual block
    POST /unblock/{ip}    — manual unblock
    WS   /ws/alerts       — live WebSocket feed for the dashboard

Run with:
    uvicorn dashboard.api:app --host 0.0.0.0 --port 8000
"""

import asyncio
import json
import time
import logging
from pathlib import Path as _Path
from collections import deque, defaultdict
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from response.auto_block import blocked_ips, manual_unblock, _block_ip, _is_already_blocked

logger = logging.getLogger(__name__)

app = FastAPI(title="SHIELD IDS API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── shared state (written by pipeline, read by API) ───────────────────────────

_recent_alerts: deque = deque(maxlen=200)   # Alert.to_dict() entries
_recent_flows:  deque = deque(maxlen=500)   # All flows (benign + malicious)
_stats: dict[str, Any] = {
    "flows_total":    0,
    "alerts_total":   0,
    "flows_per_sec":  0,
    "start_time":     time.time(),
    "last_update":    time.time(),
    "last_calc_time": time.time(),
    "last_calc_flows": 0,
    "bytes_in_total": 0,
    "bytes_out_total": 0,
    "benign_count":   0,
    "malicious_count": 0,
}
_proto_counts: dict[str, int] = defaultdict(int)
_ws_clients: list[WebSocket] = []


# ── called by the main pipeline ───────────────────────────────────────────────

def ingest_alert(alert_dict: dict):
    """Thread-safe: push a new alert into the API layer."""
    _recent_alerts.appendleft(alert_dict)
    if alert_dict.get("attack_type") != "Benign" and alert_dict.get("risk_score", 0) >= 20: 
        _stats["alerts_total"] += 1
    _stats["last_update"] = time.time()

def ingest_flow(flow_dict: dict):
    """Thread-safe: push every processed flow for the traffic table."""
    _recent_flows.appendleft(flow_dict)
    # Track aggregate stats
    _stats["bytes_in_total"] += flow_dict.get("bytes_in", 0)
    _stats["bytes_out_total"] += flow_dict.get("bytes_out", 0)
    proto = flow_dict.get("proto", "unknown")
    _proto_counts[proto] += 1
    if flow_dict.get("is_malicious", False):
        _stats["malicious_count"] += 1
    else:
        _stats["benign_count"] += 1

def increment_flow_counter():
    _stats["flows_total"] += 1


async def broadcast_alert(alert_dict: dict):
    """Push a new alert to all connected WebSocket clients."""
    dead = []
    payload = json.dumps({"type": "alert", "data": alert_dict})
    for ws in _ws_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)

async def broadcast_flow(flow_dict: dict):
    """Push a new flow to all connected WebSocket clients."""
    dead = []
    payload = json.dumps({"type": "flow", "data": flow_dict})
    for ws in _ws_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)


# ── HTTP endpoints ────────────────────────────────────────────────────────────

@app.get("/alerts")
async def get_alerts(limit: int = 50, severity: str = ""):
    alerts = list(_recent_alerts)
    if severity:
        alerts = [a for a in alerts if a.get("severity") == severity]
    return {"alerts": alerts[:limit], "total": _stats["alerts_total"]}


@app.get("/stats")
async def get_stats():
    now = time.time()
    uptime = int(now - _stats["start_time"])
    
    time_diff = now - _stats["last_calc_time"]
    if time_diff >= 1.0:
        flows_diff = _stats["flows_total"] - _stats["last_calc_flows"]
        _stats["flows_per_sec"] = int(flows_diff / time_diff)
        _stats["last_calc_time"] = now
        _stats["last_calc_flows"] = _stats["flows_total"]

    return {
        **_stats,
        "uptime_seconds": uptime,
        "blocked_count":  len(blocked_ips()),
    }


@app.get("/traffic")
async def get_traffic(limit: int = 100):
    """Return recent flows for the live traffic table."""
    flows = list(_recent_flows)[:limit]
    return {"flows": flows, "total": len(_recent_flows)}


@app.get("/traffic/stats")
async def get_traffic_stats():
    """Aggregated traffic statistics for dashboard widgets."""
    return {
        "bytes_in_total":   _stats["bytes_in_total"],
        "bytes_out_total":  _stats["bytes_out_total"],
        "benign_count":     _stats["benign_count"],
        "malicious_count":  _stats["malicious_count"],
        "protocol_breakdown": dict(_proto_counts),
    }


@app.get("/blocked")
async def get_blocked():
    return {"blocked": blocked_ips()}


@app.get("/tls/stats")
async def get_tls_stats():
    """TLS fingerprinting statistics for dashboard."""
    return _tls_stats


# ── shared TLS stats (updated by pipeline) ────────────────────────────────────
_tls_stats: dict = {
    "unique_ja3_count": 0,
    "total_fingerprinted": 0,
    "threats_detected": 0,
    "top_ja3": [],
}

def update_tls_stats(stats: dict):
    """Called by the pipeline to update TLS stats."""
    global _tls_stats
    _tls_stats = stats


@app.post("/block/{ip}")
async def block_ip_endpoint(ip: str):
    """Manually block an IP via the dashboard."""
    if _is_already_blocked(ip):
        return {"success": False, "reason": "already_blocked", "ip": ip}
    _block_ip(ip, score=100)
    return {"success": True, "ip": ip}


@app.post("/unblock/{ip}")
async def unblock(ip: str):
    ok = manual_unblock(ip)
    return {"success": ok, "ip": ip}


# ── WebSocket feed ────────────────────────────────────────────────────────────

@app.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.append(websocket)
    logger.info("Dashboard client connected (%d total)", len(_ws_clients))
    try:
        # send recent alerts on connect
        for alert in list(_recent_alerts)[:20]:
            await websocket.send_text(
                json.dumps({"type": "history", "data": alert})
            )
        # send recent flows on connect
        for flow in list(_recent_flows)[:50]:
            await websocket.send_text(
                json.dumps({"type": "flow_history", "data": flow})
            )
        # keep alive
        while True:
            await asyncio.sleep(30)
            await websocket.send_text(json.dumps({"type": "ping"}))
    except (WebSocketDisconnect, asyncio.exceptions.CancelledError, asyncio.CancelledError):
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)
        logger.info("Dashboard client disconnected")


# ── health check ──────────────────────────────────────────────────────────────
@app.post("/test-trigger")
async def trigger_test_alert():
    import time
    from random import choice, randint, uniform
    
    _stats["flows_total"] += randint(150, 400)
    
    is_malicious = choice([True, True, False])  # 66% chance malicious for demo
    
    mock_alert = {
        "uid": f"test-{time.time_ns()}",
        "timestamp": time.time(),
        "risk_score": randint(50, 99) if is_malicious else randint(2, 15),
        "severity": choice(["critical", "high", "medium"]) if is_malicious else "low",
        "attack_type": choice(["Port scan", "DDoS", "Malicious Flow Detected"]) if is_malicious else "Benign",
        "src": f"192.168.1.{randint(20,250)}:{randint(1024,65535)}",
        "dst": "192.168.1.10:443",
        "proto": "TCP",
        "top_features": [["pkt_len_var", 0.45], ["flow_duration", 0.22], ["ack_flag_cnt", 0.15]],
        "explanation": "Simulated anomaly detected in traffic pattern. Rapid sequence of connection attempts." if is_malicious else "Normal traffic pattern observed.",
        "should_block": is_malicious and choice([True, False]),
        "ja3": choice(["e7d705a3286e19ea42f587b344ee6865", "6734f37431670b3ab4292b8f60f29984", None]) if is_malicious else None,
        "tls_version": "TLSv1.3" if choice([True, False]) else "TLSv1.2",
        "ja3_blocked": False
    }
    
    # Create a corresponding flow entry
    mock_flow = {
        "uid": mock_alert["uid"],
        "timestamp": mock_alert["timestamp"],
        "src": mock_alert["src"],
        "dst": mock_alert["dst"],
        "proto": mock_alert["proto"],
        "bytes_in": randint(40, 15000),
        "bytes_out": randint(40, 25000),
        "packets_in": randint(1, 200),
        "packets_out": randint(1, 150),
        "duration_ms": round(uniform(0.5, 30000.0), 1),
        "attack_type": mock_alert["attack_type"],
        "risk_score": mock_alert["risk_score"],
        "severity": mock_alert["severity"],
        "is_malicious": is_malicious,
    }
    
    # Update local state
    _recent_alerts.appendleft(mock_alert)
    _recent_flows.appendleft(mock_flow)
    if is_malicious:
        _stats["alerts_total"] += 1
        _stats["malicious_count"] += 1
    else:
        _stats["benign_count"] += 1
    _stats["last_update"] = time.time()
    _stats["bytes_in_total"] += mock_flow["bytes_in"]
    _stats["bytes_out_total"] += mock_flow["bytes_out"]
    _proto_counts[mock_alert["proto"]] += 1
    
    # Update TLS stats manually for the test trigger
    if mock_alert["ja3"]:
        # Add to top_ja3 if not already there or increment count
        found = False
        for i, (h, c) in enumerate(_tls_stats["top_ja3"]):
            if h == mock_alert["ja3"]:
                _tls_stats["top_ja3"][i] = (h, c + 1)
                found = True
                break
        if not found:
            _tls_stats["top_ja3"].append((mock_alert["ja3"], 1))
        
        # Ensure 'unique_ja3_count' matches the number of unique entries in 'top_ja3'
        _tls_stats["unique_ja3_count"] = len(_tls_stats["top_ja3"])
        _tls_stats["top_ja3"].sort(key=lambda x: x[1], reverse=True)
    else:
        # Map to "Unknown TLS Client" if mock has no JA3
        unk_hash = "39e71e7503f07a4a983362fa92d131f4"
        found = False
        for i, (h, c) in enumerate(_tls_stats["top_ja3"]):
            if h == unk_hash:
                _tls_stats["top_ja3"][i] = (unk_hash, c + 1)
                found = True
                break
        if not found:
            _tls_stats["top_ja3"].append((unk_hash, 1))

    # Synchronize total count perfectly
    _tls_stats["total_fingerprinted"] = sum(c for h, c in _tls_stats["top_ja3"])
    
    # Broadcast to all websocket clients
    await broadcast_alert(mock_alert)
    await broadcast_flow(mock_flow)
    
    return {"status": "ok", "mock_alert": mock_alert}

@app.get("/health")
async def health():
    return {"status": "ok", "time": time.time()}

@app.get("/")
async def serve_dashboard():
    html_path = _Path(__file__).parent / "index.html"
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)
