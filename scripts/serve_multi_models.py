#!/usr/bin/env python3
"""
Spin up a minimal FastAPI server that serves multiple Helion distributed models.

This is for local testing (Choice 1): start a server that can handle multiple
models, then you can hit it with clients or curl.

Example:
    python scripts/serve_multi_models.py \
        --models meta-llama/Llama-3.2-1B-Instruct,openai/gpt-oss-model \
        --initial-peers "/ip4/203.0.113.10/tcp/31337/p2p/PeerIdHere" \
        --hf-token hf_xxx \
        --port 8080

MedGemma (multimodal) example:
    python scripts/serve_multi_models.py --models google/medgemma-4b-it --hf-token hf_xxx --port 8080

    curl -X POST http://localhost:8080/chat ^
      -H "Content-Type: application/json" ^
      -d "{\"model\":\"google/medgemma-4b-it\",\"messages\":[{\"role\":\"system\",\"content\":[{\"type\":\"text\",\"text\":\"You are an expert radiologist.\"}]},{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Describe this X-ray\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png\"}}]}],\"max_tokens\":200,\"temperature\":0}"
"""

from __future__ import annotations

import argparse
import math
import os
import asyncio
import threading
import time
import numbers
from typing import Any, Dict, List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoProcessor, AutoTokenizer

from helion.utils.auto_config import (
    AutoDistributedConfig,
    AutoDistributedModelForCausalLM,
    AutoDistributedModelForConditionalGeneration,
)


class ChatRequest(BaseModel):
    model: Optional[str] = None
    # For text-only models: {"role": "...", "content": "..."}
    # For multimodal (Gemma3/MedGemma): {"role": "...", "content": [{"type":"text","text":"..."}, {"type":"image","url":"..."}]}
    messages: List[Dict[str, Any]]
    max_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.95


def load_models(
    model_ids: List[str],
    hf_token: Optional[str],
    initial_peers: Optional[List[str]],
    dht_prefix: Optional[str],
):
    import sys
    preprocessors: Dict[str, Any] = {}
    models: Dict[str, Any] = {}

    def _normalize_dht_prefix(prefix: Optional[str]) -> Optional[str]:
        """
        Helion encodes module uids as `{dht_prefix}{UID_DELIMITER}{block_index}` where UID_DELIMITER == "."
        so the prefix must NOT contain "." or spaces (CHAIN_DELIMITER).

        If the caller provides a prefix, normalize it to a safe value. If not provided (None/empty),
        return None so model-specific defaults can apply (recommended).
        """
        if not prefix:
            return None
        # Keep this conservative: only remove characters that will break uid parsing.
        prefix = prefix.strip()
        prefix = prefix.replace(".", "-").replace(" ", "-")
        return prefix or None

    normalized_prefix = _normalize_dht_prefix(dht_prefix)

    for i, mid in enumerate(model_ids, 1):
        print(f"[{i}/{len(model_ids)}] Loading model: {mid}", file=sys.stderr, flush=True)
        
        print(f"  Loading config...", file=sys.stderr, flush=True)
        cfg = AutoDistributedConfig.from_pretrained(
            mid,
            token=hf_token,
            initial_peers=initial_peers,
            # dht_prefix=dht_prefix or mid,
            # IMPORTANT: do not default to `mid` (model id) as dht_prefix: model ids often contain "."
            # (e.g. Llama-3.2-...), and UID_DELIMITER is "." which breaks uid parsing.
            dht_prefix=normalized_prefix,
        )
        # Safety: set fp32 lists to None for GPT-OSS (bypasses transformers validation)
        if getattr(cfg, "model_type", "") in {"gpt_oss", "gpt-oss"}:
            cfg._keep_in_fp32_modules = None
            cfg._keep_in_fp32_modules_strict = None

        print(f"  Loading tokenizer/processor...", file=sys.stderr, flush=True)
        if getattr(cfg, "model_type", "") == "gemma3":
            try:
                pre = AutoProcessor.from_pretrained(mid, token=hf_token)
            except ImportError as e:
                raise SystemExit(
                    "Missing dependency for Gemma3/MedGemma processor. Install Pillow:\n"
                    "  pip install pillow\n"
                    "If you're running in Docker, rebuild the image after updating dependencies.\n"
                    f"Original error: {e}"
                )
        else:
            pre = AutoTokenizer.from_pretrained(mid, token=hf_token)
            pre.padding_side = "left"
            if getattr(pre, "pad_token", None) is None:
                pre.pad_token = pre.eos_token

        print(f"  Loading distributed model (connecting to swarm, this may take a moment)...", file=sys.stderr, flush=True)
        # Final safety: set fp32 lists to None right before model creation (defensive)
        if getattr(cfg, "model_type", "") in {"gpt_oss", "gpt-oss"}:
            cfg._keep_in_fp32_modules = None
            cfg._keep_in_fp32_modules_strict = None
        if getattr(cfg, "model_type", "") == "gemma3":
            mdl = AutoDistributedModelForConditionalGeneration.from_pretrained(mid, config=cfg, token=hf_token)
        else:
            mdl = AutoDistributedModelForCausalLM.from_pretrained(mid, config=cfg, token=hf_token)
        print(f"  ✓ Model {mid} loaded successfully", file=sys.stderr, flush=True)

        preprocessors[mid] = pre
        models[mid] = mdl

    print(f"All {len(model_ids)} models loaded. Starting server...", file=sys.stderr, flush=True)
    return models, preprocessors


def create_app(models: Dict[str, Any], preprocessors: Dict[str, Any], default_model: str):
    app = FastAPI(title="Helion Multi-Model Test Server")
    # Local-only imports: keep server startup lightweight, but allow richer state decoding in health
    from helion.data_structures import ServerState  # type: ignore
    from helion.utils.dht import compute_spans  # type: ignore

    def _sanitize_jsonable(obj: Any) -> Any:
        """
        Convert values that Starlette's JSON renderer rejects (NaN/Inf) into JSON-safe values.
        Also normalize tuples into lists to keep the output consistently JSON-serializable.
        """
        try:
            if isinstance(obj, dict):
                return {k: _sanitize_jsonable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitize_jsonable(v) for v in obj]
            if isinstance(obj, tuple):
                return [_sanitize_jsonable(v) for v in obj]
            if isinstance(obj, numbers.Real):
                f = float(obj)
                return f if math.isfinite(f) else None
        except Exception:
            pass
        return obj

    # -----------------------
    # Health state (cached)
    # -----------------------
    update_period_s = float(os.environ.get("UPDATE_PERIOD", "60"))
    _health_lock = threading.Lock()
    _health_snapshot: Dict[str, Any] = {"last_updated": None, "update_duration": None, "model_reports": []}

    def _get_sequence_manager(model: Any):
        """
        Best-effort access to Helion's RemoteSequenceManager.
        Different model wrappers expose it differently; we keep this defensive.
        """
        transformer = getattr(model, "transformer", None)
        h = getattr(transformer, "h", None) if transformer is not None else None
        seq_mgr = getattr(h, "sequence_manager", None) if h is not None else None
        return seq_mgr

    def _peer_to_b58(peer: Any) -> str:
        return getattr(peer, "to_base58", lambda: str(peer))()

    def _server_state_to_str(state_value: Any) -> str:
        """
        Normalize various representations of ServerState to a stable lowercase string
        (e.g. 'joining', 'online', 'offline').
        """
        if state_value is None:
            return "unknown"
        if isinstance(state_value, str):
            return state_value.lower()
        try:
            return ServerState(int(state_value)).name.lower()
        except Exception:
            return str(state_value).lower()

    def _server_info_to_dict(server: Any) -> Dict[str, Any]:
        # Mirror the fields the debug guide cares about, keep it stable and JSONable
        return _sanitize_jsonable(
            {
            "state": getattr(getattr(server, "state", None), "name", None)
            or int(getattr(server, "state", 0) or 0),
            "public_name": getattr(server, "public_name", None),
            "version": getattr(server, "version", None),
            "using_relay": getattr(server, "using_relay", None),
            "torch_dtype": getattr(server, "torch_dtype", None),
            "quant_type": getattr(server, "quant_type", None),
            "adapters": list(getattr(server, "adapters", []) or []),
            "throughput": getattr(server, "throughput", None),
            "network_rps": getattr(server, "network_rps", None),
            "forward_rps": getattr(server, "forward_rps", None),
            "inference_rps": getattr(server, "inference_rps", None),
            "cache_tokens_left": getattr(server, "cache_tokens_left", None),
            "next_pings": getattr(server, "next_pings", None),
            }
        )

    def _sequence_info_empty(seq_mgr: Any) -> bool:
        try:
            si = getattr(getattr(seq_mgr, "state", None), "sequence_info", None)
            if si is None or getattr(si, "last_updated_time", None) is None:
                return True
            # NOTE: spans_by_priority is ONLINE-only in core. For health/debug we also treat JOINING servers
            # (often relay-mode) as "present" so they can be surfaced.
            for block_info in getattr(si, "block_infos", None) or []:
                servers = getattr(block_info, "servers", None) or {}
                for _peer_id, server in servers.items():
                    st = getattr(server, "state", None)
                    try:
                        st_value = st.value  # Enum-like
                    except Exception:
                        st_value = st
                    try:
                        if int(st_value) >= ServerState.JOINING.value:
                            return False
                    except Exception:
                        continue
            return True
        except Exception:
            return True

    def _compute_spans_for_model(model: Any) -> Tuple[List[Dict[str, Any]], bool, Optional[str]]:
        """
        Return: (spans, used_fallback_route, last_route_repr)
        Each span is: {peer_id, start, end, length, server_info}
        """
        seq_mgr = _get_sequence_manager(model)
        if seq_mgr is None:
            return [], True, None

        # Start/prime the sequence manager thread so `sequence_info` gets populated.
        # We use an empty-range route: it forces the first DHT refresh but never requires ONLINE blocks.
        try:
            seq_mgr.make_sequence(0, 0, mode="min_latency")
        except Exception:
            pass

        # Kick an async update too (best-effort).
        try:
            # make sure the update thread is running; does not block on network
            _ = seq_mgr.ready
        except Exception:
            pass
        try:
            seq_mgr.update(wait=False)
        except Exception:
            pass

        last_route_repr = None
        try:
            last_route_repr = getattr(getattr(seq_mgr, "state", None), "last_route_repr", None)
        except Exception:
            pass

        used_fallback = _sequence_info_empty(seq_mgr)
        spans_out: List[Dict[str, Any]] = []

        if not used_fallback:
            try:
                # `spans_by_priority` is ONLINE-only; for health we also want JOINING (relay-mode) servers.
                si = seq_mgr.state.sequence_info
                spans_map = compute_spans(list(getattr(si, "block_infos", []) or []), min_state=ServerState.JOINING)
                spans = sorted(
                    spans_map.values(),
                    key=lambda s: (-int(getattr(s, "length", 0)), _peer_to_b58(getattr(s, "peer_id", ""))),
                )
                for span in spans:
                    server = getattr(span, "server_info", None)
                    spans_out.append(
                        {
                            "peer_id": _peer_to_b58(getattr(span, "peer_id", None)),
                            "start": int(getattr(span, "start", 0)),
                            "end": int(getattr(span, "end", 0)),
                            "length": int(getattr(span, "length", 0)),
                            "server_info": _server_info_to_dict(server) if server is not None else None,
                        }
                    )
            except Exception:
                used_fallback = True

        # Fallback: snapshot a single selected route (may exclude relay peers)
        if used_fallback:
            try:
                route = seq_mgr.make_sequence(0, None, mode="min_latency")
                for span in route:
                    server = getattr(span, "server_info", None)
                    spans_out.append(
                        {
                            "peer_id": _peer_to_b58(getattr(span, "peer_id", None)),
                            "start": int(getattr(span, "start", 0)),
                            "end": int(getattr(span, "end", 0)),
                            "length": int(getattr(span, "length", 0)),
                            "server_info": _server_info_to_dict(server) if server is not None else None,
                        }
                    )
                try:
                    last_route_repr = getattr(getattr(seq_mgr, "state", None), "last_route_repr", None) or last_route_repr
                except Exception:
                    pass
            except Exception:
                pass

        return spans_out, used_fallback, last_route_repr

    def _build_health_snapshot() -> Dict[str, Any]:
        t0 = time.perf_counter()
        model_reports: List[Dict[str, Any]] = []
        top_contrib: Dict[str, int] = {}
        servers_total = 0
        servers_relay = 0

        for model_id, mdl in models.items():
            spans, used_fallback, _ = _compute_spans_for_model(mdl)
            # server_rows: one per span (simple), the real gateway may aggregate; this is enough for debugging
            server_rows = []
            for sp in spans:
                peer_id = sp["peer_id"]
                short_peer_id = peer_id[-6:] if isinstance(peer_id, str) and len(peer_id) >= 6 else peer_id
                server_info = sp.get("server_info") or {}
                using_relay = bool(server_info.get("using_relay")) if server_info.get("using_relay") is not None else None
                worker_state = _server_state_to_str(server_info.get("state"))
                server_rows.append(
                    {
                        "peer_id": peer_id,
                        "short_peer_id": short_peer_id,
                        "state": worker_state,
                        "span": {
                            "start": sp["start"],
                            "end": sp["end"],
                            "length": sp["length"],
                            "server_info": server_info or None,
                        },
                    }
                )
                public_name = server_info.get("public_name")
                if public_name:
                    top_contrib[public_name] = top_contrib.get(public_name, 0) + int(sp.get("length") or 0)

                servers_total += 1
                if using_relay is True:
                    servers_relay += 1

            model_reports.append(
                {
                    "model_id": model_id,
                    "used_fallback_route": used_fallback,
                    "server_rows": server_rows,
                }
            )

        snapshot = _sanitize_jsonable(
            {
            "last_updated": time.time(),
            "update_period": update_period_s,
            "update_duration": time.perf_counter() - t0,
            "servers_num_total": servers_total,
            "servers_num_relay": servers_relay,
            "top_contributors": [{"public_name": k, "blocks": v} for k, v in sorted(top_contrib.items(), key=lambda kv: kv[1], reverse=True)],
            "model_reports": model_reports,
            }
        )
        return snapshot

    def _health_updater_loop():
        while True:
            try:
                snap = _build_health_snapshot()
                with _health_lock:
                    _health_snapshot.clear()
                    _health_snapshot.update(snap)
            except Exception:
                # never crash the server for health
                pass
            time.sleep(update_period_s)

    threading.Thread(target=_health_updater_loop, daemon=True).start()

    @app.get("/health")
    def health():
        return {"status": "ok", "models": list(models.keys())}

    # --- Health API (debug-friendly; mirrors HEALTH_API_DEBUG.md concepts) ---
    @app.get("/api/v1/state")
    def get_state():
        with _health_lock:
            return _sanitize_jsonable(dict(_health_snapshot))

    @app.get("/api/v1/debug/routes")
    def debug_routes():
        out = []
        for model_id, mdl in models.items():
            seq_mgr = _get_sequence_manager(mdl)
            if seq_mgr is None:
                out.append({"model_id": model_id, "ready": False, "sequence_info_empty": True, "error": "no_sequence_manager"})
                continue

            try:
                ready = bool(getattr(seq_mgr, "ready", None).is_set())
            except Exception:
                ready = False

            sequence_info_empty = _sequence_info_empty(seq_mgr)
            spans, used_fallback, last_route_repr = _compute_spans_for_model(mdl)

            # Only include a small, stable subset of span fields
            out.append(
                {
                    "model_id": model_id,
                    "client_peer_id": _peer_to_b58(getattr(getattr(seq_mgr, "dht", None), "peer_id", None)),
                    "dht_prefix": getattr(getattr(mdl, "config", None), "dht_prefix", None),
                    "ready": ready,
                    "used_fallback_route": used_fallback,
                    "sequence_info_empty": sequence_info_empty,
                    "last_route_repr": last_route_repr,
                    "spans": [
                        {
                            "peer_id": sp["peer_id"],
                            "start": sp["start"],
                            "end": sp["end"],
                            "server_info": sp.get("server_info"),
                        }
                        for sp in spans
                    ],
                }
            )
        return {"models": out}

    async def _try_connect_peer(p2p: Any, peer_id: Any, timeout_s: float = 5.0) -> Tuple[bool, str]:
        """
        Best-effort reachability probe. Hivemind/P2P APIs differ across versions, so try several.
        """
        methods = ("connect_peer", "connect", "dial_peer")
        last_err: Optional[Exception] = None
        for name in methods:
            fn = getattr(p2p, name, None)
            if fn is None:
                continue
            try:
                res = fn(peer_id)
                if asyncio.iscoroutine(res):
                    await asyncio.wait_for(res, timeout=timeout_s)
                else:
                    # If it's sync, just run it; still respect timeout by yielding back
                    await asyncio.wait_for(asyncio.to_thread(fn, peer_id), timeout=timeout_s)
                return True, f"{name} succeeded"
            except Exception as e:
                last_err = e
                continue
        return False, f"no_connect_method_succeeded: {repr(last_err)}"

    @app.get("/api/v1/is_reachable/{peer_id}")
    async def is_reachable(peer_id: str):
        # Pick any model that has a sequence manager and use its replicated p2p
        seq_mgr = None
        for mdl in models.values():
            seq_mgr = _get_sequence_manager(mdl)
            if seq_mgr is not None:
                break
        if seq_mgr is None:
            raise HTTPException(status_code=500, detail="No sequence_manager available on loaded models")

        try:
            from hivemind import PeerID  # pyright: ignore[reportMissingImports]

            target_peer = PeerID.from_base58(peer_id)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid peer id: {peer_id}")

        p2p = getattr(getattr(seq_mgr, "state", None), "p2p", None)
        if p2p is None:
            raise HTTPException(status_code=500, detail="No p2p available on sequence_manager.state")

        ok, msg = await _try_connect_peer(p2p, target_peer, timeout_s=float(os.environ.get("REACHABILITY_TIMEOUT", "5")))
        return {"success": ok, "message": msg}

    @app.get("/metrics")
    def metrics():
        # Prometheus-compatible text (cached snapshot)
        with _health_lock:
            snap = dict(_health_snapshot)
        lines = []
        lines.append("# HELP servers_num_total Number of server spans in the last snapshot")
        lines.append("# TYPE servers_num_total gauge")
        lines.append(f"servers_num_total {int(snap.get('servers_num_total') or 0)}")
        lines.append("# HELP servers_num_relay Number of server spans reported as using relay")
        lines.append("# TYPE servers_num_relay gauge")
        lines.append(f"servers_num_relay {int(snap.get('servers_num_relay') or 0)}")
        for mr in snap.get("model_reports", []) or []:
            model_id = mr.get("model_id", "unknown")
            server_rows = mr.get("server_rows", []) or []
            lines.append("# HELP model_servers_num_total Number of server spans for a model")
            lines.append("# TYPE model_servers_num_total gauge")
            lines.append(f'model_servers_num_total{{model="{model_id}"}} {len(server_rows)}')
        return "\n".join(lines) + "\n"

    @app.get("/models")
    def list_models():
        return {"data": [{"id": mid, "object": "model"} for mid in models]}

    @app.post("/chat")
    def chat(req: ChatRequest):
        model_id = req.model or default_model
        if model_id not in models:
            raise HTTPException(status_code=400, detail=f"Model {model_id} not loaded")

        mdl = models[model_id]
        pre = preprocessors[model_id]
        model_type = getattr(getattr(mdl, "config", None), "model_type", None)

        if model_type == "gemma3":
            def _normalize_gemma3_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                out: List[Dict[str, Any]] = []
                for m in messages:
                    role = m.get("role")
                    content = m.get("content")
                    if isinstance(content, list):
                        new_parts = []
                        for part in content:
                            if not isinstance(part, dict):
                                continue
                            if part.get("type") == "image_url":
                                url = (part.get("image_url") or {}).get("url")
                                if url:
                                    new_parts.append({"type": "image", "url": url})
                                continue
                            new_parts.append(part)
                        out.append({"role": role, "content": new_parts})
                    else:
                        out.append({"role": role, "content": content})
                return out

            # Multimodal: AutoProcessor returns a dict with input_ids/token_type_ids/pixel_values (if any)
            inputs = pre.apply_chat_template(
                _normalize_gemma3_messages(req.messages),
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            # Move tensors to model device
            for k, v in list(inputs.items()):
                if hasattr(v, "to"):
                    inputs[k] = v.to(mdl.device)
            input_len = int(inputs["input_ids"].shape[-1])
            outputs = mdl.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                do_sample=req.temperature > 0.0,
                temperature=max(0.0, req.temperature),
                top_p=req.top_p,
            )
            gen = outputs[0][input_len:]
            text = pre.decode(gen, skip_special_tokens=True)
        else:
            prompt = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in req.messages)
            if hasattr(pre, "apply_chat_template"):
                input_ids = pre.apply_chat_template(
                    req.messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(mdl.device)
            else:
                input_ids = pre(prompt, return_tensors="pt").input_ids.to(mdl.device)

            outputs = mdl.generate(
                input_ids,
                max_new_tokens=req.max_tokens,
                do_sample=req.temperature > 0.0,
                temperature=max(0.0, req.temperature),
                top_p=req.top_p,
            )
            text = pre.decode(outputs[0], skip_special_tokens=True)
        return {"model": model_id, "output": text}

    return app


def main():
    parser = argparse.ArgumentParser(description="Serve multiple Helion models locally for testing")
    parser.add_argument(
        "--models",
        required=False,
        default=os.environ.get("MODELS", ""),
        help="Comma-separated list of model IDs (or set MODELS env)",
    )
    parser.add_argument(
        "--initial-peers",
        default="/ip4/34.143.175.170/tcp/31337/p2p/QmPXGhXJRDZZLRPDKSRXGnM9FiccEncxqrLjWSLZhjgKZS",
        help="Comma-separated initial peers",
    )
    parser.add_argument("--hf-token", default="")
    parser.add_argument("--dht-prefix", default=None, help="Optional shared DHT prefix")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_ids:
        raise SystemExit("No models provided. Set --models or MODELS env to a comma-separated list.")
    peers = [p.strip() for p in args.initial_peers.split(",") if p.strip()] or None

    models, preprocessors = load_models(model_ids, args.hf_token, peers, args.dht_prefix)
    default_model = model_ids[0]
    app = create_app(models, preprocessors, default_model)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

