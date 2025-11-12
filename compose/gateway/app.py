import os
import json
import time
import uuid
import logging
import re
import math
import numbers
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoTokenizer, TextIteratorStreamer
from helion import AutoDistributedModelForCausalLM
# import httpx


# --------- Model bootstrap ---------
MODEL_ID = os.environ.get("MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")
ALLOWED_MODELS_ENV = os.environ.get("ALLOWED_MODELS", "").strip()
ALLOWED_MODELS: List[str] = [m.strip() for m in ALLOWED_MODELS_ENV.split(",") if m.strip()] or [MODEL_ID]
_initial_peers_env = os.environ.get("INITIAL_PEERS", "").strip()
INITIAL_PEERS: Optional[List[str]] = None
if _initial_peers_env:
    INITIAL_PEERS = [p.strip() for p in _initial_peers_env.split(",") if p.strip()]

# Optional Hugging Face token for gated repositories
HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")
VALIDATOR_URL = os.environ.get("VALIDATOR_URL")

# Initialize and preload all allowed models/tokenizers
TOKENIZERS: Dict[str, AutoTokenizer] = {}
MODELS: Dict[str, AutoDistributedModelForCausalLM] = {}

_model_kwargs: Dict[str, Any] = {"initial_peers": INITIAL_PEERS}
for _mid in ALLOWED_MODELS:
    # Tokenizer with token fallback (supports both `token` and legacy `use_auth_token`)
    if HF_TOKEN:
        try:
            _tok = AutoTokenizer.from_pretrained(_mid, token=HF_TOKEN)
        except TypeError:
            _tok = AutoTokenizer.from_pretrained(_mid, use_auth_token=HF_TOKEN)
    else:
        _tok = AutoTokenizer.from_pretrained(_mid)
    TOKENIZERS[_mid] = _tok

    # Model with token fallback
    if HF_TOKEN:
        try:
            _mdl = AutoDistributedModelForCausalLM.from_pretrained(_mid, token=HF_TOKEN, **_model_kwargs)
        except TypeError:
            _mdl = AutoDistributedModelForCausalLM.from_pretrained(_mid, use_auth_token=HF_TOKEN, **_model_kwargs)
    else:
        _mdl = AutoDistributedModelForCausalLM.from_pretrained(_mid, **_model_kwargs)
    MODELS[_mid] = _mdl

DEFAULT_MODEL_ID = ALLOWED_MODELS[0]


app = FastAPI(title="OpenAI-compatible API over Helion")


# --------- Helpers ---------
_route_capture_lock: Lock = Lock()
_last_route_text: Optional[str] = None
_last_route_parsed: Optional[List[Dict[str, Any]]] = None


def _parse_route_from_text(text: str) -> Optional[List[Dict[str, Any]]]:
    """
    Best-effort parser for route lines logged by Helion. Handles tuple-like segments
    such as (peer_id, start, end) and simple "peer:start-end" tokens.
    Returns a list of dicts or None if nothing could be parsed.
    """
    results: List[Dict[str, Any]] = []

    # Pattern 1: tuple segments like (peer_id, start, end)
    tuple_matches = re.findall(r"\(([^)]+)\)", text)
    for seg in tuple_matches:
        parts = [p.strip() for p in seg.split(",")]
        if len(parts) >= 3:
            peer_id = parts[0]
            try:
                start = int(re.findall(r"-?\d+", parts[1])[0])
            except Exception:
                start = None
            try:
                end = int(re.findall(r"-?\d+", parts[2])[0])
            except Exception:
                end = None
            results.append({"peer_id": peer_id, "start": start, "end": end})

    # Pattern 2: tokens like peer_id:start-end
    token_matches = re.findall(r"\b([^\s,\[\](){}:]+):(\d+)-(\d+)\b", text)
    for peer_id, start_s, end_s in token_matches:
        results.append({"peer_id": peer_id, "start": int(start_s), "end": int(end_s)})

    return results or None


class _RouteCaptureHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        global _last_route_text, _last_route_parsed
        try:
            msg = self.format(record) if self.formatter else record.getMessage()
        except Exception:
            msg = record.getMessage()

        text = msg or ""
        lowered = text.lower()
        # Heuristic: capture lines that likely describe the selected route
        if (
            ("selected route" in lowered)
            or ("route found" in lowered)
            or ("route" in lowered and ("peer" in lowered or "layer" in lowered))
        ):
            with _route_capture_lock:
                _last_route_text = text
                parsed = _parse_route_from_text(text)
                if parsed:
                    _last_route_parsed = parsed


# Attach the handler to Helion loggers (fallback to root if unavailable)
try:
    _route_handler = _RouteCaptureHandler(level=logging.INFO)
    logging.getLogger("helion").addHandler(_route_handler)
    logging.getLogger("helion").setLevel(min(logging.getLogger("helion").level or logging.INFO, logging.INFO))
except Exception:
    logging.getLogger().addHandler(_route_handler)

def _sanitize_jsonable(obj: Any):
    """
    Recursively convert non-finite numbers to None and tuples to lists
    so the structure is safe to serialize to JSON.
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

def _try_get_worker_chain(selected_model) -> Optional[List[Dict[str, Any]]]:
    """
    Best-effort introspection of the last route used by Helion to execute the model.
    Returns a list of {peer_id, start, end} if available, otherwise None.
    This relies on internal attributes that may differ across Helion versions and is fully optional.
    """
    # Defensive: never raise from here
    try:
        # Common potential locations for the last/active route in Helion internals
        candidates: List[Any] = []

        # e.g., model.router.last_route
        router = getattr(selected_model, "router", None)
        if router is not None:
            candidates.append(getattr(router, "last_route", None))
            # Sometimes a cache object keeps the last good route
            route_cache = getattr(router, "route_cache", None)
            if route_cache is not None:
                candidates.append(getattr(route_cache, "last_good", None))

        # e.g., model.last_route
        candidates.append(getattr(selected_model, "last_route", None))

        # Try to reach RemoteSequenceManager and its stored last route
        transformer = getattr(selected_model, "transformer", None)
        h = getattr(transformer, "h", None) if transformer is not None else None
        seq_mgr = getattr(h, "sequence_manager", None) if h is not None else None
        if seq_mgr is not None:
            seq_state = getattr(seq_mgr, "state", None)
            if seq_state is not None:
                candidates.append(getattr(seq_state, "last_route", None))
            candidates.append(getattr(seq_mgr, "last_route", None))

        # e.g., model.client.last_route or model._client.last_route
        client = getattr(selected_model, "client", None) or getattr(selected_model, "_client", None)
        if client is not None:
            candidates.append(getattr(client, "last_route", None))
            runtime = getattr(client, "runtime", None)
            if runtime is not None:
                candidates.append(getattr(runtime, "last_route", None))

        # Pick the first truthy candidate
        route = next((c for c in candidates if c), None)
        if route is None:
            # As a last resort, try computing a route now via the sequence manager
            try:
                transformer = getattr(selected_model, "transformer", None)
                h = getattr(transformer, "h", None) if transformer is not None else None
                seq_mgr = getattr(h, "sequence_manager", None) if h is not None else None
                if seq_mgr is not None:
                    spans = seq_mgr.make_sequence(0, None, mode="min_latency")
                    detailed: List[Dict[str, Any]] = []
                    for span in spans:
                        server = getattr(span, "server_info", None)
                        item: Dict[str, Any] = {
                            "peer_id": getattr(getattr(span, "peer_id", None), "to_base58", lambda: str(getattr(span, "peer_id", None)))(),
                            "start": int(getattr(span, "start", 0)),
                            "end": int(getattr(span, "end", 0)),
                        }
                        if server is not None:
                            try:
                                item["server"] = {
                                    "state": getattr(getattr(server, "state", None), "name", None) or int(getattr(server, "state", 0)),
                                    "throughput": float(getattr(server, "throughput", None)) if getattr(server, "throughput", None) is not None else None,
                                    "network_rps": float(getattr(server, "network_rps", None)) if getattr(server, "network_rps", None) is not None else None,
                                    "forward_rps": float(getattr(server, "forward_rps", None)) if getattr(server, "forward_rps", None) is not None else None,
                                    "inference_rps": float(getattr(server, "inference_rps", None)) if getattr(server, "inference_rps", None) is not None else None,
                                    "cache_tokens_left": int(getattr(server, "cache_tokens_left", None)) if getattr(server, "cache_tokens_left", None) is not None else None,
                                    "next_pings": getattr(server, "next_pings", None),
                                    "public_name": getattr(server, "public_name", None),
                                    "version": getattr(server, "version", None),
                                    "using_relay": getattr(server, "using_relay", None),
                                    "adapters": list(getattr(server, "adapters", []) or []),
                                    "torch_dtype": getattr(server, "torch_dtype", None),
                                    "quant_type": getattr(server, "quant_type", None),
                                }
                            except Exception:
                                pass
                        detailed.append(item)
                    if detailed:
                        return _sanitize_jsonable(detailed)
            except Exception:
                pass

            # Fallback to captured logs
            with _route_capture_lock:
                if _last_route_parsed:
                    return _sanitize_jsonable(_last_route_parsed)
                if _last_route_text:
                    parsed_from_text = _parse_route_from_text(_last_route_text)
                    if parsed_from_text:
                        return _sanitize_jsonable(parsed_from_text)
            return None

        # If the candidate is a plain string, try to parse it
        if isinstance(route, str):
            parsed = _parse_route_from_text(route)
            if parsed:
                return _sanitize_jsonable(parsed)

        # Normalize to list for iteration
        # If we already have a list of dicts, return as-is
        if isinstance(route, list) and route and isinstance(route[0], dict):
            return _sanitize_jsonable(route)  # expected shape: {peer_id, start, end}

        route_items = list(route) if not isinstance(route, list) else route
        result: List[Dict[str, Any]] = []
        for r in route_items:
            # Try object-like fields first
            peer_id = getattr(r, "peer_id", None) or getattr(r, "peer", None) or getattr(r, "uid", None)
            start = getattr(r, "start", None) or getattr(r, "start_layer", None)
            end = getattr(r, "end", None) or getattr(r, "end_layer", None)

            # Fallback: tuple/list shapes like (peer_id, start, end)
            if peer_id is None and isinstance(r, (list, tuple)):
                if len(r) >= 3:
                    peer_id, start, end = r[0], r[1], r[2]
                elif len(r) == 2:
                    peer_id, start = r[0], r[1]

            # Best-effort conversion
            item: Dict[str, Any] = {
                "peer_id": None if peer_id is None else str(peer_id),
                "start": None if start is None else int(start),
                "end": None if end is None else int(end),
            }
            result.append(item)

        # Filter empty results
        if any(x.get("peer_id") is not None for x in result):
            return _sanitize_jsonable(result)
    except Exception:
        pass
    return None

def _try_get_route_repr(selected_model) -> Optional[str]:
    """
    Best-effort retrieval of a human-readable route string from the sequence manager or captured logs.
    """
    try:
        transformer = getattr(selected_model, "transformer", None)
        h = getattr(transformer, "h", None) if transformer is not None else None
        seq_mgr = getattr(h, "sequence_manager", None) if h is not None else None
        seq_state = getattr(seq_mgr, "state", None) if seq_mgr is not None else None
        route_repr = getattr(seq_state, "last_route_repr", None) if seq_state is not None else None
        if route_repr:
            return route_repr
    except Exception:
        pass
    # Fallback to last captured log line
    with _route_capture_lock:
        return _last_route_text
def build_prompt_from_messages(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> str:
    lines: List[str] = []
    # Minimal OpenAI chat-to-prompt conversion
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            # content could be a list of parts; concatenate text parts
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            content = "\n".join(text_parts)
        if role == "tool":
            # Tool outputs: include as a tagged block
            tool_name = m.get("name") or m.get("tool_name") or "tool"
            lines.append(f"[{tool_name} result]: {content}")
        else:
            lines.append(f"{role}: {content}")
    lines.append("assistant:")
    return "\n".join(lines)


def iterate_chat_chunks(
    input_ids,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    model_name: str,
    tokenizer,
    model,
) -> Iterable[bytes]:
    created = int(time.time())
    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0.0,
        temperature=max(0.0, temperature),
        top_p=top_p,
        repetition_penalty=1.05,
        streamer=streamer,
    )

    # Run generation in a background thread
    import threading

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    # Send initial chunk (role notification)
    first = {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
        ],
    }
    yield f"data: {json.dumps(first)}\n\n".encode()

    buffer: List[str] = []
    for token in streamer:
        buffer.append(token)
        chunk = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [
                {"index": 0, "delta": {"content": token}, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n".encode()

    # Send final chunk with finish_reason
    final_chunk = {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [
            {"index": 0, "delta": {}, "finish_reason": "stop"}
        ],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n".encode()
    yield b"data: [DONE]\n\n"


# --------- Routes ---------
@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": mid, "object": "model", "created": 0, "owned_by": "petals"}
            for mid in ALLOWED_MODELS
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    tools = body.get("tools")  # accepted but not executed
    stream = bool(body.get("stream", False))
    max_tokens = int(body.get("max_tokens", 5000))
    temperature = float(body.get("temperature", 0.7))
    top_p = float(body.get("top_p", 0.95))
    requested_model: Optional[str] = body.get("model") or None

    # Select model
    model_id = requested_model or DEFAULT_MODEL_ID
    if model_id not in ALLOWED_MODELS:
        return JSONResponse(
            {
                "error": "Model not allowed",
                "allowed_models": ALLOWED_MODELS,
            },
            status_code=400,
        )
    selected_tokenizer = TOKENIZERS[model_id]
    selected_model = MODELS[model_id]

    # Build inputs using a chat template if available
    if hasattr(selected_tokenizer, "apply_chat_template"):
        inputs = selected_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        prompt = build_prompt_from_messages(messages, tools)
        inputs = selected_tokenizer(prompt, return_tensors="pt")["input_ids"]

    # Best-effort: proactively compute a route to make it available for response metadata
    try:
        transformer = getattr(selected_model, "transformer", None)
        h = getattr(transformer, "h", None) if transformer is not None else None
        seq_mgr = getattr(h, "sequence_manager", None) if h is not None else None
        if seq_mgr is not None:
            cache_tokens_needed = int(max_tokens)
            # Trigger route selection so that last_route/last_route_repr are populated
            seq_mgr.make_sequence(0, None, mode="min_latency", cache_tokens_needed=cache_tokens_needed)
    except Exception:
        pass

    if stream:
        return StreamingResponse(
            iterate_chat_chunks(
                input_ids=inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                model_name=model_id,
                tokenizer=selected_tokenizer,
                model=selected_model,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming: generate full text
    outputs = selected_model.generate(
        inputs,
        max_new_tokens=max_tokens,
        do_sample=temperature > 0.0,
        temperature=max(0.0, temperature),
        top_p=top_p,
        repetition_penalty=1.05,
    )
    text = selected_tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the assistant completion if using the simple prompt builder
    if not hasattr(selected_tokenizer, "apply_chat_template") and "assistant:" in text:
        text = text.split("assistant:")[-1].lstrip()

    # Optional: call external validator if a payload is provided and validator URL is configured
    # validation_result: Optional[Dict[str, Any]] = None
    # validator_payload = body.get("validator_payload")
    # if VALIDATOR_URL and validator_payload:
    #     try:
    #         async with httpx.AsyncClient(timeout=10.0) as client:
    #             r = await client.post(
    #                 f"{VALIDATOR_URL.rstrip('/')}/validate", json=validator_payload
    #             )
    #             r.raise_for_status()
    #             validation_result = r.json()
    #     except Exception as e:
    #         validation_result = {"error": str(e)}

    resp = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text, "tool_calls": []},
                "finish_reason": "stop",
            }
        ],
        "usage": None,
    }
    # Best-effort: include both worker chain and route text independently
    worker_chain = _try_get_worker_chain(selected_model)
    if worker_chain:
        resp["worker_chain"] = worker_chain
    route_repr = _try_get_route_repr(selected_model)
    if route_repr:
        resp["route_repr"] = route_repr
    # if validation_result is not None:
    #     resp["validation"] = validation_result
    return JSONResponse(resp)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))


