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
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer

from helion.utils.auto_config import AutoDistributedConfig, AutoDistributedModelForCausalLM


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Dict[str, str]]
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
    tokenizers: Dict[str, AutoTokenizer] = {}
    models: Dict[str, AutoDistributedModelForCausalLM] = {}

    for i, mid in enumerate(model_ids, 1):
        print(f"[{i}/{len(model_ids)}] Loading model: {mid}", file=sys.stderr, flush=True)
        
        print(f"  Loading config...", file=sys.stderr, flush=True)
        cfg = AutoDistributedConfig.from_pretrained(
            mid,
            token=hf_token,
            initial_peers=initial_peers,
            dht_prefix=dht_prefix or mid,
        )
        # Safety: set fp32 lists to None for GPT-OSS (bypasses transformers validation)
        if getattr(cfg, "model_type", "") in {"gpt_oss", "gpt-oss"}:
            cfg._keep_in_fp32_modules = None
            cfg._keep_in_fp32_modules_strict = None

        print(f"  Loading tokenizer...", file=sys.stderr, flush=True)
        tok = AutoTokenizer.from_pretrained(mid, token=hf_token)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        print(f"  Loading distributed model (connecting to swarm, this may take a moment)...", file=sys.stderr, flush=True)
        # Final safety: set fp32 lists to None right before model creation (defensive)
        if getattr(cfg, "model_type", "") in {"gpt_oss", "gpt-oss"}:
            cfg._keep_in_fp32_modules = None
            cfg._keep_in_fp32_modules_strict = None
        mdl = AutoDistributedModelForCausalLM.from_pretrained(mid, config=cfg, token=hf_token)
        print(f"  ✓ Model {mid} loaded successfully", file=sys.stderr, flush=True)

        tokenizers[mid] = tok
        models[mid] = mdl

    print(f"All {len(model_ids)} models loaded. Starting server...", file=sys.stderr, flush=True)
    return models, tokenizers


def create_app(models: Dict[str, AutoDistributedModelForCausalLM], tokenizers: Dict[str, AutoTokenizer], default_model: str):
    app = FastAPI(title="Helion Multi-Model Test Server")

    @app.get("/health")
    def health():
        return {"status": "ok", "models": list(models.keys())}

    @app.get("/models")
    def list_models():
        return {"data": [{"id": mid, "object": "model"} for mid in models]}

    @app.post("/chat")
    def chat(req: ChatRequest):
        model_id = req.model or default_model
        if model_id not in models:
            raise HTTPException(status_code=400, detail=f"Model {model_id} not loaded")

        tok = tokenizers[model_id]
        mdl = models[model_id]

        if hasattr(tok, "apply_chat_template"):
            input_ids = tok.apply_chat_template(
                req.messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(mdl.device)
        else:
            prompt = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in req.messages)
            input_ids = tok(prompt, return_tensors="pt").input_ids.to(mdl.device)

        outputs = mdl.generate(
            input_ids,
            max_new_tokens=req.max_tokens,
            do_sample=req.temperature > 0.0,
            temperature=max(0.0, req.temperature),
            top_p=req.top_p,
        )
        text = tok.decode(outputs[0], skip_special_tokens=True)
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

    models, tokenizers = load_models(model_ids, args.hf_token, peers, args.dht_prefix)
    default_model = model_ids[0]
    app = create_app(models, tokenizers, default_model)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

