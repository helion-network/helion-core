from __future__ import annotations

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3DecoderLayer,
    Gemma3RotaryEmbedding,
    token_type_ids_mask_function,
)


class WrappedGemma3Block(Gemma3DecoderLayer):
    """
    A single Gemma3 decoder layer adapted to Helion's server conventions.

    Server calls:
    - forward(hidden_states, layer_past=..., use_cache=True, token_type_ids=...) during inference
    - forward(hidden_states) during regular forward
    """

    def __init__(self, config, layer_idx: int = 0):
        # `config` is DistributedGemma3Config (full), but Gemma3DecoderLayer expects text config.
        full_config = config
        text_config = getattr(config, "text_config", None) or config
        super().__init__(text_config, layer_idx=layer_idx)

        self._full_config = full_config
        self._text_config = text_config
        self.layer_idx = layer_idx

        # Rotary embeddings are normally owned by Gemma3TextModel; we compute them inside the block.
        self.rotary_emb = Gemma3RotaryEmbedding(config=text_config)
        local_cfg = copy.deepcopy(text_config)
        local_cfg.rope_theta = getattr(local_cfg, "rope_local_base_freq", getattr(local_cfg, "rope_theta", None))
        local_cfg.rope_scaling = {"rope_type": "default"}
        self.rotary_emb_local = Gemma3RotaryEmbedding(config=local_cfg)

        self.mm_tokens_per_image = int(getattr(full_config, "mm_tokens_per_image", 256) or 256)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Helion uses bloom-shaped KV caches: (key, value) where key is [B*Hkv, D, T], value is [B*Hkv, T, D]
        batch_size, seq_length, _ = hidden_states.shape

        past_length = 0
        cache = None
        if layer_past is not None:
            past_length = int(layer_past[0].shape[-1])
            cache = DynamicCache()
            key, value = self._reorder_cache_from_bloom(layer_past, batch_size, past_length)
            cache.update(key, value, self.layer_idx)
        elif use_cache:
            cache = DynamicCache()

        # Gemma3 uses cache_position/position_ids for masks and RoPE
        cache_position = torch.arange(past_length, past_length + seq_length, device=hidden_states.device)
        position_ids = cache_position.unsqueeze(0)

        # Build attention mask (Gemma3 switches between full and sliding attention by layer type)
        # NOTE: client does not send custom masks; we compute server-side.
        mask_kwargs = dict(
            config=self._text_config,
            input_embeds=hidden_states,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=cache,
            position_ids=position_ids,
        )

        if token_type_ids is not None and seq_length != 1:
            # Same logic as upstream: bidirectional attention within each image token block
            is_image = (token_type_ids == 1).to(cache_position.device)
            new_image_start = is_image & ~nn.functional.pad(is_image, (1, 0), value=0)[:, :-1]
            image_group_ids = torch.cumsum(new_image_start.int(), dim=1) - 1
            image_group_ids = torch.where(is_image, image_group_ids, torch.full_like(token_type_ids, -1))
            mask_kwargs["or_mask_function"] = token_type_ids_mask_function(
                token_type_ids.to(cache_position.device), image_group_ids.to(cache_position.device), self.mm_tokens_per_image
            )

        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
        }
        attn_mask_for_layer = causal_mask_mapping[self.attention_type]

        # RoPE embeddings (global + local)
        position_embeddings_global = self.rotary_emb(hidden_states, position_ids)
        position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids)

        outputs = super().forward(
            hidden_states,
            position_embeddings_global=position_embeddings_global,
            position_embeddings_local=position_embeddings_local,
            attention_mask=attn_mask_for_layer,
            position_ids=position_ids,
            past_key_value=cache,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        if not use_cache:
            return (hidden_states,)

        present_cache = cache.layers[self.layer_idx]
        present_key_value = self._reorder_cache_to_bloom(
            (present_cache.keys, present_cache.values), batch_size=batch_size
        )
        return hidden_states, present_key_value

    def _reorder_cache_from_bloom(
        self, key_value: Tuple[torch.Tensor, torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key_states, value_states = key_value
        # key: [B*Hkv, D, T] -> [B, Hkv, T, D]
        key_states = key_states.permute(0, 2, 1).contiguous()
        num_kv_heads = int(getattr(self._text_config, "num_key_value_heads", 1))
        head_dim = int(getattr(self.self_attn, "head_dim", key_states.shape[-1]))
        key_states = key_states.view(batch_size, num_kv_heads, seq_length, head_dim)
        # value: [B*Hkv, T, D] -> [B, Hkv, T, D]
        value_states = value_states.view(batch_size, num_kv_heads, seq_length, head_dim)
        return key_states, value_states

    def _reorder_cache_to_bloom(
        self, key_value: Tuple[torch.Tensor, torch.Tensor], *, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key_states, value_states = key_value
        # keys/values are [B, Hkv, T, D] -> bloom: key [B*Hkv, D, T], value [B*Hkv, T, D]
        bsz, num_kv_heads, seq_len, head_dim = key_states.shape
        assert bsz == batch_size
        value_states = value_states.view(batch_size * num_kv_heads, seq_len, head_dim).contiguous()
        key_states = key_states.view(batch_size * num_kv_heads, seq_len, head_dim).permute(0, 2, 1).contiguous()
        return key_states, value_states


