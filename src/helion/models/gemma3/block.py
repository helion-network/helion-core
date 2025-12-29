from __future__ import annotations

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
from hivemind.utils.logging import get_logger
from packaging.version import Version
from transformers.cache_utils import DynamicCache

logger = get_logger(__name__)
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
            # Derive past_length from actual tensor shape to handle cache mismatches
            # layer_past[0] is key cache in bloom format: [B*Hkv, D, T]
            # Use the actual T dimension from the tensor, not a potentially incorrect parameter
            key_cache_shape = layer_past[0].shape
            if len(key_cache_shape) == 3:
                # Bloom format: [B*Hkv, D, T]
                past_length = int(key_cache_shape[-1])
            else:
                # Fallback: try to infer from shape
                past_length = int(key_cache_shape[-1]) if key_cache_shape else 0
            cache = DynamicCache()
            # Pass past_length as hint, but _reorder_cache_from_bloom will use actual tensor shape
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
            # If there are no image tokens, don't enable the special mask path.
            # This avoids requiring torch>=2.6 for text-only prompts where token_type_ids is present but all zeros.
            if bool(is_image.any().item()):
                # `or_mask_function` / `and_mask_function` require torch>=2.6
                if Version(torch.__version__.split("+", 1)[0]) < Version("2.6"):
                    raise RuntimeError(
                        "Gemma3/MedGemma multimodal attention masks require torch>=2.6 on the worker. "
                        f"Detected torch=={torch.__version__}. Please upgrade the worker image / PyTorch."
                    )
            new_image_start = is_image & ~nn.functional.pad(is_image, (1, 0), value=0)[:, :-1]
            image_group_ids = torch.cumsum(new_image_start.int(), dim=1) - 1
            image_group_ids = torch.where(is_image, image_group_ids, torch.full_like(token_type_ids, -1))
            if bool(is_image.any().item()):
                mask_kwargs["or_mask_function"] = token_type_ids_mask_function(
                    token_type_ids.to(cache_position.device),
                    image_group_ids.to(cache_position.device),
                    self.mm_tokens_per_image,
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
        # Get head_dim from config first, fallback to tensor shape
        head_dim = int(getattr(self.self_attn, "head_dim", None) or getattr(self._text_config, "head_dim", None) or key_states.shape[-1])
        
        # Always derive actual sequence length from total element count to avoid shape mismatches
        # This is more robust than relying on shape dimensions which may be incorrect
        total_elements = key_states.numel()
        expected_batch_kv = batch_size * num_kv_heads
        actual_batch_kv = key_states.shape[0]
        
        # Verify batch/KV-heads dimension first
        if actual_batch_kv != expected_batch_kv:
            raise ValueError(
                f"Cache batch/KV-heads mismatch: expected {expected_batch_kv} (batch={batch_size} * kv_heads={num_kv_heads}), "
                f"but tensor has {actual_batch_kv} in first dimension. Tensor shape: {key_states.shape}"
            )
        
        # Calculate head_dim from element count if shape-based calculation seems wrong
        # After permute: key_states is [B*Hkv, T, D], so D = head_dim
        if len(key_states.shape) >= 2:
            shape_head_dim = key_states.shape[-1]
            # If head_dim from config doesn't match shape, recalculate from elements
            if head_dim != shape_head_dim:
                logger.warning(
                    f"Head dim mismatch: config says {head_dim} but tensor shape suggests {shape_head_dim}. "
                    f"Using shape-based value."
                )
                head_dim = shape_head_dim
        
        divisor = batch_size * num_kv_heads * head_dim
        if divisor == 0:
            raise ValueError(f"Invalid divisor: batch_size={batch_size}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")
        
        if total_elements % divisor != 0:
            raise ValueError(
                f"Cannot reshape cache: total elements ({total_elements}) is not divisible by "
                f"batch*kv_heads*head_dim ({divisor} = {batch_size}*{num_kv_heads}*{head_dim}). "
                f"Tensor shape: {key_states.shape}"
            )
        actual_seq_length = total_elements // divisor
        
        if actual_seq_length != seq_length:
            logger.warning(
                f"Cache sequence length mismatch: expected {seq_length} from past_length, "
                f"but tensor has {actual_seq_length} tokens (calculated from {total_elements} elements / {divisor}). Using actual size."
            )
        
        # Double-check the reshape will work before attempting it
        expected_view_elements = batch_size * num_kv_heads * actual_seq_length * head_dim
        if expected_view_elements != total_elements:
            raise ValueError(
                f"View shape mismatch: expected {expected_view_elements} elements for shape "
                f"[{batch_size}, {num_kv_heads}, {actual_seq_length}, {head_dim}], "
                f"but tensor has {total_elements} elements. Tensor shape: {key_states.shape}"
            )
        
        key_states = key_states.view(batch_size, num_kv_heads, actual_seq_length, head_dim)
        # value: [B*Hkv, T, D] -> [B, Hkv, T, D]
        # Calculate value sequence length from element count for consistency
        value_total_elements = value_states.numel()
        value_divisor = batch_size * num_kv_heads * head_dim
        if value_divisor == 0:
            raise ValueError(f"Invalid value divisor: batch_size={batch_size}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")
        
        if value_total_elements % value_divisor != 0:
            raise ValueError(
                f"Cannot reshape value cache: total elements ({value_total_elements}) is not divisible by "
                f"batch*kv_heads*head_dim ({value_divisor} = {batch_size}*{num_kv_heads}*{head_dim}). "
                f"Tensor shape: {value_states.shape}"
            )
        value_actual_seq_length = value_total_elements // value_divisor
        
        # Use the minimum of key and value sequence lengths to ensure consistency
        if value_actual_seq_length != actual_seq_length:
            logger.warning(
                f"Value cache sequence length ({value_actual_seq_length}) differs from key cache ({actual_seq_length}). "
                f"Using minimum for consistency."
            )
            actual_seq_length = min(actual_seq_length, value_actual_seq_length)
        
        # Double-check the reshape will work before attempting it
        expected_value_view_elements = batch_size * num_kv_heads * actual_seq_length * head_dim
        if expected_value_view_elements != value_total_elements:
            # If they don't match, try to recalculate from value cache
            if value_total_elements % (batch_size * num_kv_heads * head_dim) == 0:
                value_actual_seq_length = value_total_elements // (batch_size * num_kv_heads * head_dim)
                actual_seq_length = min(actual_seq_length, value_actual_seq_length)
                logger.warning(
                    f"Recalculated value cache sequence length to {value_actual_seq_length} from element count. "
                    f"Using {actual_seq_length} for both key and value."
                )
            else:
                raise ValueError(
                    f"Value view shape mismatch: expected {expected_value_view_elements} elements for shape "
                    f"[{batch_size}, {num_kv_heads}, {actual_seq_length}, {head_dim}], "
                    f"but tensor has {value_total_elements} elements. Tensor shape: {value_states.shape}"
                )
        
        value_states = value_states.view(batch_size, num_kv_heads, actual_seq_length, head_dim)
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


