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

    def _validate_and_fix_cache_dimensions(
        self, cache: DynamicCache, batch_size: int, expected_seq_len: int
    ) -> bool:
        """
        Validate cache dimensions and attempt to fix if incorrect.
        Returns True if cache is valid, False if fix was attempted.
        """
        if cache is None or self.layer_idx not in cache.layers:
            return True
        
        stored_cache = cache.layers[self.layer_idx]
        stored_keys = stored_cache.keys
        stored_values = stored_cache.values
        
        num_kv_heads = int(getattr(self._text_config, "num_key_value_heads", 1))
        head_dim = int(
            getattr(self.self_attn, "head_dim", None)
            or getattr(self._text_config, "head_dim", None)
            or stored_keys.shape[-1] if len(stored_keys.shape) >= 1 else None
        )
        
        expected_shape = (batch_size, num_kv_heads, expected_seq_len, head_dim)
        
        # Check if shapes match expected
        if stored_keys.shape == expected_shape and stored_values.shape == expected_shape:
            return True
        
        # Log the mismatch
        logger.warning(
            f"Cache dimension mismatch detected: "
            f"keys={stored_keys.shape}, values={stored_values.shape}, "
            f"expected={expected_shape}"
        )
        
        # Try to reshape if element counts match
        if stored_keys.numel() == batch_size * num_kv_heads * expected_seq_len * head_dim:
            try:
                stored_keys = stored_keys.view(expected_shape).contiguous()
                stored_values = stored_values.view(expected_shape).contiguous()
                cache.update(stored_keys, stored_values, self.layer_idx)
                logger.info(f"Fixed cache dimensions to {expected_shape}")
                return False
            except Exception as e:
                logger.error(f"Failed to fix cache dimensions: {e}")
                return False
        
        return False

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
        initial_past_length = None  # Store for error reporting
        cache = None
        if layer_past is not None:
            # Derive past_length from actual tensor shape to handle cache mismatches
            # layer_past[0] is key cache in bloom format: [B*Hkv, D, T]
            # Use the actual T dimension from the tensor, not a potentially incorrect parameter
            key_cache_shape = layer_past[0].shape
            if len(key_cache_shape) == 3:
                # Bloom format: [B*Hkv, D, T]
                initial_past_length = int(key_cache_shape[-1])
            else:
                # Fallback: try to infer from shape
                initial_past_length = int(key_cache_shape[-1]) if key_cache_shape else 0
            cache = DynamicCache()
            # Pass initial_past_length as hint, but _reorder_cache_from_bloom will use actual tensor shape
            key, value = self._reorder_cache_from_bloom(layer_past, batch_size, initial_past_length)
            # Update past_length to match the actual sequence length after reordering (may be truncated)
            past_length = key.shape[2]  # [B, Hkv, T, D] -> T is at index 2
            # Validate cache shapes match expected format before updating
            assert key.shape[0] == batch_size, f"Key cache batch size mismatch: {key.shape[0]} != {batch_size}"
            assert value.shape[0] == batch_size, f"Value cache batch size mismatch: {value.shape[0]} != {batch_size}"
            assert key.shape[2] == value.shape[2], f"Key and value cache sequence length mismatch: {key.shape[2]} != {value.shape[2]}"
            # Ensure key and value have matching shapes in all dimensions
            assert key.shape == value.shape, f"Key and value cache shape mismatch: key={key.shape}, value={value.shape}"
            # Log if truncation occurred
            if past_length != initial_past_length:
                logger.warning(
                    f"Cache truncated from {initial_past_length} to {past_length} tokens. "
                    f"Key shape: {key.shape}, Value shape: {value.shape}"
                )
            cache.update(key, value, self.layer_idx)
            # Verify the cache was stored correctly by checking what we can retrieve
            stored_cache = cache.layers[self.layer_idx]
            stored_keys = stored_cache.keys
            stored_values = stored_cache.values
            
            if stored_keys.shape != key.shape or stored_values.shape != value.shape:
                logger.error(
                    f"Cache shape mismatch after storage: stored keys={stored_keys.shape} (expected {key.shape}), "
                    f"stored values={stored_values.shape} (expected {value.shape}). "
                    f"This will cause dimension mismatch in upstream code."
                )
                # Try to fix by re-updating with correct shapes
                logger.warning("Attempting to fix cache by re-updating with correct shapes...")
                cache.update(key, value, self.layer_idx)
                # Re-verify after fix attempt
                stored_cache = cache.layers[self.layer_idx]
                if stored_cache.keys.shape != key.shape or stored_cache.values.shape != value.shape:
                    raise RuntimeError(
                        f"Failed to store cache with correct shapes. "
                        f"Stored: keys={stored_cache.keys.shape}, values={stored_cache.values.shape}. "
                        f"Expected: keys={key.shape}, values={value.shape}"
                    )
            
            # Verify DynamicCache sequence length tracking
            try:
                cache_seq_length = cache.get_seq_length(self.layer_idx)
                if cache_seq_length != past_length:
                    logger.warning(
                        f"DynamicCache.get_seq_length() returns {cache_seq_length} but actual cache has {past_length} tokens. "
                        f"This may cause issues in upstream code that relies on get_seq_length()."
                    )
            except Exception as e:
                logger.debug(f"Could not verify cache.get_seq_length(): {e}")
            
            # Log detailed cache information
            logger.debug(
                f"Cache stored successfully: keys={stored_keys.shape}, values={stored_values.shape}, "
                f"past_length={past_length} (truncated from {initial_past_length})"
            )
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

        # Validate cache dimensions before passing to upstream code
        if cache is not None and self.layer_idx in cache.layers:
            stored_cache = cache.layers[self.layer_idx]
            stored_keys = stored_cache.keys
            stored_values = stored_cache.values
            
            # Expected format: [B, Hkv, T, D]
            expected_batch = batch_size
            num_kv_heads = int(getattr(self._text_config, "num_key_value_heads", 1))
            expected_seq_len = past_length
            head_dim = int(getattr(self.self_attn, "head_dim", None) or getattr(self._text_config, "head_dim", None) or stored_keys.shape[-1] if len(stored_keys.shape) >= 1 else None)
            
            # Validate dimensions
            if len(stored_keys.shape) != 4 or len(stored_values.shape) != 4:
                raise ValueError(
                    f"Cache tensors must be 4D [B, Hkv, T, D]. "
                    f"Got keys shape: {stored_keys.shape}, values shape: {stored_values.shape}"
                )
            
            key_batch, key_heads, key_seq, key_dim = stored_keys.shape
            value_batch, value_heads, value_seq, value_dim = stored_values.shape
            
            # Check batch size
            if key_batch != expected_batch or value_batch != expected_batch:
                raise ValueError(
                    f"Cache batch size mismatch: keys={key_batch}, values={value_batch}, expected={expected_batch}"
                )
            
            # Check number of KV heads
            if key_heads != num_kv_heads or value_heads != num_kv_heads:
                logger.warning(
                    f"Cache KV heads mismatch: keys={key_heads}, values={value_heads}, expected={num_kv_heads}. "
                    f"This may cause dimension mismatch in upstream code."
                )
            
            # Check sequence length
            if key_seq != expected_seq_len or value_seq != expected_seq_len:
                logger.warning(
                    f"Cache sequence length mismatch: keys={key_seq}, values={value_seq}, expected={expected_seq_len}. "
                    f"This may cause dimension mismatch in upstream code."
                )
            
            # Check head dimension
            if head_dim is not None and (key_dim != head_dim or value_dim != head_dim):
                logger.warning(
                    f"Cache head dimension mismatch: keys={key_dim}, values={value_dim}, expected={head_dim}. "
                    f"This may cause dimension mismatch in upstream code."
                )
            
            # Ensure key and value have matching shapes
            if stored_keys.shape != stored_values.shape:
                raise ValueError(
                    f"Cache key and value shape mismatch: keys={stored_keys.shape}, values={stored_values.shape}. "
                    f"Both must have shape [B={expected_batch}, Hkv={num_kv_heads}, T={expected_seq_len}, D={head_dim}]"
                )
            
            # Log cache shapes for debugging
            logger.debug(
                f"Cache validated before upstream forward: "
                f"keys={stored_keys.shape}, values={stored_values.shape}, "
                f"past_length={past_length}, seq_length={seq_length}, "
                f"cache_position range=[{cache_position[0].item()}, {cache_position[-1].item()}]"
            )
        
        # Final validation and fix attempt before upstream forward
        if cache is not None:
            cache_was_valid = self._validate_and_fix_cache_dimensions(cache, batch_size, past_length)
            # Re-verify cache after potential fix (whether it was valid or fixed)
            if self.layer_idx in cache.layers:
                stored_cache = cache.layers[self.layer_idx]
                logger.debug(
                    f"Final cache check before upstream forward: "
                    f"keys={stored_cache.keys.shape}, values={stored_cache.values.shape}, "
                    f"was_valid={cache_was_valid}"
                )

        # Wrap upstream forward in try-catch to provide better error messages
        try:
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
        except RuntimeError as e:
            if "Sizes of tensors must match" in str(e) or "Expected size" in str(e):
                # Provide detailed error information for dimension mismatch
                error_msg = str(e)
                if cache is not None and self.layer_idx in cache.layers:
                    stored_cache = cache.layers[self.layer_idx]
                    stored_keys = stored_cache.keys
                    stored_values = stored_cache.values
                    error_msg += (
                        f"\nCache state when error occurred: "
                        f"keys shape={stored_keys.shape}, values shape={stored_values.shape}, "
                        f"past_length={past_length}, seq_length={seq_length}, "
                        f"batch_size={batch_size}, "
                        f"cache_position range=[{cache_position[0].item()}, {cache_position[-1].item()}]"
                    )
                    # Check if truncation occurred
                    if initial_past_length is not None and past_length != initial_past_length:
                        error_msg += (
                            f"\nNote: Cache was truncated from {initial_past_length} to {past_length} tokens. "
                            f"This truncation may be causing the dimension mismatch."
                        )
                raise RuntimeError(error_msg) from e
            else:
                # Re-raise other RuntimeErrors as-is
                raise

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
        # After permute: key_states is [B*Hkv, T, D], so D = head_dim
        shape_head_dim = key_states.shape[-1] if len(key_states.shape) >= 2 else None
        config_head_dim = int(getattr(self.self_attn, "head_dim", None) or getattr(self._text_config, "head_dim", None) or 0)
        head_dim = shape_head_dim if shape_head_dim is not None else config_head_dim
        
        if head_dim == 0:
            raise ValueError(
                f"Cannot determine head_dim: config={config_head_dim}, shape={shape_head_dim}, "
                f"key_states.shape={key_states.shape}"
            )
        
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
        
        # If head_dim from config doesn't match shape, use shape-based value and log warning
        if config_head_dim > 0 and head_dim != config_head_dim:
            logger.warning(
                f"Head dim mismatch: config says {config_head_dim} but tensor shape suggests {head_dim}. "
                f"Using shape-based value {head_dim}."
            )
        
        divisor = batch_size * num_kv_heads * head_dim
        if divisor == 0:
            raise ValueError(f"Invalid divisor: batch_size={batch_size}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")
        
        if total_elements % divisor != 0:
            raise ValueError(
                f"Cannot reshape cache: total elements ({total_elements}) is not divisible by "
                f"batch*kv_heads*head_dim ({divisor} = {batch_size}*{num_kv_heads}*{head_dim}). "
                f"Tensor shape: {key_states.shape}"
            )
        key_seq_length = total_elements // divisor
        
        # Calculate value sequence length from element count for consistency
        value_total_elements = value_states.numel()
        if value_total_elements % divisor != 0:
            raise ValueError(
                f"Cannot reshape value cache: total elements ({value_total_elements}) is not divisible by "
                f"batch*kv_heads*head_dim ({divisor} = {batch_size}*{num_kv_heads}*{head_dim}). "
                f"Tensor shape: {value_states.shape}"
            )
        value_seq_length = value_total_elements // divisor
        
        # Use the minimum of key and value sequence lengths to ensure consistency
        # Both caches should have the same sequence length, so if they differ, use the smaller one
        actual_seq_length = min(key_seq_length, value_seq_length)
        
        if key_seq_length != value_seq_length:
            logger.warning(
                f"Cache sequence length mismatch: key cache has {key_seq_length} tokens, "
                f"value cache has {value_seq_length} tokens. Using minimum ({actual_seq_length}) for both."
            )
        
        if actual_seq_length != seq_length:
            logger.warning(
                f"Cache sequence length mismatch: expected {seq_length} from past_length, "
                f"but calculated {actual_seq_length} tokens from element counts. Using calculated size. "
                f"Key cache: {key_seq_length} tokens, Value cache: {value_seq_length} tokens, "
                f"Using minimum: {actual_seq_length}. This truncation may cause dimension mismatches in upstream code."
            )
        
        # Double-check the reshape will work before attempting it
        expected_view_elements = batch_size * num_kv_heads * actual_seq_length * head_dim
        if expected_view_elements > total_elements:
            raise ValueError(
                f"Key cache has insufficient elements ({total_elements}) for shape "
                f"[{batch_size}, {num_kv_heads}, {actual_seq_length}, {head_dim}] "
                f"(needs {expected_view_elements}). Key shape: {key_states.shape}"
            )
        if expected_view_elements > value_total_elements:
            raise ValueError(
                f"Value cache has insufficient elements ({value_total_elements}) for shape "
                f"[{batch_size}, {num_kv_heads}, {actual_seq_length}, {head_dim}] "
                f"(needs {expected_view_elements}). Value shape: {value_states.shape}"
            )
        
        # Reshape key cache - truncate if necessary
        if key_seq_length > actual_seq_length:
            # Need to truncate key cache
            key_states = key_states.view(batch_size * num_kv_heads, key_seq_length, head_dim)
            key_states = key_states[:, :actual_seq_length, :].contiguous()
        
        # Use try-except to provide better error message if view still fails
        try:
            key_states = key_states.view(batch_size, num_kv_heads, actual_seq_length, head_dim)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to reshape key cache to [{batch_size}, {num_kv_heads}, {actual_seq_length}, {head_dim}]: "
                f"tensor has {key_states.numel()} elements, shape={key_states.shape}. Original error: {e}"
            ) from e
        
        # Reshape value cache - truncate if necessary
        if value_seq_length > actual_seq_length:
            # Need to truncate value cache
            value_states = value_states.view(batch_size * num_kv_heads, value_seq_length, head_dim)
            value_states = value_states[:, :actual_seq_length, :].contiguous()
        
        # Use try-except to provide better error message if view still fails
        try:
            value_states = value_states.view(batch_size, num_kv_heads, actual_seq_length, head_dim)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to reshape value cache to [{batch_size}, {num_kv_heads}, {actual_seq_length}, {head_dim}]: "
                f"tensor has {value_states.numel()} elements, shape={value_states.shape}. Original error: {e}"
            ) from e
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


