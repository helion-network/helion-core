from __future__ import annotations

from collections import Counter
from itertools import chain
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from hivemind import BatchTensorDescriptor, TensorDescriptor
from hivemind.moe.expert_uid import ExpertUID
from hivemind.moe.server.module_backend import ModuleBackend
from hivemind.utils import get_logger
from tensor_parallel import TensorParallel
from tensor_parallel.tensor_parallel import PerDeviceTensors
from transformers import PretrainedConfig

from helion.data_structures import InferenceMetadata
from helion.server.memory_cache import MemoryCache
from helion.server.task_pool import PrioritizedTaskPool
from helion.utils.misc import get_size_in_bytes, is_dummy

logger = get_logger(__name__)


class TransformerBackend(ModuleBackend):
    """A wrapper for a transformer block that can process requests for forward, backward and inference"""

    _peft_module = None

    def __init__(
        self,
        *args,
        config: PretrainedConfig,
        memory_cache: MemoryCache,
        backend_dtype: torch.dtype,
        max_chunk_size_bytes: int,
        **kwargs,
    ):
        import helion.utils.peft as _peft_module

        self._peft_module = _peft_module

        super().__init__(*args, **kwargs)
        assert isinstance(self.module, TensorParallel)
        self.config = config
        self.memory_cache = memory_cache
        self.max_chunk_size_bytes = max_chunk_size_bytes

        for name, param in self.module.named_parameters():
            assert not param.requires_grad, f"Block parameters must not accumulate gradients, but {name} does"
        for name, buf in self.module.named_buffers():
            assert not buf.requires_grad, f"Block parameters must not accumulate gradients, but {name} does"

        max_batch_size = self.forward_pool.max_batch_size
        device = self.module.devices[self.module.output_device_index]
        self.inference_pool = PrioritizedTaskPool(
            self.inference_step, max_batch_size=max_batch_size, device=device, name=f"{self.name}_inference"
        )  # note: inference_pools may be merged later, see merge_inference_pools_inplace
        self.forward_pool = PrioritizedTaskPool(
            self.forward, max_batch_size=max_batch_size, device=device, name=f"{self.name}_forward"
        )
        self.backward_pool = PrioritizedTaskPool(
            self.backward, max_batch_size=max_batch_size, device=device, name=f"{self.name}_backward"
        )

        self.dtype = backend_dtype
        self.dtype_bytes = get_size_in_bytes(self.dtype)
        self.shard_num_heads = []
        for shard in self.module.module_shards:
            for submodule in shard.modules():
                if isinstance(submodule, config.attn_class):
                    heads = getattr(submodule, "num_heads", None)
                    if heads is None:
                        heads = getattr(submodule, "num_attention_heads", None)
                    if heads is None:
                        heads = getattr(config, "num_attention_heads", None)
                    if heads is None:
                        raise AttributeError(
                            f"Cannot determine number of attention heads for {type(submodule).__name__}"
                        )
                    self.shard_num_heads.append(heads)
        assert len(self.shard_num_heads) == len(self.module.devices)
        assert sum(self.shard_num_heads) == config.num_attention_heads

        self.inference_schema = (
            (
                *self.args_schema,
                BatchTensorDescriptor((), dtype=self.dtype),
                BatchTensorDescriptor((), dtype=torch.int64),
            ),
            self.kwargs_schema,
        )

        self.cache_bytes_per_token: Dict[torch.device, int] = Counter()
        for descr in self.get_inference_cache_descriptors(batch_size=1, max_length=1):
            self.cache_bytes_per_token[descr.device] += descr.numel() * get_size_in_bytes(descr.dtype)

    def get_inference_cache_descriptors(self, batch_size: int, max_length: int) -> Sequence[TensorDescriptor]:
        """Create tensor descriptors for attention cache tensors used during inference_step"""
        # Prefer explicit head_dim if provided by the model config (e.g., Qwen3),
        # otherwise fall back to hidden_size // num_attention_heads.
        head_dim = int(getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads))
        cache_tensors = []
        for device, num_heads in zip(self.module.devices, self.shard_num_heads):
            num_heads //= self.config.num_key_value_groups
            if hasattr(self.config, "num_key_value_heads"):
                num_heads = self.config.num_key_value_heads
            keys = TensorDescriptor((batch_size, num_heads, head_dim, max_length), dtype=self.dtype, device=device)
            values = TensorDescriptor((batch_size, num_heads, max_length, head_dim), dtype=self.dtype, device=device)
            cache_tensors.extend((keys, values))
        return cache_tensors

    def forward(self, *inputs: Union[torch.Tensor, str]) -> Tuple[torch.Tensor, ...]:
        *inputs, active_adapter = inputs
        with self._peft_module.using_adapter(active_adapter):
            return super().forward(*inputs)

    def backward(self, *inputs: Union[torch.Tensor, str]) -> Tuple[torch.Tensor, ...]:
        *inputs, active_adapter = inputs
        with self._peft_module.using_adapter(active_adapter):
            return super().backward(*inputs)

    @torch.inference_mode()
    def inference_step(
        self,
        hidden_states: torch.Tensor,
        hypo_ids: torch.LongTensor,
        inference_info: InferenceMetadata,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        assert hidden_states.ndim == 3, "expected hidden states to be 3-dimensional: [batch_size, seq_len, hid_size]"
        seq_len = hidden_states.shape[1]

        with self.memory_cache.use_cache(
            *inference_info.cache_handles
        ) as cache_tensors, self._peft_module.using_adapter(inference_info.active_adapter):
            self._reorder_cache_inplace(cache_tensors, hypo_ids)

            # We chunk the inputs so that peak memory for long sequences fits into `autograd_memory`
            # reserved in `Server._choose_num_blocks()`. This saves us from OOMs if `max_chunk_size_bytes`
            # is at least 4-6x less than `autograd_memory`.
            max_chunk_length = self._estimate_max_chunk_length(hidden_states, inference_info)
            # Gemma3 multimodal masking needs consistent q/kv indices; chunking would break it.
            if token_type_ids is not None and not is_dummy(token_type_ids) and hidden_states.shape[1] > 1:
                max_chunk_length = hidden_states.shape[1]
            output_hidden_states = torch.empty_like(hidden_states) if seq_len > max_chunk_length else None
            layer_past, actual_prefix_length = self._select_layer_past(cache_tensors, inference_info.prefix_length)
            
            # Track actual cache size - start with what we selected from cache
            # This may be less than prefix_length if cache was truncated
            actual_cache_size = actual_prefix_length
            
            for offset in range(0, seq_len, max_chunk_length):
                chunk_length = min(max_chunk_length, seq_len - offset)
                hidden_states_chunk = hidden_states[:, offset : offset + chunk_length, :]
                if token_type_ids is not None and not is_dummy(token_type_ids):
                    output_hidden_states_chunk, new_kvs = self.module.forward(
                        hidden_states_chunk, layer_past=layer_past, use_cache=True, token_type_ids=token_type_ids
                    )
                else:
                    output_hidden_states_chunk, new_kvs = self.module.forward(
                        hidden_states_chunk, layer_past=layer_past, use_cache=True
                    )
                if seq_len > max_chunk_length:
                    output_hidden_states[:, offset : offset + chunk_length] = output_hidden_states_chunk
                else:
                    output_hidden_states = output_hidden_states_chunk  # saves one memcopy
                layer_past = new_kvs
                
                # Check if cache was truncated by examining the returned cache size
                # new_kvs[0] is key cache in bloom format: [B*Hkv, D, T]
                if new_kvs and len(new_kvs) >= 2:
                    returned_key = new_kvs[0]
                    if len(returned_key.shape) == 3:
                        # Bloom format: [B*Hkv, D, T], T is at index 2
                        returned_cache_size = int(returned_key.shape[2])
                        # Expected size after processing this chunk
                        expected_size = actual_cache_size + chunk_length
                        # If returned size is less than expected, truncation occurred
                        if returned_cache_size < expected_size:
                            # Calculate what the actual cache size was before this chunk
                            actual_cache_size = returned_cache_size - chunk_length
                            logger.debug(
                                f"Detected cache truncation: actual cache size is {actual_cache_size} "
                                f"(expected {inference_info.prefix_length + offset})"
                            )
                        else:
                            # Update actual_cache_size to reflect the new chunk
                            actual_cache_size = returned_cache_size

            # Use actual_cache_size for cache update to ensure consistency
            self._update_cache_inplace(cache_tensors, new_kvs, actual_cache_size)
            
            # Return both hidden_states and actual_cache_size so caller can update prefix_length
            return (output_hidden_states, actual_cache_size)

    def _estimate_max_chunk_length(self, hidden_states: torch.Tensor, inference_info: InferenceMetadata) -> int:
        # We assume that attention logit matrices are the main thing that consumes memory, given that
        # the model uses multi-query attention
        batch_size, seq_length, hidden_size = hidden_states.shape
        worst_case_length = inference_info.prefix_length + seq_length
        attn_bytes_per_token = max(self.shard_num_heads) * batch_size * self.dtype_bytes * worst_case_length
        return max(1, self.max_chunk_size_bytes // attn_bytes_per_token)

    def _reorder_cache_inplace(self, cache_tensors: torch.Tensor, hypo_ids: torch.Tensor):
        """If hypo_ids is specified, reorder elements of each cache tensor in-place by taking indices from hypo_ids"""
        if not is_dummy(hypo_ids):
            for cache_tensor in cache_tensors:
                cache_tensor[...] = cache_tensor[hypo_ids.to(cache_tensor.device)]  # in-place reorder cache by hypo ids

    def _select_layer_past(self, cache_tensors: Sequence[torch.Tensor], prefix_length: int) -> Tuple[Sequence[torch.Tensor], int]:
        """
        Extract first {prefix_length} tokens and reshape them such that they can be used as layer_past.
        Returns (layer_past, actual_selected_length) where actual_selected_length is the actual number
        of tokens selected (may be less than prefix_length if cache is truncated).
        """
        key_cache, value_cache = list(cache_tensors[0::2]), list(cache_tensors[1::2])
        min_actual_seq_len = None
        
        for i in range(len(key_cache)):
            # Flatten cache: [batch, num_kv_heads, seq_len, head_dim] -> [batch*num_kv_heads, seq_len, head_dim]
            key_flat = key_cache[i].flatten(0, 1)
            value_flat = value_cache[i].flatten(0, 1)
            
            # Get actual sequence length from value cache shape (more reliable than key cache which may be in bloom format)
            # Value shape after flatten: [batch*num_kv_heads, seq_len, head_dim]
            # For value cache, seq_len is at index 1
            if len(value_flat.shape) >= 2:
                actual_seq_len = int(value_flat.shape[1])
            elif len(key_flat.shape) == 3:
                # Key is in bloom format: [B*Hkv, D, T], so T is at index 2
                actual_seq_len = int(key_flat.shape[2])
            else:
                # Fallback: try to infer from total elements
                actual_seq_len = value_flat.shape[1] if len(value_flat.shape) >= 2 else key_flat.shape[-1] if len(key_flat.shape) >= 1 else 0
                actual_seq_len = int(actual_seq_len)
            
            # Track minimum actual sequence length across all layers
            if min_actual_seq_len is None:
                min_actual_seq_len = actual_seq_len
            else:
                min_actual_seq_len = min(min_actual_seq_len, actual_seq_len)
            
            # Clamp prefix_length to actual cache size to avoid shape mismatches
            safe_prefix_length = min(prefix_length, actual_seq_len) if actual_seq_len > 0 else prefix_length
            
            if safe_prefix_length != prefix_length and actual_seq_len > 0:
                logger.warning(
                    f"Cache prefix_length mismatch: requested {prefix_length} but cache only has {actual_seq_len} tokens. "
                    f"Using {safe_prefix_length} instead."
                )
            
            # For key: [batch*num_kv_heads, head_dim, seq_len] -> slice to [:,:,:safe_prefix_length]
            # For value: [batch*num_kv_heads, seq_len, head_dim] -> slice to [:, :safe_prefix_length, :]
            if len(key_flat.shape) == 3:
                # Key is in bloom format: [B*Hkv, D, T]
                # Ensure we don't slice beyond actual size
                max_slice = min(safe_prefix_length, key_flat.shape[2])
                if max_slice < key_flat.shape[2]:
                    key_cache[i] = key_flat[:, :, :max_slice].contiguous()
                else:
                    key_cache[i] = key_flat
            else:
                # Fallback for other formats
                max_slice = min(safe_prefix_length, key_flat.shape[1] if len(key_flat.shape) >= 2 else safe_prefix_length)
                if max_slice < (key_flat.shape[1] if len(key_flat.shape) >= 2 else key_flat.shape[0]):
                    key_cache[i] = key_flat[:, :max_slice].contiguous()
                else:
                    key_cache[i] = key_flat
            
            # Ensure value slice doesn't exceed actual size
            max_value_slice = min(safe_prefix_length, value_flat.shape[1] if len(value_flat.shape) >= 2 else safe_prefix_length)
            if max_value_slice < value_flat.shape[1]:
                value_cache[i] = value_flat[:, :max_value_slice].contiguous()
            else:
                value_cache[i] = value_flat
            
            # Final shape: key [batch * num_kv_heads, head_dim, kv_length] (bloom format)
            # Final shape: value [batch * num_kv_heads, kv_length, head_dim]
        
        layer_past = tuple(chain(*zip(key_cache, value_cache)))
        result = PerDeviceTensors(*layer_past) if len(self.module.module_shards) > 1 else layer_past
        actual_selected_length = min_actual_seq_len if min_actual_seq_len is not None else prefix_length
        return result, min(prefix_length, actual_selected_length)

    def _update_cache_inplace(
        self, cache_tensors: Sequence[torch.Tensor], new_kvs: Sequence[torch.Tensor], prefix_length: int
    ):
        """Writes new key/value tensors back into cache, works in-place"""
        _batch_size_times_num_kv_heads, head_dim, new_length = new_kvs[0].shape
        for cache_key, new_key in zip(cache_tensors[0::2], new_kvs[0::2]):
            new_key = new_key.view(*cache_key.shape[:3], new_length)
            cache_key[:, :, :, prefix_length:new_length] = new_key[:, :, :, prefix_length:new_length]
        for cache_value, new_value in zip(cache_tensors[1::2], new_kvs[1::2]):
            new_value = new_value.view(*cache_value.shape[:2], new_length, head_dim)
            cache_value[:, :, prefix_length:new_length, :] = new_value[:, :, prefix_length:new_length, :]

    def get_pools(self) -> Sequence[PrioritizedTaskPool]:
        return self.forward_pool, self.backward_pool, self.inference_pool

    def get_info(self) -> Dict[str, Any]:
        """Get module parameters and stats. Used by RemoteExpert to check shapes and for DMoE orchestration."""
        return dict(super().get_info(), inference_schema=self.inference_schema)

    def shutdown(self):
        # Break the cyclic references, otherwise TransformerBackend may be not garbage-collected
        self.forward_pool = self.backward_pool = self.inference_pool = None

        # Explicitly free the GPU memory. This is not necessary at the time this code is written,
        # but may help to avoid future issues when the module is not garbage-collected for some reasons
        dummy = torch.tensor([])
        for p in self.module.parameters():
            p.data = dummy


def merge_inference_pools_inplace(backends: Dict[ExpertUID, TransformerBackend]):
    """Replace each backend's rpc_inference pools with a combined pool runs multiple blocks in one call"""
    assert len(backends) != 0 and all(isinstance(b, TransformerBackend) for b in backends.values())
    first_pool = next(iter(backends.values())).inference_pool
    merged_pool = PrioritizedTaskPool(
        _MergedInferenceStep(backends),
        max_batch_size=first_pool.max_batch_size,
        device=first_pool.device,
        name=f"merged_inference",
    )
    for backend in backends.values():
        assert not backend.inference_pool.is_alive()
        backend.inference_pool = merged_pool


class _MergedInferenceStep:
    def __init__(self, backends: Dict[ExpertUID, TransformerBackend]):
        self.backends = backends

    @torch.inference_mode()
    def __call__(
        self,
        hidden_states: torch.Tensor,
        hypo_ids: torch.LongTensor,
        token_type_ids: torch.Tensor,
        inference_infos: Sequence[InferenceMetadata],
        *optional_prompts: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        assert len(inference_infos) == len(
            optional_prompts
        ), f"found {len(inference_infos)} blocks but {len(optional_prompts)} prompts"
        min_actual_cache_size = None
        for inference_info, optional_prompt in zip(inference_infos, optional_prompts):
            if optional_prompt is not None:
                hidden_states[:, : optional_prompt.shape[1]] += optional_prompt
            result = self.backends[inference_info.uid].inference_step(
                hidden_states, hypo_ids, inference_info, token_type_ids=token_type_ids
            )
            # Handle both old (single value) and new (tuple with cache size) return formats
            if len(result) == 2:
                hidden_states, actual_cache_size = result
                # Track minimum cache size across all backends
                if min_actual_cache_size is None:
                    min_actual_cache_size = actual_cache_size
                else:
                    min_actual_cache_size = min(min_actual_cache_size, actual_cache_size)
            else:
                (hidden_states,) = result
        
        # Return both hidden_states and actual_cache_size if available
        if min_actual_cache_size is not None:
            return (hidden_states, min_actual_cache_size)
        else:
            return (hidden_states,)
