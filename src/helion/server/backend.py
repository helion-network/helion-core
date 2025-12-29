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

        # Track the *effective* KV cache length for each active inference cache (per session).
        # This is needed for models/layers that return truncated KV caches (e.g., sliding-window attention).
        # Keyed by cache handle tuple (unique per allocated inference cache).
        self._kv_cache_lengths: Dict[Tuple[int, ...], int] = {}

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
            layer_past = self._select_layer_past(
                cache_tensors, prefix_length=inference_info.prefix_length, cache_handles=inference_info.cache_handles
            )

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

            # Write KV caches back into storage.
            # IMPORTANT: some models/layers can return a KV cache shorter than (prefix_length + seq_len)
            # (e.g., sliding window). We align the returned cache to the *end position*.
            end_pos = inference_info.prefix_length + seq_len
            self._update_cache_inplace(cache_tensors, new_kvs, end_pos=end_pos, cache_handles=inference_info.cache_handles)

            return (output_hidden_states,)

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

    def _select_layer_past(
        self, cache_tensors: Sequence[torch.Tensor], *, prefix_length: int, cache_handles: Tuple[int, ...]
    ) -> Sequence[torch.Tensor]:
        """Extract past KV tensors for this step, accounting for models that keep a truncated KV cache."""
        key_cache, value_cache = list(cache_tensors[0::2]), list(cache_tensors[1::2])

        # If we previously observed that this cache is truncated (e.g., sliding-window),
        # only feed that many tokens from the *end* of the prefix.
        kv_len = int(self._kv_cache_lengths.get(cache_handles, prefix_length))
        kv_len = min(int(prefix_length), kv_len)
        start = int(prefix_length) - kv_len
        if start < 0:
            start = 0

        for i in range(len(key_cache)):
            # key cache tensor: [B, Hkv, D, max_len] -> slice along last dim, then flatten
            key_slice = key_cache[i][..., start:prefix_length].contiguous()
            key_cache[i] = key_slice.flatten(0, 1)  # [B*Hkv, D, kv_len]

            # value cache tensor: [B, Hkv, max_len, D] -> slice along token dim, then flatten
            value_slice = value_cache[i][:, :, start:prefix_length, :].contiguous()
            value_cache[i] = value_slice.flatten(0, 1)  # [B*Hkv, kv_len, D]

        layer_past = tuple(chain(*zip(key_cache, value_cache)))
        return PerDeviceTensors(*layer_past) if len(self.module.module_shards) > 1 else layer_past

    def _update_cache_inplace(
        self,
        cache_tensors: Sequence[torch.Tensor],
        new_kvs: Sequence[torch.Tensor],
        *,
        end_pos: int,
        cache_handles: Tuple[int, ...],
    ) -> None:
        """
        Writes new key/value tensors back into cache, works in-place.

        Unlike the old implementation, this supports models that return a KV cache shorter than `end_pos`
        (e.g., sliding-window attention). In that case, we align the returned cache window to `end_pos`.
        """
        if not new_kvs:
            return

        # Infer returned KV length for this step from bloom-format key: [B*Hkv, D, T]
        returned_lengths = []
        for new_key in new_kvs[0::2]:
            if new_key.ndim != 3:
                continue
            returned_lengths.append(int(new_key.shape[-1]))
        if returned_lengths:
            returned_kv_len = min(returned_lengths)
            if max(returned_lengths) != returned_kv_len:
                logger.warning(f"Inconsistent returned KV lengths across shards: {returned_lengths}, using {returned_kv_len}")
        else:
            returned_kv_len = int(end_pos)

        # Persist effective KV length for subsequent steps (per cache/session).
        self._kv_cache_lengths[cache_handles] = returned_kv_len

        start_pos = int(end_pos) - int(returned_kv_len)
        if start_pos < 0:
            # This shouldn't happen, but be safe.
            start_pos = 0

        # Write keys: cache_key is [B, Hkv, D, max_len], new_key is [B*Hkv, D, returned_kv_len]
        for cache_key, new_key in zip(cache_tensors[0::2], new_kvs[0::2]):
            if new_key.ndim != 3:
                continue
            _, head_dim, kv_len = new_key.shape
            max_len = int(cache_key.shape[-1])
            if end_pos > max_len:
                raise ValueError(f"end_pos={end_pos} exceeds cache length={max_len}")

            kv_len = int(kv_len)
            write_start = start_pos
            write_end = int(end_pos)
            write_len = write_end - write_start
            # Reshape to cache layout
            new_key_view = new_key.view(*cache_key.shape[:3], kv_len)
            if kv_len == write_len:
                cache_key[:, :, :, write_start:write_end] = new_key_view
            else:
                # If the returned KV cache is longer than the requested write span (can happen if start_pos was clamped),
                # align to the *end* of the returned cache.
                cache_key[:, :, :, write_start:write_end] = new_key_view[:, :, :, -write_len:]

        # Write values: cache_value is [B, Hkv, max_len, D], new_value is [B*Hkv, returned_kv_len, D]
        for cache_value, new_value in zip(cache_tensors[1::2], new_kvs[1::2]):
            if new_value.ndim != 3:
                continue
            _, kv_len, head_dim = new_value.shape
            max_len = int(cache_value.shape[-2])
            if end_pos > max_len:
                raise ValueError(f"end_pos={end_pos} exceeds cache length={max_len}")

            kv_len = int(kv_len)
            write_start = start_pos
            write_end = int(end_pos)
            write_len = write_end - write_start
            new_value_view = new_value.view(*cache_value.shape[:2], kv_len, head_dim)
            if kv_len == write_len:
                cache_value[:, :, write_start:write_end, :] = new_value_view
            else:
                cache_value[:, :, write_start:write_end, :] = new_value_view[:, :, -write_len:, :]

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
        for inference_info, optional_prompt in zip(inference_infos, optional_prompts):
            if optional_prompt is not None:
                hidden_states[:, : optional_prompt.shape[1]] += optional_prompt
            (hidden_states,) = self.backends[inference_info.uid].inference_step(
                hidden_states, hypo_ids, inference_info, token_type_ids=token_type_ids
            )
        return (hidden_states,)
