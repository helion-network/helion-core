"""
Bloom intermediate layer
Based on https://github.com/huggingface/transformers/commit/ca2a55e9dfb245527b5e1c954fec6ffbb7aef07b
See commit history for authorship.
"""
from typing import Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.bloom.modeling_bloom import BloomBlock, build_alibi_tensor

from helion.utils.misc import is_dummy


class WrappedBloomBlock(BloomBlock):
    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        alibi: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ):
        assert attention_mask is None, "Non-causal attention masks are not supported yet"
        batch_size, seq_length = hidden_states.shape[:2]

        cache = None
        if layer_past is not None:
            if is_dummy(layer_past[0]):
                layer_past = None
            else:
                cache = DynamicCache()
                key_states, value_states = self._reorder_cache_from_bloom(layer_past, batch_size)
                cache.update(key_states, value_states, self.self_attention.layer_idx)
        if cache is None and use_cache:
            cache = DynamicCache()

        past_length = 0 if layer_past is None else layer_past[0].shape[-1]
        seq_length_with_past = seq_length + past_length

        cache_position = torch.arange(past_length, seq_length_with_past, device=hidden_states.device)
        attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
        if alibi is None:
            alibi = build_alibi_tensor(attention_mask, num_heads=self.num_heads, dtype=hidden_states.dtype)
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask=attention_mask,
            input_shape=(batch_size, seq_length),
            inputs_embeds=hidden_states,
            past_key_values_length=past_length,
        ).bool()

        hidden_states, _ = super().forward(
            hidden_states,
            *args,
            attention_mask=attention_mask,
            alibi=alibi,
            layer_past=cache,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        if use_cache:
            present = self._reorder_cache_to_bloom(cache, batch_size, seq_length_with_past)
            return hidden_states, present
        return (hidden_states,)

    def _reorder_cache_from_bloom(
        self, key_value: Tuple[torch.Tensor, torch.Tensor], batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key_states, value_states = key_value
        num_heads = self.self_attention.num_heads
        head_dim = self.self_attention.head_dim
        seq_length = key_states.shape[-1]

        key_states = key_states.view(batch_size, num_heads, head_dim, seq_length).permute(0, 1, 3, 2)
        value_states = value_states.view(batch_size, num_heads, seq_length, head_dim)
        return key_states, value_states

    def _reorder_cache_to_bloom(
        self, cache: DynamicCache, batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        layer_cache = cache.layers[self.self_attention.layer_idx]
        key_states = layer_cache.keys
        value_states = layer_cache.values
        num_heads = self.self_attention.num_heads
        head_dim = self.self_attention.head_dim

        key_states = key_states.permute(0, 1, 3, 2).contiguous()
        key_states = key_states.view(batch_size * num_heads, head_dim, seq_length)
        value_states = value_states.contiguous().view(batch_size * num_heads, seq_length, head_dim)
        return key_states, value_states
