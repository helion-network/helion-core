from typing import Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssDecoderLayer, GptOssRotaryEmbedding

from helion.utils.misc import is_dummy


class WrappedGptOssBlock(GptOssDecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)

        self.layer_idx = layer_idx
        self.config = config
        self._attn_implementation = config._attn_implementation
        self.sliding_window = config.sliding_window if self.attention_type == "sliding_attention" else None
        self.rotary_emb = GptOssRotaryEmbedding(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        batch_size, seq_length, _ = hidden_states.shape

        if layer_past is not None and is_dummy(layer_past[0]):
            layer_past = None

        cache = None
        past_key_values_length = 0
        if layer_past is not None:
            past_key_values_length = layer_past[0].shape[2]
            cache = DynamicCache()
            key_states, value_states = self._reorder_cache_from_bloom(layer_past, batch_size, past_key_values_length)
            cache.update(key_states, value_states, self.layer_idx)
        if cache is None and use_cache:
            cache = DynamicCache()

        seq_length_with_past = seq_length + past_key_values_length
        cache_position = torch.arange(
            past_key_values_length, past_key_values_length + seq_length, device=hidden_states.device
        )

        if self._attn_implementation == "flash_attention_2":
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa":
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                hidden_states,
                past_key_values_length,
            )
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                hidden_states,
                past_key_values_length,
                sliding_window=self.sliding_window,
            )

        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length,
                past_key_values_length + seq_length,
                dtype=torch.long,
                device=hidden_states.device,
            ).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        hidden_states = super().forward(
            hidden_states,
            *args,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=cache,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        if use_cache:
            present_cache = cache.layers[self.layer_idx]
            present_key_value = self._reorder_cache_to_bloom(
                (present_cache.keys, present_cache.values), batch_size, seq_length_with_past
            )
            return hidden_states, present_key_value

        return (hidden_states,)

    def _reorder_cache_from_bloom(
        self, key_value: Tuple[torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor]:
        key_states, value_states = key_value
        num_kv_heads = self._get_num_kv_heads()
        key_states = key_states.permute(0, 2, 1)
        key_states = key_states.view(batch_size, num_kv_heads, seq_length, self.self_attn.head_dim)
        value_states = value_states.view(*key_states.shape)
        return (key_states, value_states)

    def _reorder_cache_to_bloom(
        self, key_value: Tuple[torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor]:
        key_states, value_states = key_value
        num_kv_heads = self._get_num_kv_heads()
        value_states = value_states.view(batch_size * num_kv_heads, seq_length, self.self_attn.head_dim)
        key_states = key_states.view(*value_states.shape)
        key_states = key_states.permute(0, 2, 1)
        return (key_states, value_states)

    def _get_num_kv_heads(self) -> int:
        num_kv_heads = getattr(self.self_attn, "num_key_value_heads", None)
        if num_kv_heads is not None:
            return num_kv_heads
        num_groups = getattr(self.self_attn, "num_key_value_groups", 1)
        num_heads = getattr(self.self_attn, "num_attention_heads", self.config.num_attention_heads)
        return max(1, num_heads // max(1, num_groups))

