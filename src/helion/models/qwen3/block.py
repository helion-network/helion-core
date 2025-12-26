"""
Qwen3 intermediate layer
Based on Llama implementation pattern, adapted for Qwen3 architecture.
Qwen3 uses a Llama-like architecture with similar structure.
"""
import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

try:
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3Attention,
        Qwen3Config,
        Qwen3DecoderLayer,
        Qwen3MLP,
        Qwen3RMSNorm,
        Qwen3RotaryEmbedding,
        repeat_kv,
        rotate_half,
    )
except ImportError:
    # Fallback: Qwen3 might use Qwen2 classes
    try:
        from transformers.models.qwen2.modeling_qwen2 import (
            Qwen2Attention as Qwen3Attention,
            Qwen2Config as Qwen3Config,
            Qwen2DecoderLayer as Qwen3DecoderLayer,
            Qwen2MLP as Qwen3MLP,
            Qwen2RMSNorm as Qwen3RMSNorm,
            Qwen2RotaryEmbedding as Qwen3RotaryEmbedding,
            repeat_kv,
            rotate_half,
        )
    except ImportError:
        raise ImportError(
            "Qwen3 model classes not found in transformers. "
            "Please ensure you have a transformers version that supports Qwen3."
        )

from helion.utils.cuda_graphs import make_inference_graphed_callable


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class OptimizedQwen3Attention(Qwen3Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rotary_graph = None
        if not hasattr(self, "rotary_emb") or self.rotary_emb is None:
            # HF>=4.45 expects position embeddings to be passed externally; keep rotary_emb for legacy callers
            # Create without device to avoid meta tensor issues during initialization; will be moved with module
            device = self.q_proj.weight.device if self.q_proj.weight.device.type != "meta" else None
            self.rotary_emb = Qwen3RotaryEmbedding(self.config, device=device)
        if not hasattr(self, "num_heads"):
            self.num_heads = self.config.num_attention_heads
        if not hasattr(self, "num_key_value_heads"):
            self.num_key_value_heads = getattr(
                self.config, "num_key_value_heads", self.config.num_attention_heads // self.config.num_key_value_groups
            )
        if not hasattr(self, "hidden_size"):
            self.hidden_size = self.config.hidden_size

    def _optimized_apply_rotary(self, query_states, key_states, cos, sin):
        if self._rotary_graph is None:
            self._rotary_graph = make_inference_graphed_callable(
                apply_rotary_pos_emb, sample_args=(query_states, key_states, cos, sin)
            )
        return self._rotary_graph(query_states, key_states, cos, sin)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        assert not output_attentions
        if position_ids is None:
            past_seen_tokens = past_key_value[0].shape[2] if past_key_value is not None else 0
            position_ids = torch.arange(
                past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
            ).unsqueeze(0)

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        # Determine head_dim to use - check cache if available
        head_dim_to_use = self.head_dim
        if past_key_value is not None:
            past_key_head_dim = past_key_value[0].shape[-1]
            if past_key_head_dim != self.head_dim:
                # Cache has different head_dim (e.g., from tensor parallelism)
                # Use the cache's head_dim to ensure compatibility
                head_dim_to_use = past_key_head_dim
                warnings.warn(
                    f"Using cache's head_dim={past_key_head_dim} instead of model's head_dim={self.head_dim} "
                    f"to ensure compatibility. This may be due to tensor parallelism."
                )
        
        # Use head_dim_to_use for all states to ensure consistency
        # If cache has different head_dim, we need to use it for query, key, and value
        q_output_size = query_states.shape[-1]
        kv_output_size = key_states.shape[-1]
        
        if head_dim_to_use != self.head_dim:
            # Recalculate num_heads and num_key_value_heads based on actual output and target head_dim
            # This handles tensor parallelism where head_dim is split
            actual_num_heads = q_output_size // head_dim_to_use
            actual_num_kv_heads = kv_output_size // head_dim_to_use
            if actual_num_heads * head_dim_to_use != q_output_size:
                raise ValueError(
                    f"Cannot reshape query states: output_size={q_output_size} is not divisible by "
                    f"head_dim_to_use={head_dim_to_use}. Cache head_dim={past_key_head_dim}, "
                    f"model head_dim={self.head_dim}"
                )
            if actual_num_kv_heads * head_dim_to_use != kv_output_size:
                raise ValueError(
                    f"Cannot reshape key/value states: output_size={kv_output_size} is not divisible by "
                    f"head_dim_to_use={head_dim_to_use}. Cache head_dim={past_key_head_dim}, "
                    f"model head_dim={self.head_dim}"
                )
            query_states = query_states.view(bsz, q_len, actual_num_heads, head_dim_to_use).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, actual_num_kv_heads, head_dim_to_use).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, actual_num_kv_heads, head_dim_to_use).transpose(1, 2)
        else:
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # HF>=4.45: Use externally provided position embeddings if available, otherwise compute internally
        if position_embeddings is not None:
            cos, sin = position_embeddings
            cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        else:
            # Fallback for legacy callers
            cos, sin = self.rotary_emb(value_states, position_ids)
            cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)

        if q_len == 1 and torch.is_inference_mode_enabled() and hidden_states.device.type == "cuda":
            query_states, key_states = self._optimized_apply_rotary(query_states, key_states, cos, sin)
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Use head_dim_to_use for scaling to match the actual head_dim being used
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim_to_use)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        # NOTE: under tensor-parallel sharding, per-shard `hidden_size` can differ from the
        # actual attention output width (num_heads * head_dim). Reshape dynamically.
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class OptimizedQwen3DecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config: Qwen3Config):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = OptimizedQwen3Attention(config=config, layer_idx=0)
        # layer_idx only matters for KV caching, and we re-implement it in Petals
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.pre_attn_graph = None
        self.post_attn_graph = None

    def _optimized_input_layernorm(self, hidden_states):
        if self.pre_attn_graph is None:
            self.pre_attn_graph = make_inference_graphed_callable(
                self.input_layernorm.forward, sample_args=(hidden_states,)
            )
        return self.pre_attn_graph(hidden_states)

    def _optimized_output_layernorm(self, hidden_states):
        if self.post_attn_graph is None:
            self.post_attn_graph = make_inference_graphed_callable(
                self.post_attention_layernorm.forward, sample_args=(hidden_states,)
            )
        return self.post_attn_graph(hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            position_embeddings (`Tuple(torch.Tensor, torch.Tensor)`, *optional*): externally computed position embeddings (cos, sin)
        """

        residual = hidden_states

        if hidden_states.size(1) == 1 and torch.is_inference_mode_enabled() and hidden_states.device.type == "cuda":
            hidden_states = self._optimized_input_layernorm(hidden_states)
        else:
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states

        if hidden_states.size(1) == 1 and torch.is_inference_mode_enabled() and hidden_states.device.type == "cuda":
            hidden_states = self._optimized_output_layernorm(hidden_states)
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class WrappedQwen3Block(OptimizedQwen3DecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        batch_size, seq_length, _ = hidden_states.shape
        if not hasattr(self, "config"):
            self.config = self.self_attn.config

        seq_length_with_past = seq_length
        past_key_values_length = 0

        past_key_value = layer_past
        if past_key_value is not None:
            # Infer sequence length from value tensor which is [B*Hkv, T, D] in Bloom format
            # This is more reliable than using key tensor shape[2]
            past_key_values_length = past_key_value[1].shape[1]
            seq_length_with_past = seq_length_with_past + past_key_values_length
            past_key_value = self._reorder_cache_from_bloom_to_qwen3(past_key_value, batch_size, past_key_values_length)

        assert position_ids is None

        # Compute position_ids for position embeddings
        position_ids = torch.arange(
            past_key_values_length, past_key_values_length + seq_length, device=hidden_states.device
        ).unsqueeze(0)

        # HF>=4.45: Compute position embeddings externally
        # Create a dummy tensor with the shape that rotary_emb expects (batch_size, num_key_value_heads, seq_length, head_dim)
        # The rotary_emb only uses the shape, not the actual values
        num_kv_heads = self._get_num_kv_heads()
        head_dim = getattr(self.self_attn, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        dummy_value_states = torch.zeros(
            (batch_size, num_kv_heads, seq_length, head_dim),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        position_embeddings = self.self_attn.rotary_emb(dummy_value_states, position_ids)

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
            )
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask=attention_mask,
            input_shape=(batch_size, seq_length),
            inputs_embeds=hidden_states,
            past_key_values_length=past_key_values_length,
        )

        outputs = super().forward(
            hidden_states,
            *args,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        if use_cache:
            present_key_value = outputs[-1]
            present_key_value = self._reorder_cache_from_qwen3_to_bloom(
                present_key_value, batch_size, seq_length_with_past
            )
            outputs = outputs[:-1] + (present_key_value,)

        return outputs

    def _reorder_cache_from_bloom_to_qwen3(
        self, key_value: Tuple[torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor]:
        key_states, value_states = key_value
        # Bloom format: key is [B*Hkv, D, T], value is [B*Hkv, T, D]
        num_kv_heads = self._get_num_kv_heads()
        head_dim = self.self_attn.head_dim
        
        # Use value_states to infer dimensions since it's already in [B*Hkv, T, D] format
        # This is more reliable than using key_states which needs permute
        batch_kv_heads = value_states.shape[0]  # B*Hkv
        actual_seq_length = value_states.shape[1]  # T (sequence length)
        actual_head_dim = value_states.shape[2]  # D
        
        # Use actual head_dim from tensor instead of expected head_dim
        # This handles cases with tensor parallelism or different cache formats
        # If there's a mismatch, log a warning but use the tensor's actual dimension
        if actual_head_dim != head_dim:
            warnings.warn(
                f"Head dimension mismatch: expected {head_dim}, using actual {actual_head_dim} "
                f"from value_states shape {value_states.shape}. This may be due to tensor parallelism."
            )
        head_dim_to_use = actual_head_dim
        
        # Infer batch_size from tensor shape
        # batch_kv_heads should equal batch_size * num_kv_heads
        if batch_kv_heads % num_kv_heads != 0:
            raise ValueError(
                f"Cannot infer batch_size: batch_kv_heads={batch_kv_heads} is not divisible by "
                f"num_kv_heads={num_kv_heads}. Value states shape: {value_states.shape}"
            )
        inferred_batch_size = batch_kv_heads // num_kv_heads
        
        # Permute key_states from [B*Hkv, D, T] to [B*Hkv, T, D]
        key_states = key_states.permute(0, 2, 1)
        
        # Verify key_states matches value_states shape after permute
        if key_states.shape != value_states.shape:
            raise ValueError(
                f"Key and value states shape mismatch after permute: "
                f"key_states.shape={key_states.shape}, value_states.shape={value_states.shape}"
            )
        
        # Reshape using inferred dimensions (use actual head_dim from tensor)
        key_states = key_states.view(inferred_batch_size, num_kv_heads, actual_seq_length, head_dim_to_use)
        value_states = value_states.view(inferred_batch_size, num_kv_heads, actual_seq_length, head_dim_to_use)
        return (key_states, value_states)

    def _reorder_cache_from_qwen3_to_bloom(
        self, key_value: Tuple[torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor]:
        key_states, value_states = key_value
        num_kv_heads = self._get_num_kv_heads()
        # Infer actual sequence length from tensor shape to avoid shape mismatches
        # key_states and value_states are [B, Hkv, T, D], so shape[2] is the sequence length
        actual_seq_length = key_states.shape[2]
        value_states = value_states.view(batch_size * num_kv_heads, actual_seq_length, self.self_attn.head_dim)
        key_states = key_states.view(*value_states.shape)
        key_states = key_states.permute(0, 2, 1)
        return (key_states, value_states)

    def _get_num_kv_heads(self) -> int:
        num_kv_heads = getattr(self.self_attn, "num_key_value_heads", None)
        if num_kv_heads is not None:
            return num_kv_heads
        num_groups = getattr(self.self_attn, "num_key_value_groups", 1)
        num_heads = getattr(
            self.self_attn,
            "num_attention_heads",
            getattr(self.self_attn, "num_heads", getattr(self.config, "num_attention_heads", None)),
        )
        if num_heads is None:
            raise AttributeError("Unable to resolve number of attention heads for Qwen3 attention")
        return max(1, num_heads // max(1, num_groups))

