import types

import torch

from helion.models.qwen3.block import WrappedQwen3Block


def _make_dummy_block(*, num_kv_heads: int, head_dim: int) -> WrappedQwen3Block:
    """
    Create a WrappedQwen3Block instance without running HF init, just enough to test
    Bloom <-> Qwen3 cache shape adapters.
    """
    block = WrappedQwen3Block.__new__(WrappedQwen3Block)
    block.self_attn = types.SimpleNamespace(
        head_dim=head_dim,
        num_key_value_heads=num_kv_heads,
        num_key_value_groups=1,
        num_heads=num_kv_heads,
    )
    block.config = types.SimpleNamespace(num_attention_heads=num_kv_heads)
    return block


def test_qwen3_cache_reorder_roundtrip():
    batch_size = 2
    num_kv_heads = 4
    head_dim = 64
    seq_len = 17

    # Helion Bloom cache shapes:
    #  - key:   [B*Hkv, D, T]
    #  - value: [B*Hkv, T, D]
    key_bloom = torch.randn(batch_size * num_kv_heads, head_dim, seq_len)
    value_bloom = torch.randn(batch_size * num_kv_heads, seq_len, head_dim)

    block = _make_dummy_block(num_kv_heads=num_kv_heads, head_dim=head_dim)

    # Bloom -> Qwen3: [B, Hkv, T, D]
    key_q, value_q = block._reorder_cache_from_bloom_to_qwen3((key_bloom, value_bloom), batch_size, seq_len)
    assert key_q.shape == (batch_size, num_kv_heads, seq_len, head_dim)
    assert value_q.shape == (batch_size, num_kv_heads, seq_len, head_dim)

    # Qwen3 -> Bloom (roundtrip)
    key_b2, value_b2 = block._reorder_cache_from_qwen3_to_bloom((key_q, value_q), batch_size, seq_len)
    assert key_b2.shape == key_bloom.shape
    assert value_b2.shape == value_bloom.shape

    torch.testing.assert_close(key_b2, key_bloom)
    torch.testing.assert_close(value_b2, value_bloom)


