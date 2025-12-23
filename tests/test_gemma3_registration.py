def test_gemma3_is_registered():
    import helion  # noqa: F401

    from helion.utils.auto_config import _CLASS_MAPPING, AutoDistributedModelForConditionalGeneration

    assert "gemma3" in _CLASS_MAPPING
    assert _CLASS_MAPPING["gemma3"].model_for_conditional_generation is not None
    assert AutoDistributedModelForConditionalGeneration is not None


