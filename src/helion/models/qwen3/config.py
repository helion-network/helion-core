import os
from typing import Optional, Union

from hivemind import get_logger

try:
    from transformers.models.qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
except ImportError:
    # Fallback: Qwen3 might be under qwen2 or use a different path
    try:
        from transformers.models.qwen2 import Qwen2Config as Qwen3Config
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention as Qwen3Attention
    except ImportError:
        # If neither works, we'll need to check the actual transformers version
        raise ImportError(
            "Qwen3 model classes not found in transformers. "
            "Please ensure you have a transformers version that supports Qwen3."
        )

from helion.client.config import ClientConfig
from helion.client.lm_head import LMHeadConfig
from helion.client.ptune import PTuneConfig
from helion.models.qwen3.block import WrappedQwen3Block

logger = get_logger(__name__)


class DistributedQwen3Config(Qwen3Config, ClientConfig, PTuneConfig, LMHeadConfig):
    block_class = WrappedQwen3Block
    attn_class = Qwen3Attention
    block_prefix = "model.layers"

    @property
    def num_key_value_groups(self):
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: Union[str, os.PathLike, None], *args, dht_prefix: Optional[str] = None, **kwargs
    ):
        loading_from_repo = model_name_or_path is not None and not os.path.isdir(model_name_or_path)
        if loading_from_repo and dht_prefix is None:
            dht_prefix = str(model_name_or_path)
            dht_prefix = dht_prefix.split("/")[-1]  # Use only repo name to merge blocks hosted by different accounts
            dht_prefix = dht_prefix.replace(".", "-")
            if not dht_prefix.endswith("-hf"):
                dht_prefix += "-hf"
            logger.info(f"Using DHT prefix: {dht_prefix}")

        result = super().from_pretrained(model_name_or_path, *args, dht_prefix=dht_prefix, **kwargs)
        config = result[0] if isinstance(result, tuple) else result
        config.pretraining_tp = 1  # This may give less accurate results but it doesn't matter if we use quantization
        config.use_cache = True  # use_cache=False leads to identical results but is slower and not supported by Petals
        return result

