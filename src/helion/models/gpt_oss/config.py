from typing import Optional, Union

from hivemind import get_logger
from transformers.models.gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssAttention

from helion.client.config import ClientConfig
from helion.client.lm_head import LMHeadConfig
from helion.client.ptune import PTuneConfig
from helion.models.gpt_oss.block import WrappedGptOssBlock

logger = get_logger(__name__)


class DistributedGptOssConfig(GptOssConfig, ClientConfig, PTuneConfig, LMHeadConfig):
    block_class = WrappedGptOssBlock
    attn_class = GptOssAttention
    block_prefix = "layers"

    @property
    def num_key_value_groups(self):
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: Union[str, None], *args, dht_prefix: Optional[str] = None, **kwargs
    ):
        loading_from_repo = model_name_or_path is not None
        if loading_from_repo and dht_prefix is None:
            dht_prefix = str(model_name_or_path).split("/")[-1]
            dht_prefix = dht_prefix.replace(".", "-")
            logger.info(f"Using DHT prefix: {dht_prefix}")

        result = super().from_pretrained(model_name_or_path, *args, dht_prefix=dht_prefix, **kwargs)
        config = result[0] if isinstance(result, tuple) else result

        if getattr(config, "pad_token_id", None) is None:
            config.pad_token_id = 0

        # Ensure problematic fp32 list is cleared even if loaded from JSON
        # Set to None (not empty tuple) to bypass transformers' validation check
        try:
            config._keep_in_fp32_modules = None
            config._keep_in_fp32_modules_strict = None
        except Exception:
            pass

        return result

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # GPT-OSS has no post_attention_layernorm in the distributed stack
        # Set to None (not empty tuple) to bypass transformers' validation check
        self._keep_in_fp32_modules = None
        self._keep_in_fp32_modules_strict = None

