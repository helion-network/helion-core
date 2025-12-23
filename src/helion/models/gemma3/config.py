from __future__ import annotations

import os
from typing import Optional, Tuple, Union

from hivemind import get_logger
from transformers.models.gemma3 import Gemma3Config
from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention

from helion.client.config import ClientConfig
from helion.client.lm_head import LMHeadConfig
from helion.client.ptune import PTuneConfig
from helion.models.gemma3.block import WrappedGemma3Block

logger = get_logger(__name__)


class DistributedGemma3Config(Gemma3Config, ClientConfig, PTuneConfig, LMHeadConfig):
    """
    Distributed config for Gemma3 (including MedGemma).

    Gemma3 is a multimodal model with a vision tower and a decoder-only text model.
    Helion distributes only the *decoder layers*; vision + multimodal projector stay local.
    """

    block_class = WrappedGemma3Block
    attn_class = Gemma3Attention
    # NOTE: weight layouts differ across checkpoints; we keep a small set of stable candidates.
    # `load_pretrained_block()` will also try adding a leading "model." automatically.
    block_prefix: Tuple[str, ...] = (
        "model.language_model.layers",
        "language_model.layers",
        "model.language_model.model.layers",
        "language_model.model.layers",
    )

    @property
    def hidden_size(self) -> int:  # used by server-side cache sizing and LM head
        return int(self.text_config.hidden_size)

    @property
    def vocab_size(self) -> int:  # used by LM head
        return int(self.text_config.vocab_size)

    @property
    def num_hidden_layers(self) -> int:
        return int(self.text_config.num_hidden_layers)

    @property
    def num_attention_heads(self) -> int:
        return int(self.text_config.num_attention_heads)

    @property
    def num_key_value_heads(self) -> int:
        return int(self.text_config.num_key_value_heads)

    @property
    def num_key_value_groups(self) -> int:
        return int(self.num_attention_heads // self.num_key_value_heads)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, os.PathLike, None],
        *args,
        dht_prefix: Optional[str] = None,
        **kwargs,
    ):
        loading_from_repo = model_name_or_path is not None and not os.path.isdir(model_name_or_path)
        if loading_from_repo and dht_prefix is None:
            dht_prefix = str(model_name_or_path).split("/")[-1]
            dht_prefix = dht_prefix.replace(".", "-")
            logger.info(f"Using DHT prefix: {dht_prefix}")
        return super().from_pretrained(model_name_or_path, *args, dht_prefix=dht_prefix, **kwargs)


