from helion.models.gpt_oss.block import WrappedGptOssBlock
from helion.models.gpt_oss.config import DistributedGptOssConfig
from helion.models.gpt_oss.model import (
    DistributedGptOssForCausalLM,
    DistributedGptOssForSequenceClassification,
    DistributedGptOssModel,
)
from helion.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedGptOssConfig,
    model=DistributedGptOssModel,
    model_for_causal_lm=DistributedGptOssForCausalLM,
    model_for_sequence_classification=DistributedGptOssForSequenceClassification,
)

