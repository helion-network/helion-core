from helion.models.gemma3.block import WrappedGemma3Block
from helion.models.gemma3.config import DistributedGemma3Config
from helion.models.gemma3.model import (
    DistributedGemma3ForConditionalGeneration,
    DistributedGemma3Model,
)
from helion.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedGemma3Config,
    model=DistributedGemma3Model,
    model_for_conditional_generation=DistributedGemma3ForConditionalGeneration,
)


