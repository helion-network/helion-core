from typing import Optional

import torch
import torch.nn as nn
from hivemind import DHT
from hivemind import get_logger
from transformers.modeling_outputs import MoeModelOutputWithPast, SequenceClassifierOutputWithPast
from transformers.models.gpt_oss import GptOssForCausalLM, GptOssModel, GptOssPreTrainedModel

from helion.client.from_pretrained import FromPretrainedMixin
from helion.client.lm_head import LMHead
from helion.client.ptune import PTuneMixin
from helion.client.remote_generation import RemoteGenerationMixin, RemotePastKeyValues
from helion.client.remote_sequential import RemoteSequential
from helion.models.gpt_oss.config import DistributedGptOssConfig

logger = get_logger(__name__)


class DistributedGptOssModel(FromPretrainedMixin, PTuneMixin, GptOssModel):
    """GptOssModel, but transformer layers are hosted by the swarm"""

    # Transformers copies these from the *class* into the instance in PreTrainedModel.__init__.
    # Set to None to bypass module-name validation (our distributed model does not have local blocks).
    _keep_in_fp32_modules = None
    _keep_in_fp32_modules_strict = None

    _keys_to_ignore_on_load_missing = PTuneMixin._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = [r"^model\.layers\."]

    config_class = DistributedGptOssConfig

    def __init__(self, config: DistributedGptOssConfig, *, dht: Optional[DHT] = None):
        # Set fp32 lists to None BEFORE super().__init__() calls post_init() which validates it
        # None bypasses the validation check in transformers
        config._keep_in_fp32_modules = None
        config._keep_in_fp32_modules_strict = None
        num_layers, config.num_hidden_layers = config.num_hidden_layers, 0
        # Set again right before super call as final safety
        config._keep_in_fp32_modules = None
        config._keep_in_fp32_modules_strict = None
        super().__init__(config)
        assert len(self.layers) == 0
        config.num_hidden_layers = num_layers

        self.layers = RemoteSequential(config, dht=dht)

        self.requires_grad_(False)
        self.init_prompts(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[RemotePastKeyValues] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        assert attention_mask is None or (attention_mask == 1).all(), "Custom attention masks are not supported"
        if cache_position is not None:
            assert position_ids is not None and torch.all(torch.eq(cache_position, position_ids)).item()
        assert (
            position_ids is None or (position_ids[:, 1:] - position_ids[:, :-1] == 1).all()
        ), "Non-consecutive position_ids are not supported"
        assert head_mask is None, f"Custom head masks are not supported, {head_mask=}"
        assert use_cache is None or use_cache, f"{use_cache=} is not supported"
        assert not output_attentions, f"{output_attentions=} is not supported"
        assert not output_hidden_states, f"{output_hidden_states=} is not supported"
        assert return_dict is None or return_dict, f"{return_dict=} is not supported"
        assert not output_router_logits, f"{output_router_logits=} is not supported"

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        use_prompts = self.config.tuning_mode and "ptune" in self.config.tuning_mode and self.layers.position == 0
        if use_prompts:
            batch_size = inputs_embeds.shape[0]
            prompts, intermediate_prompts = self.get_prompt(batch_size)
            inputs_embeds = torch.cat([prompts, inputs_embeds], dim=1)
        else:
            prompts = intermediate_prompts = None

        hidden_states = inputs_embeds
        output_shape = input_shape + (hidden_states.size(-1),)

        if past_key_values is None:
            past_key_values = RemotePastKeyValues()
        past_key_values.update_seen(hidden_states.size(1))

        hidden_states = self.layers(
            hidden_states,
            prompts=intermediate_prompts,
            hypo_ids=past_key_values.hypo_ids if past_key_values is not None else None,
        )

        if use_prompts:
            hidden_states = hidden_states[:, self.pre_seq_len :]

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )

    @property
    def word_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    @property
    def word_embeddings_layernorm(self) -> nn.Module:
        return nn.Identity()

    @property
    def h(self) -> RemoteSequential:
        return self.layers

    @property
    def ln_f(self) -> nn.Module:
        return self.norm


class DistributedGptOssForCausalLM(FromPretrainedMixin, RemoteGenerationMixin, GptOssForCausalLM):
    # See note in DistributedGptOssModel: must be overridden at class-level.
    _keep_in_fp32_modules = None
    _keep_in_fp32_modules_strict = None

    _keys_to_ignore_on_load_missing = DistributedGptOssModel._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = DistributedGptOssModel._keys_to_ignore_on_load_unexpected

    config_class = DistributedGptOssConfig

    def __init__(self, config: DistributedGptOssConfig):
        # Safety: set fp32 lists to None before any parent init or model creation
        # None bypasses the validation check in transformers
        config._keep_in_fp32_modules = None
        config._keep_in_fp32_modules_strict = None

        GptOssPreTrainedModel.__init__(self, config)
        # Set again before creating DistributedGptOssModel
        config._keep_in_fp32_modules = None
        config._keep_in_fp32_modules_strict = None
        self.model = DistributedGptOssModel(config)
        self.lm_head = LMHead(config)
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    @property
    def transformer(self) -> DistributedGptOssModel:
        return self.model


class DistributedGptOssForSequenceClassification(FromPretrainedMixin, GptOssPreTrainedModel):
    # See note in DistributedGptOssModel: must be overridden at class-level.
    _keep_in_fp32_modules = None
    _keep_in_fp32_modules_strict = None

    _keys_to_ignore_on_load_missing = DistributedGptOssModel._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = DistributedGptOssModel._keys_to_ignore_on_load_unexpected

    config_class = DistributedGptOssConfig

    def __init__(self, config: DistributedGptOssConfig):
        # Set fp32 lists to None before creating DistributedGptOssModel
        # None bypasses the validation check in transformers
        config._keep_in_fp32_modules = None
        config._keep_in_fp32_modules_strict = None
        GptOssPreTrainedModel.__init__(self, config)
        self.num_labels = config.num_labels

        self.model = DistributedGptOssModel(config)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[RemotePastKeyValues] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        transformer_outputs: MoeModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = transformer_outputs.last_hidden_state
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} may produce unexpected results when padding tokens are used with `inputs_embeds`."
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                pooled_logits=pooled_logits,
                config=self.config,
            )

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

