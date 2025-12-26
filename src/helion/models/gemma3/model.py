from __future__ import annotations

from typing import Optional, Union

import hivemind
import torch
import torch.nn as nn
from hivemind.utils.logging import get_logger
from transformers.cache_utils import Cache
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3CausalLMOutputWithPast,
    Gemma3Model,
    Gemma3ModelOutputWithPast,
    Gemma3PreTrainedModel,
    Gemma3ForConditionalGeneration,
)

from helion.client.from_pretrained import FromPretrainedMixin
from helion.client.lm_head import LMHead
from helion.client.ptune import PTuneMixin
from helion.client.remote_generation import RemoteGenerationMixin, RemotePastKeyValues
from helion.client.remote_sequential import RemoteSequential
from helion.models.gemma3.config import DistributedGemma3Config

logger = get_logger(__name__)


class DistributedGemma3Model(FromPretrainedMixin, PTuneMixin, Gemma3Model):
    """
    Gemma3Model, but all decoder layers are hosted by the swarm.
    Vision tower + multimodal projector remain local.
    """

    _keys_to_ignore_on_load_missing = PTuneMixin._keys_to_ignore_on_load_missing
    # Ignore decoder layer weights: they are hosted by the swarm and should NOT be loaded locally.
    # Gemma3 checkpoints may store them under either:
    # - model.language_model.layers.*
    # - model.language_model.model.layers.*
    _keys_to_ignore_on_load_unexpected = [
        r"^model\.language_model\.layers\.",
        r"^model\.language_model\.model\.layers\.",
    ]

    config_class = DistributedGemma3Config

    def __init__(self, config: DistributedGemma3Config, *, dht: Optional[hivemind.DHT] = None):
        # Prevent initialization of local decoder layers by zeroing text_config.num_hidden_layers
        text_cfg = config.get_text_config()
        n_layer, text_cfg.num_hidden_layers = int(text_cfg.num_hidden_layers), 0
        super().__init__(config)
        text_cfg.num_hidden_layers = n_layer

        # Swap in RemoteSequential for the decoder stack
        self.language_model.layers = RemoteSequential(config, dht=dht)

        self.requires_grad_(False)  # keep local pieces frozen in distributed inference
        self.init_prompts(config)

    @property
    def h(self) -> RemoteSequential:
        return self.language_model.layers

    @property
    def ln_f(self) -> nn.Module:
        return self.language_model.norm

    @property
    def word_embeddings(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()

    @property
    def word_embeddings_layernorm(self) -> nn.Module:
        return nn.Identity()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[list[torch.FloatTensor], Cache]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **lm_kwargs,
    ):
        # Mirrors upstream Gemma3Model.forward, but routes decoder layers through RemoteSequential.
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if output_attentions:
            raise AssertionError("output_attentions is not supported in distributed Gemma3")
        if output_hidden_states:
            raise AssertionError("output_hidden_states is not supported in distributed Gemma3")
        if return_dict is not None and not return_dict:
            raise AssertionError("return_dict=False is not supported in distributed Gemma3")

        # Replace image token with PAD if it is OOV
        llm_input_ids = input_ids
        if input_ids is not None and getattr(self.config, "image_token_id", -1) >= self.config.text_config.vocab_size:
            special_image_mask = input_ids == self.config.image_token_id
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # Merge text and images (local vision + projector)
        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # Prompt tuning (optional)
        use_prompts = (
            self.config.tuning_mode
            and "ptune" in self.config.tuning_mode
            and self.h.position == 0
        )
        if use_prompts:
            batch_size = inputs_embeds.shape[0]
            prompts, intermediate_prompts = self.get_prompt(batch_size)
            inputs_embeds = torch.cat([prompts, inputs_embeds], dim=1)
        else:
            intermediate_prompts = None

        hidden_states = inputs_embeds

        # transformers generation may pass its own cache objects (e.g. DynamicCache/Cache) here.
        # Convert them into RemotePastKeyValues, which is what RemoteSequential expects.
        if past_key_values is None:
            past_key_values = RemotePastKeyValues()
        elif not isinstance(past_key_values, RemotePastKeyValues):
            converted = RemotePastKeyValues()
            # Best-effort: preserve "seen tokens" so cache_position stays consistent.
            try:
                converted.update_seen(int(past_key_values.get_seq_length()))  # type: ignore[attr-defined]
            except Exception:
                pass
            # Best-effort: preserve beam search bookkeeping if present.
            try:
                converted.hypo_ids = getattr(past_key_values, "hypo_ids", None)
            except Exception:
                pass
            past_key_values = converted
        assert isinstance(past_key_values, RemotePastKeyValues)

        # If no active inference session is present, we cannot pass extra kwargs to RemoteSequential.
        tt_ids = token_type_ids if self.h.active_session is not None else None

        hidden_states = self.h(
            hidden_states,
            prompts=intermediate_prompts,
            hypo_ids=past_key_values.hypo_ids if past_key_values is not None else None,
            token_type_ids=tt_ids,
        )
        past_key_values.update_seen(hidden_states.size(1))

        # Remove prompt prefix
        if use_prompts:
            hidden_states = hidden_states[:, self.pre_seq_len :]

        hidden_states = self.language_model.norm(hidden_states)
        return Gemma3ModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
            image_hidden_states=image_features if pixel_values is not None else None,
        )


class DistributedGemma3ForConditionalGeneration(FromPretrainedMixin, RemoteGenerationMixin, Gemma3ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = DistributedGemma3Model._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = DistributedGemma3Model._keys_to_ignore_on_load_unexpected

    config_class = DistributedGemma3Config

    def __init__(self, config: DistributedGemma3Config):
        Gemma3PreTrainedModel.__init__(self, config)
        self.model = DistributedGemma3Model(config)
        self.lm_head = LMHead(config)
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    @property
    def transformer(self) -> DistributedGemma3Model:
        return self.model


