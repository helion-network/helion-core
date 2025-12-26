import ast
import contextlib
import dataclasses
from contextvars import ContextVar
from typing import Any, ContextManager, Dict, List, Optional, Tuple

import torch
import transformers
from hivemind.utils.logging import get_logger
from torch import Tensor
from transformers.generation.utils import ModelOutput

from helion.client.inference_session import InferenceSession
from helion.client.remote_sequential import RemoteSequential
from helion.utils.misc import DUMMY, docstring_from

logger = get_logger(__name__)


@dataclasses.dataclass
class RemotePastKeyValues:
    """
    A minimal, version-agnostic cache object for Transformers generation.

    We intentionally do NOT inherit from `transformers.cache_utils.Cache` because its constructor and
    invariants vary across Transformers versions (and may require per-layer cache classes).

    This object only tracks the number of seen tokens and stores `hypo_ids` for beam-search plumbing.
    """

    _seen_tokens: int = 0
    hypo_ids: Optional[torch.LongTensor] = None

    def __getitem__(self, _index: int) -> List[torch.Tensor]:
        return [DUMMY]  # For compatibility with BloomForCausalLM.prepare_inputs_for_generation()

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return int(self._seen_tokens)

    def get_max_length(self) -> Optional[int]:
        return None

    def update_seen(self, new_seen: int) -> None:
        self._seen_tokens += int(new_seen)

    def reorder_cache(self, beam_idx):
        # Some Transformers versions call `past_key_values.reorder_cache(...)` instead of model._reorder_cache.
        self.hypo_ids = beam_idx
        return self


_skipped_tokens = ContextVar("skipped_tokens", default=0)


class _SkipTokensMixin:
    # This override is used in RemoteGenerationMixin by has to be defined in a class not named as "GenerationMixin"
    # due to how transformers.PreTrainedModel.can_generate() works
    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs) -> dict:
        input_ids = input_ids[:, _skipped_tokens.get() :]
        _skipped_tokens.set(0)
        return super().prepare_inputs_for_generation(input_ids, **kwargs)


class RemoteGenerationMixin(_SkipTokensMixin):
    """
    This class is an upgrade to `transformers.GenerationMixin` that:

    - Designed to be compatible with most `transformers.GenerationMixin` strategies and options
    - Supports generation inside a remote InferenceSession, so that remote servers store your attention caches and
      you don't have to rerun the prefix through all the servers to generate each new token
    - Supports multiple `.generate()` calls inside one InferenceSession, so you can easily run interactive generation
      by showing tokens on the fly (multiple calls like `.generate(None, max_new_tokens=1, ...)`) or
      accept prompts from a user in a chat bot (multiple calls like `.generate(new_prompts, ...)`).
    - If there is no active session, `.generate()` will create a new InferenceSession with proper `max_length`.
      Otherwise, `.generate()` will use the active session. You can use the `session=...` argument to override that.
    """

    @docstring_from(RemoteSequential.active_session)
    @property
    def active_session(self) -> Optional[InferenceSession]:
        return self.transformer.h.active_session

    @docstring_from(RemoteSequential.use_session)
    def use_session(self, session: Optional[InferenceSession]) -> ContextManager[InferenceSession]:
        return self.transformer.h.use_session(session)

    @docstring_from(RemoteSequential.inference_session)
    def inference_session(self, **kwargs) -> ContextManager[InferenceSession]:
        return self.transformer.h.inference_session(**kwargs)

    @docstring_from(transformers.GenerationMixin.generate.__doc__)
    def generate(
        self, inputs: Optional[torch.Tensor] = None, *args, session: Optional[InferenceSession] = None, **kwargs
    ):
        self._fix_generate_kwargs(kwargs)
        if inputs is None:
            inputs = kwargs.pop("input_ids", None)

        if session is not None:
            # If a session specified explicitly, use it
            context_manager = self.use_session(session)
        elif self.active_session is not None:
            # If there's an active session, don't do anything
            context_manager = contextlib.nullcontext(self.active_session)
        else:
            # If there's no active session, create a new one

            max_length = kwargs.get("max_length")
            max_new_tokens = kwargs.get("max_new_tokens")
            assert (max_length is None) != (
                max_new_tokens is None
            ), "You should set `max_length` or `max_new_tokens` (but not both) to reserve server-side attention caches"

            session_max_length = self.transformer.config.pre_seq_len
            if max_length is not None:
                session_max_length += max_length
            else:
                session_max_length += (inputs.shape[1] if inputs is not None else 0) + max_new_tokens
            context_manager = self.inference_session(max_length=session_max_length)

        with context_manager as session:
            # Prepend the tokens from the previous .generate() call
            n_prev_tokens = session.output_ids.shape[1] if session.output_ids is not None else 0
            if n_prev_tokens > 0:
                if kwargs.get("num_beams", 1) > 1:
                    logger.warning(
                        "Beam search will not work properly in the resumed petals.InferenceSession "
                        "since intermediate beam entries are lost"
                    )

                if inputs is not None:
                    inputs = torch.cat([session.output_ids, inputs], dim=1)
                else:
                    inputs = session.output_ids

                # Don't actually run all previous tokens through the transformer,
                # but keep them for transformers.GenerationMixin (e.g., to compute repetition_penalty)
                _skipped_tokens.set(max(0, n_prev_tokens - 1))

            # transformers.GenerationMixin validates model_kwargs against the model's forward signature.
            # Some tokenizers (and gateways) pass extra keys such as token_type_ids; some models ignore attention_mask.
            # If that happens, retry generation with the unused kwargs removed.
            try:
                result = super().generate(inputs, *args, **kwargs)
            except ValueError as e:
                msg = str(e)
                needle = "The following `model_kwargs` are not used by the model:"
                if needle not in msg:
                    raise

                # Example msg:
                # "The following `model_kwargs` are not used by the model: ['attention_mask', 'token_type_ids'] (note: ...)"
                unused_part = msg.split(needle, 1)[1]
                unused_part = unused_part.split("(note:", 1)[0].strip()
                try:
                    unused_keys = ast.literal_eval(unused_part)
                except Exception:
                    raise

                if not isinstance(unused_keys, (list, tuple)) or not all(isinstance(k, str) for k in unused_keys):
                    raise

                filtered_kwargs = dict(kwargs)
                removed = []
                for key in unused_keys:
                    if key in filtered_kwargs:
                        removed.append(key)
                        filtered_kwargs.pop(key, None)

                if not removed:
                    raise

                logger.warning(f"Retrying .generate() with unsupported model_kwargs removed: {removed}")
                result = super().generate(inputs, *args, **filtered_kwargs)

            sequences = result.sequences if isinstance(result, ModelOutput) else result
            # Save tokens from this .generate() call
            session.output_ids = sequences
            # Crop the last tokens from the previous call
            sequences = sequences[:, n_prev_tokens:].clone()
            if isinstance(result, ModelOutput):
                result.sequences = sequences
            else:
                result = sequences

        return result

    @staticmethod
    def _fix_generate_kwargs(kwargs: dict):
        # Suppress inappropriate "Both max_new_tokens and max_length" HF warning
        if "max_length" in kwargs and kwargs["max_length"] is None:
            del kwargs["max_length"]

        # Support do_sample = {0, 1} for backward compatibility with Petals < 2.1.0
        do_sample = kwargs.get("do_sample")
        if isinstance(do_sample, int):
            kwargs["do_sample"] = bool(do_sample)

    @staticmethod
    def _reorder_cache(past_key_values: RemotePastKeyValues, beam_idx: torch.LongTensor) -> RemotePastKeyValues:
        return dataclasses.replace(past_key_values, hypo_ids=beam_idx)
