# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Model adapters for executing a contiguous decoder layer range."""

from __future__ import annotations

import inspect
import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.sequence import IntermediateTensors
from vllm.v1.core.layered_prefill import LayeredForwardOutput

logger = init_logger(__name__)

LayeredPrefillFrontier = tuple[torch.Tensor, torch.Tensor | None]


class LayeredPrefillModelAdapter(ABC):
    """Execute layer groups without adding a second forward to model classes.

    Decoder-only vLLM models expose the same structural building blocks even
    though their top-level ``forward`` implementations are model-specific. The
    adapter owns the range loop and leaves exceptional state transitions to
    small subclasses.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.backbone = self._find_backbone(model)
        self.layers = self.backbone.layers
        self.start_layer = int(self.backbone.start_layer)
        self.end_layer = int(self.backbone.end_layer)
        self.num_hidden_layers = int(self.backbone.config.num_hidden_layers)
        self._validate_structure()

    @staticmethod
    def _find_backbone(model: nn.Module) -> nn.Module:
        candidates = (getattr(model, "model", None), model)
        required = (
            "config",
            "layers",
            "start_layer",
            "end_layer",
            "embed_input_ids",
            "norm",
        )
        for candidate in candidates:
            if candidate is not None and all(
                hasattr(candidate, attribute) for attribute in required
            ):
                return candidate
        raise TypeError(
            f"Model {type(model).__name__} does not expose a decoder backbone "
            "with layers, layer bounds, embedding, and final norm"
        )

    def _validate_structure(self) -> None:
        if not 0 <= self.start_layer < self.end_layer <= self.num_hidden_layers:
            raise TypeError(
                f"Model {type(self.model).__name__} has invalid decoder layer "
                f"bounds [{self.start_layer}, {self.end_layer}) for "
                f"{self.num_hidden_layers} layers"
            )
        if len(self.layers) < self.end_layer:
            raise TypeError(
                f"Model {type(self.model).__name__} does not use globally "
                "indexed decoder layers"
            )

    def _validate_range(self, layer_start: int, layer_end: int) -> None:
        if not 0 <= layer_start < layer_end <= self.num_hidden_layers:
            raise ValueError(
                f"invalid layered layer range [{layer_start}, {layer_end})"
            )

    def _prepare_initial_state(
        self,
        *,
        input_ids: torch.Tensor | None,
        frontier: LayeredPrefillFrontier | None,
        inputs_embeds: torch.Tensor | None,
        intermediate_tensors: IntermediateTensors | None,
    ) -> LayeredPrefillFrontier:
        if frontier is not None:
            if inputs_embeds is not None or intermediate_tensors is not None:
                raise ValueError(
                    "frontier and initial activation inputs are mutually exclusive"
                )
            return frontier
        if intermediate_tensors is not None:
            if inputs_embeds is not None:
                raise ValueError(
                    "PP intermediate tensors and input embeddings are mutually "
                    "exclusive"
                )
            return (
                intermediate_tensors["hidden_states"],
                intermediate_tensors.tensors.get("residual"),
            )
        if inputs_embeds is not None:
            return inputs_embeds, None
        if input_ids is None:
            raise ValueError(
                "the initial layered group requires input_ids or inputs_embeds"
            )
        return self.backbone.embed_input_ids(input_ids), None

    @abstractmethod
    def _forward_layer(
        self,
        layer: nn.Module,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        input_ids: torch.Tensor | None,
    ) -> LayeredPrefillFrontier:
        """Execute one decoder layer."""

    @abstractmethod
    def _finalize(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> LayeredPrefillFrontier:
        """Apply the model's final hidden-state transformation."""

    def forward(
        self,
        *,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        layer_start: int,
        layer_end: int,
        frontier: LayeredPrefillFrontier | None = None,
        inputs_embeds: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> LayeredForwardOutput:
        """Execute the local portion of a contiguous global layer group."""

        self._validate_range(layer_start, layer_end)
        hidden_states, residual = self._prepare_initial_state(
            input_ids=input_ids,
            frontier=frontier,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors,
        )

        trace_layered = os.environ.get("VLLM_LAYERED_PREFILL_TRACE") == "1"
        trace_rows = os.environ.get("VLLM_LAYERED_ROW_TRACE") == "1"
        if trace_rows:
            logger.info(
                "layered_row adapter_in range=[%d,%d) hidden=%s residual=%s "
                "input_ids=%s positions=%s frontier=%s",
                layer_start,
                layer_end,
                tuple(hidden_states.shape),
                None if residual is None else tuple(residual.shape),
                None if input_ids is None else tuple(input_ids.shape),
                tuple(positions.shape),
                frontier is not None,
            )
        enter_rows = int(hidden_states.shape[0])
        if trace_layered:
            logger.info(
                "Layered trace enter model=%s range=[%d,%d) tokens=%d "
                "frontier=%s input_sum=%.6e position_sum=%.6e",
                type(self.model).__name__,
                layer_start,
                layer_end,
                hidden_states.shape[0],
                frontier is not None,
                hidden_states.float().sum().item(),
                positions.float().sum().item(),
            )

        local_start = max(layer_start, self.start_layer)
        local_end = min(layer_end, self.end_layer)
        for global_idx in range(local_start, local_end):
            layer = self.layers[global_idx]
            if isinstance(layer, PPMissingLayer):
                continue
            hidden_states, residual = self._forward_layer(
                layer,
                positions,
                hidden_states,
                residual,
                input_ids,
            )
            if trace_rows and int(hidden_states.shape[0]) != enter_rows:
                logger.info(
                    "layered_row adapter_layer_row_change layer=%d "
                    "hidden=%s residual=%s enter_rows=%d",
                    global_idx,
                    tuple(hidden_states.shape),
                    None if residual is None else tuple(residual.shape),
                    enter_rows,
                )
                enter_rows = int(hidden_states.shape[0])
            if trace_layered:
                logger.info(
                    "Layered trace layer=%d hidden_sum=%.6e residual_sum=%s",
                    global_idx,
                    hidden_states.float().sum().item(),
                    "none"
                    if residual is None
                    else f"{residual.float().sum().item():.6e}",
                )

        is_final_layer = layer_end == self.num_hidden_layers
        if is_final_layer and self.end_layer == self.num_hidden_layers:
            hidden_states, residual = self._finalize(hidden_states, residual)
        if trace_rows:
            logger.info(
                "layered_row adapter_out range=[%d,%d) hidden=%s residual=%s "
                "final=%s",
                layer_start,
                layer_end,
                tuple(hidden_states.shape),
                None if residual is None else tuple(residual.shape),
                is_final_layer,
            )
        return LayeredForwardOutput(hidden_states, residual, is_final_layer)

    def make_transport_frontier(
        self,
        num_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> LayeredPrefillFrontier:
        """Create a correctly shaped placeholder for pre-owner PP stages."""

        factory = getattr(self.model, "make_empty_intermediate_tensors", None)
        if factory is None:
            factory = getattr(
                self.backbone, "make_empty_intermediate_tensors", None
            )
        if factory is None:
            raise TypeError(
                f"Model {type(self.model).__name__} cannot create PP "
                "intermediate tensors"
            )
        intermediate_tensors = factory(num_tokens, dtype, device)
        return (
            intermediate_tensors["hidden_states"],
            intermediate_tensors.tensors.get("residual"),
        )

    @abstractmethod
    def to_intermediate_tensors(
        self, output: LayeredForwardOutput
    ) -> IntermediateTensors:
        """Convert a layer-group result to the model's normal PP schema."""


class StandardDecoderLayeredPrefillAdapter(LayeredPrefillModelAdapter):
    """Automatic adapter for the common vLLM hidden/residual decoder contract."""

    def __init__(self, model: nn.Module):
        super().__init__(model)
        self._validate_layer_contract()

    def _validate_layer_contract(self) -> None:
        local_layers = [
            self.layers[index]
            for index in range(self.start_layer, self.end_layer)
            if not isinstance(self.layers[index], PPMissingLayer)
        ]
        if not local_layers:
            raise TypeError(
                f"Model {type(self.model).__name__} has no local decoder layer"
            )
        for layer in local_layers:
            try:
                parameters = self._forward_parameters(layer)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    f"Cannot inspect decoder layer {type(layer).__name__} in "
                    f"model {type(self.model).__name__}"
                ) from error
            parameter_names = tuple(parameter.name for parameter in parameters)
            standard_prefix = parameters[:3]
            standard_kinds = tuple(parameter.kind for parameter in standard_prefix)
            has_only_extra_kwargs = (
                len(parameters) == 4
                and parameters[-1].kind == inspect.Parameter.VAR_KEYWORD
            )
            standard_parameters_valid = len(parameters) >= 3 and all(
                kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
                for kind in standard_kinds
            )
            if not standard_parameters_valid or not (
                len(parameters) == 3 or has_only_extra_kwargs
            ):
                raise TypeError(
                    f"Model {type(self.model).__name__} does not use the standard "
                    "decoder layer contract (positions, hidden_states, residual); "
                    f"{type(layer).__name__} exposes {parameter_names}"
                )
        if not isinstance(self.backbone.norm, PPMissingLayer):
            try:
                inspect.signature(self.backbone.norm.forward).bind(None, None)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    f"Model {type(self.model).__name__} final norm does not "
                    "accept (hidden_states, residual)"
                ) from error

    @staticmethod
    def _forward_parameters(layer: nn.Module) -> tuple[inspect.Parameter, ...]:
        """Inspect the class method when an offloader wraps the instance."""

        parameters = tuple(inspect.signature(layer.forward).parameters.values())
        is_generic_wrapper = bool(parameters) and all(
            parameter.kind
            in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            for parameter in parameters
        )
        if not is_generic_wrapper:
            return parameters

        class_forward = inspect.signature(type(layer).forward)
        class_parameters = tuple(class_forward.parameters.values())
        if class_parameters and class_parameters[0].name in {"self", "cls"}:
            class_parameters = class_parameters[1:]
        return class_parameters

    def _forward_layer(
        self,
        layer: nn.Module,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        input_ids: torch.Tensor | None,
    ) -> LayeredPrefillFrontier:
        del input_ids
        output = layer(positions, hidden_states, residual)
        if not isinstance(output, tuple) or len(output) != 2:
            raise TypeError(
                f"Layer {type(layer).__name__} must return "
                "(hidden_states, residual) for layered prefill"
            )
        return output

    def _finalize(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> LayeredPrefillFrontier:
        output = self.backbone.norm(hidden_states, residual)
        if isinstance(output, tuple):
            if len(output) != 2:
                raise TypeError(
                    f"Final norm {type(self.backbone.norm).__name__} must return "
                    "a tensor or (hidden_states, residual)"
                )
            hidden_states = output[0]
        else:
            hidden_states = output
        return hidden_states, None

    def make_transport_frontier(
        self,
        num_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> LayeredPrefillFrontier:
        hidden_states, residual = super().make_transport_frontier(
            num_tokens, dtype, device
        )
        if residual is None:
            raise TypeError(
                f"Model {type(self.model).__name__} PP schema has no residual "
                "tensor required by the standard decoder adapter"
            )
        return hidden_states, residual

    def to_intermediate_tensors(
        self, output: LayeredForwardOutput
    ) -> IntermediateTensors:
        if output.residual is None:
            raise RuntimeError(
                "standard decoder layered output is missing its residual"
            )
        return IntermediateTensors(
            {
                "hidden_states": output.hidden_states,
                "residual": output.residual,
            }
        )
