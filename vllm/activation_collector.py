# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Activation collector for extracting intermediate layer activations during inference."""

import torch
import torch.nn as nn

from vllm.logger import init_logger

logger = init_logger(__name__)


class ActivationCollector:
    """
    Collects activations from specific layers during model forward pass.
    Uses PyTorch forward hooks to capture intermediate outputs.

    This is model-agnostic: it discovers decoder layers dynamically via
    common attribute paths (model.model.layers, model.transformer.h, etc.)
    and works with any transformer architecture (Llama, Gemma, Nemotron, etc.).
    """

    def __init__(self, model: nn.Module, layer_indices: set[int] | None = None):
        """
        Args:
            model: The transformer model (e.g., LlamaForCausalLM)
            layer_indices: Set of layer indices to collect from.
                           None = all layers.
        """
        self.model = model
        self.layer_indices = layer_indices
        self.activations: dict[int, torch.Tensor] = {}
        self.hooks: list[torch.utils.hooks.RemovableHook] = []

    def _make_hook(self, layer_idx: int):
        """Create a hook function for a specific layer."""

        def hook(module, input, output):
            try:
                if isinstance(output, tuple):
                    activation = output[0]
                else:
                    activation = output

                if not isinstance(activation, torch.Tensor):
                    logger.warning(
                        "Layer %d output is not a tensor: %s",
                        layer_idx,
                        type(activation),
                    )
                    return

                self.activations[layer_idx] = activation.detach().cpu().clone()
            except Exception as e:
                logger.error(
                    "Error collecting activation from layer %d: %s",
                    layer_idx,
                    e,
                    exc_info=True,
                )

        return hook

    def _get_layers(self):
        """Get the transformer layers from the model.

        Supports common model structures:
        - model.model.layers (Llama, Gemma, Nemotron, etc.)
        - model.transformer.h (GPT-style models)
        - model.layers (direct access)
        """
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "h"
        ):
            layers = self.model.transformer.h
        elif hasattr(self.model, "layers"):
            layers = self.model.layers
        else:
            raise ValueError(
                "Could not find transformer layers in model. "
                f"Model type: {type(self.model)}"
            )

        if self.layer_indices is not None:
            num_layers = len(layers)
            invalid_layers = [idx for idx in self.layer_indices if idx >= num_layers]
            if invalid_layers:
                logger.warning(
                    "Layer indices out of range: %s. Model has %d layers (0-%d).",
                    invalid_layers,
                    num_layers,
                    num_layers - 1,
                )
                self.layer_indices = {
                    idx for idx in self.layer_indices if idx < num_layers
                }

        return layers

    def register_hooks(self):
        """Register forward hooks on the target layers."""
        layers = self._get_layers()

        is_compiled = hasattr(self.model, "_compiled_callable") or hasattr(
            self.model, "compiled"
        )
        if is_compiled:
            logger.warning(
                "Model appears to be compiled. PyTorch hooks may not work "
                "with compiled models. Consider using enforce_eager=True."
            )

        for layer_idx, layer in enumerate(layers):
            if self.layer_indices is None or layer_idx in self.layer_indices:
                hook = layer.register_forward_hook(self._make_hook(layer_idx))
                self.hooks.append(hook)

        logger.debug("Registered activation hooks on %d layers.", len(self.hooks))

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_activations(self) -> dict[int, torch.Tensor]:
        """Get collected activations and clear the buffer."""
        activations = self.activations.copy()
        self.activations.clear()
        return activations

    def __enter__(self):
        """Context manager entry."""
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.remove_hooks()
