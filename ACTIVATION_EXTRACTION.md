# Activation Extraction from vLLM Models

## Overview

The `extract-activations` branch implements model-agnostic activation extraction from vLLM models during inference. Activations from any decoder layer can be captured via PyTorch forward hooks without modifying individual model files.

## Architecture

### Design: Model-Agnostic Hook-Based Capture (Approach A)

The implementation uses PyTorch `register_forward_hook` on decoder layers, discovered dynamically at runtime. This works with any model that follows the standard vLLM pattern:

```
ForCausalLM.forward()        -> returns hidden_states
ForCausalLM.compute_logits() -> called separately by gpu_model_runner
    Model.forward()           -> loops through DecoderLayer list, applies final norm
        DecoderLayer.forward() -> attention + MLP with residual connections
```

All supported architectures (Gemma, Llama, Nemotron, etc.) expose their decoder layers at `model.model.layers`, making the hook registration model-agnostic.

### Data Flow

```
SamplingParams(extract_activations=True, activation_layers=[5, 10])
    |
    v
gpu_model_runner / gpu/model_runner
    -> checks per-request flags in ExtraData
    -> wraps model forward() with ActivationCollector context manager
    -> hooks fire on specified decoder layers, capturing output tensors
    -> activations sliced per-request, moved to CPU
    |
    v
ModelRunnerOutput.activations: dict[req_id -> dict[layer_idx -> Tensor]]
    |
    v
Scheduler -> EngineCoreOutput.activations
    |
    v
OutputProcessor -> RequestState.activations -> CompletionOutput.activations
    |
    v
User receives: output.outputs[0].activations  (dict[int, torch.Tensor])
```

### Key Components

| File | Role |
|---|---|
| `vllm/activation_collector.py` | `ActivationCollector` — registers/removes forward hooks, collects activations |
| `vllm/sampling_params.py` | `extract_activations`, `activation_layers`, `activation_type` fields |
| `vllm/outputs.py` | `CompletionOutput.activations` field |
| `vllm/v1/engine/__init__.py` | `EngineCoreOutput.activations` field |
| `vllm/v1/engine/output_processor.py` | Propagates activations from engine output to `CompletionOutput` |
| `vllm/v1/request.py` | `Request.activations` field |
| `vllm/v1/outputs.py` | `ModelRunnerOutput.activations` field |
| `vllm/v1/worker/gpu_model_runner.py` | Legacy runner: activation check, collection, per-request mapping |
| `vllm/v1/worker/gpu/model_runner.py` | New runner: same logic with `ExtraData`-based flag lookup |
| `vllm/v1/worker/gpu/states.py` | `ExtraData` carries `extract_activations` and `activation_layers` per-request |
| `vllm/v1/core/sched/scheduler.py` | Passes activations from `ModelRunnerOutput` into `EngineCoreOutput` |

### ActivationCollector

The collector discovers decoder layers via common attribute paths:

1. `model.model.layers` — Llama, Gemma, Nemotron, Mistral, etc.
2. `model.transformer.h` — GPT-style models
3. `model.layers` — direct access

It registers `forward_hook` on each target layer. The hook extracts the first element of the output tuple (the hidden states tensor), detaches, clones, and moves it to CPU.

Compilation must be disabled when using hooks (`enforce_eager=True` or `CompilationMode.NONE`), since `torch.compile` bypasses PyTorch hooks.

## Supported Models

Because the design is model-agnostic, **no per-model code changes are needed**. Any model with a `model.layers` list of decoder layers works out of the box.

### Verified Architectures

| Model Family | Example Model | Layer Access Path |
|---|---|---|
| Gemma | `google/gemma-3-270m-it` | `model.model.layers` |
| Llama | `meta-llama/Llama-3.2-1B` | `model.model.layers` |
| Nemotron | `nvidia/Nemotron-Mini-4B-Instruct` | `model.model.layers` |
| NemotronH (hybrid) | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` | `model.model.layers` |

### Architectural Differences (Transparent to Activation Extraction)

| Feature | Gemma | Llama | Nemotron | NemotronH |
|---|---|---|---|---|
| Norm type | `GemmaRMSNorm` | `RMSNorm` | `NemotronLayerNorm1P` | `RMSNorm` |
| MLP activation | `GeluAndMul` (gated) | `SiluAndMul` (gated) | `relu2` (ungated) | `relu2` (ungated) |
| MLP structure | gate_up_proj + down_proj | gate_up_proj + down_proj | up_proj + down_proj | MoE + Mamba + Attention |
| RoPE | Full | Full | Partial (50%) | Full (attention layers only) |
| Embedding norm | `sqrt(hidden_size)` | None | None | None |

These differences affect the content of activations but not the extraction mechanism. Each decoder layer outputs a `(hidden_states, residual)` tuple regardless of architecture.

### Hybrid Models (NemotronH)

`NemotronHForCausalLM` (used by `NVIDIA-Nemotron-3-Nano-30B-A3B-FP8`) is a hybrid architecture with four heterogeneous layer types mixed according to `hybrid_override_pattern`:

| Pattern Char | Layer Class | What the hook captures |
|---|---|---|
| `M` | `NemotronHMambaDecoderLayer` | Mamba-2 SSM output |
| `E` | `NemotronHMoEDecoderLayer` | Sparse Mixture-of-Experts output (128 experts, top-6) |
| `*` | `NemotronHAttentionDecoderLayer` | Self-attention output |
| `-` | `NemotronHMLPDecoderLayer` | Dense MLP output |

The 30B-A3B model uses the pattern `MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME` (52 layers). All four layer types return `(hidden_states, residual)` tuples, so the hook works uniformly.

When specifying `activation_layers`, be aware that each index may correspond to a different layer type. For example, layer 0 is Mamba (`M`), layer 5 is Attention (`*`), layer 6 is MoE (`E`).

## Usage

```python
from vllm import LLM
from vllm.config.compilation import CompilationConfig, CompilationMode
from vllm.sampling_params import SamplingParams

llm = LLM(
    model="google/gemma-3-270m-it",  # or any supported model
    max_model_len=512,
    compilation_config=CompilationConfig(mode=CompilationMode.NONE),
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=10,
    extract_activations=True,
    activation_layers=[5, 10],  # specific layers, or None for all
)

outputs = llm.generate(["What is the capital of France?"], sampling_params)

for output in outputs:
    activations = output.outputs[0].activations
    if activations:
        for layer_idx, tensor in sorted(activations.items()):
            print(f"Layer {layer_idx}: shape={tensor.shape}, dtype={tensor.dtype}")
```

### Test Script

```bash
python test_activation.py                                    # default: gemma
python test_activation.py --model llama --layers 5 10 15
python test_activation.py --model nemotron --layers 0 11 23
python test_activation.py --model nvidia/Nemotron-Mini-4B-Instruct
```

## Limitations

- **Compilation must be disabled**: PyTorch hooks are bypassed by `torch.compile`. The runner auto-disables compilation when activation extraction is requested, but models already compiled at startup need `enforce_eager=True` or `CompilationMode.NONE`.
- **Per-request slicing**: The legacy runner (`gpu_model_runner.py`) stores full batch activations per request (TODO: slice by token range). The new runner (`gpu/model_runner.py`) properly slices using `query_start_loc`.
- **Memory**: Activations are cloned and moved to CPU per layer per request. For large models or many layers, this adds significant memory overhead.
- **Serialization**: `EngineCoreOutput` uses msgspec which has limitations with nested dicts of tensors. Activations are passed through the pipeline but may not serialize correctly across process boundaries in all configurations.
