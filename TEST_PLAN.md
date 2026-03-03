# Activation Extraction Test Plan — Gemma3 & Qwen3

## What Changed

This branch adds Gemma3 support for the activation extraction feature:

1. **`vllm/model_executor/models/gemma3.py`** — `Gemma3ForCausalLM` now implements `SupportsEagle3`, adding `get_eagle3_aux_hidden_state_layers()` which enables the `aux_hidden_state` mechanism needed for activation extraction.
2. **`vllm/model_executor/models/gemma3_mm.py`** — `Gemma3ForConditionalGeneration` (multimodal) delegates `set_aux_hidden_state_layers` / `get_eagle3_aux_hidden_state_layers` to its inner language model.
3. **`vllm/v1/worker/gpu_model_runner.py`** — Refactored activation extraction to:
   - Configure layers once at startup (merged with EAGLE3 aux layers if present)
   - Use per-request `extract_activations: bool` instead of per-request layer lists
   - Compute per-request token offset/count ranges for correct batch slicing

### API Contract

- **Startup** (engine-level): `--extract-activation-layers 5 10` or `extract_activation_layers=[5, 10]` in `LLM()`
- **Per-request** (sampling): `SamplingParams(extract_activations=True)` — boolean opt-in
- **Response**: `CompletionOutput.activations: dict[int, torch.Tensor] | None`
- **OpenAI JSON**: `choices[].activations: {"5": [float, ...], "10": [float, ...]} | null`

---

## Prerequisites

| Requirement | Detail |
|---|---|
| **GPU** | NVIDIA GPU with sufficient VRAM (≥4 GB for 270m, ≥8 GB for 0.6B) |
| **Eager mode** | `enforce_eager=True` or `CompilationConfig(mode=CompilationMode.NONE)` — aux_hidden_state hooks work without CUDA graphs |
| **Models** | `google/gemma-3-270m-it` (Gemma3, 18 layers, hidden_size=640), `Qwen/Qwen3-0.6B` (28 layers, hidden_size=1024) |
| **Access** | HuggingFace token set (`HF_TOKEN` env var) if gated models require it |

---

## Phase 1 — Smoke Tests (Offline Python API)

All tests use the offline `LLM` class. Run from the repo root.

### 1.1 Baseline generation (no activations)

Verify normal generation without any activation config.

```bash
# Gemma3 baseline
python -c "
from vllm import LLM
from vllm.sampling_params import SamplingParams

llm = LLM(model='google/gemma-3-270m-it', max_model_len=512, enforce_eager=True)
out = llm.generate(['Hello, how are you?'], SamplingParams(max_tokens=10))
text = out[0].outputs[0].text
act  = out[0].outputs[0].activations
assert text, 'No output generated'
assert act is None, f'Expected None activations, got {type(act)}'
print(f'Generated: {text}')
print('PASS: Gemma3 baseline')
"
```

```bash
# Qwen3 baseline
python -c "
from vllm import LLM
from vllm.sampling_params import SamplingParams

llm = LLM(model='Qwen/Qwen3-0.6B', max_model_len=512, enforce_eager=True)
out = llm.generate(['Hello, how are you?'], SamplingParams(max_tokens=10))
text = out[0].outputs[0].text
act  = out[0].outputs[0].activations
assert text, 'No output generated'
assert act is None, f'Expected None activations, got {type(act)}'
print(f'Generated: {text}')
print('PASS: Qwen3 baseline')
"
```

**Expected**: Both produce text, `activations` is `None`.

---

### 1.2 Single-layer activation extraction

Note on activation shapes: The aux_hidden_state mechanism captures activations from the
**last forward step** of generation. During autoregressive decode, each step processes 1
token, so `collected_activations[layer]` is overwritten each step. The final shape is
`[num_tokens_in_last_step, hidden_size]`, which is typically `[1, hidden_size]` for
decode steps. Prefill may produce `[prompt_len, hidden_size]`.

```bash
# Gemma3 — extract layer 5
python -c "
import torch
from vllm import LLM
from vllm.sampling_params import SamplingParams

HIDDEN_SIZE = 640  # google/gemma-3-270m-it

llm = LLM(
    model='google/gemma-3-270m-it',
    max_model_len=512,
    enforce_eager=True,
    extract_activation_layers=[5],
)
sp = SamplingParams(temperature=0, max_tokens=10, extract_activations=True)
out = llm.generate(['What is the capital of France?'], sp)
act = out[0].outputs[0].activations

assert act is not None, 'No activations returned'
assert set(act.keys()) == {5}, f'Expected key {{5}}, got {set(act.keys())}'
tensor = act[5]
print(f'Shape: {tensor.shape}, dtype: {tensor.dtype}')
print(f'Mean: {tensor.float().mean():.4f}, Std: {tensor.float().std():.4f}')
assert tensor.shape[-1] == HIDDEN_SIZE, f'Expected hidden_size={HIDDEN_SIZE}, got {tensor.shape[-1]}'
assert tensor.ndim == 2, f'Expected 2D tensor, got {tensor.ndim}D'
assert torch.isfinite(tensor).all(), 'Tensor contains NaN or Inf'
assert tensor.abs().sum() > 0, 'Tensor is all zeros'
print('PASS: Gemma3 single-layer extraction')
"
```

```bash
# Qwen3 — extract layer 5
python -c "
import torch
from vllm import LLM
from vllm.sampling_params import SamplingParams

HIDDEN_SIZE = 1024  # Qwen/Qwen3-0.6B

llm = LLM(
    model='Qwen/Qwen3-0.6B',
    max_model_len=512,
    enforce_eager=True,
    extract_activation_layers=[5],
)
sp = SamplingParams(temperature=0, max_tokens=10, extract_activations=True)
out = llm.generate(['What is the capital of France?'], sp)
act = out[0].outputs[0].activations

assert act is not None, 'No activations returned'
assert set(act.keys()) == {5}, f'Expected key {{5}}, got {set(act.keys())}'
tensor = act[5]
print(f'Shape: {tensor.shape}, dtype: {tensor.dtype}')
print(f'Mean: {tensor.float().mean():.4f}, Std: {tensor.float().std():.4f}')
assert tensor.shape[-1] == HIDDEN_SIZE, f'Expected hidden_size={HIDDEN_SIZE}, got {tensor.shape[-1]}'
assert tensor.ndim == 2, f'Expected 2D tensor, got {tensor.ndim}D'
assert torch.isfinite(tensor).all(), 'Tensor contains NaN or Inf'
assert tensor.abs().sum() > 0, 'Tensor is all zeros'
print('PASS: Qwen3 single-layer extraction')
"
```

**Check**:
- `activations` dict has exactly one key: `5`
- Tensor shape: `[num_tokens, hidden_size]` where hidden_size is 640 (Gemma3) / 1024 (Qwen3)
- `num_tokens` is the number of tokens in the last forward step (typically 1 during decode)
- No NaN/Inf, not all zeros

---

### 1.3 Multi-layer activation extraction

```bash
# Gemma3 — 18 layers, valid indices 0–17
python -c "
import torch
from vllm import LLM
from vllm.sampling_params import SamplingParams

LAYERS = [0, 5, 10, 17]
HIDDEN_SIZE = 640  # google/gemma-3-270m-it

llm = LLM(
    model='google/gemma-3-270m-it',
    max_model_len=512,
    enforce_eager=True,
    extract_activation_layers=LAYERS,
)
sp = SamplingParams(temperature=0, max_tokens=10, extract_activations=True)
out = llm.generate(['What is the capital of France?'], sp)
act = out[0].outputs[0].activations

assert act is not None, 'No activations returned'
assert set(act.keys()) == set(LAYERS), f'Expected {set(LAYERS)}, got {set(act.keys())}'
shapes = {k: v.shape for k, v in act.items()}
print(f'Layer shapes: {shapes}')

# All should have same hidden_size and same num_tokens
hidden_sizes = set(v.shape[-1] for v in act.values())
num_tokens   = set(v.shape[0] for v in act.values())
assert hidden_sizes == {HIDDEN_SIZE}, f'Expected hidden_size={HIDDEN_SIZE}, got {hidden_sizes}'
assert len(num_tokens) == 1, f'Inconsistent token counts: {num_tokens}'

# First vs last layer should differ
assert not torch.allclose(act[0].float(), act[17].float()), 'Layer 0 and 17 are identical'
for k, v in act.items():
    assert torch.isfinite(v).all(), f'Layer {k} has NaN/Inf'
    assert v.abs().sum() > 0, f'Layer {k} is all zeros'
print('PASS: Gemma3 multi-layer extraction')
"
```

```bash
# Qwen3 — 28 layers, valid indices 0–27
python -c "
import torch
from vllm import LLM
from vllm.sampling_params import SamplingParams

LAYERS = [0, 5, 14, 27]
HIDDEN_SIZE = 1024  # Qwen/Qwen3-0.6B

llm = LLM(
    model='Qwen/Qwen3-0.6B',
    max_model_len=512,
    enforce_eager=True,
    extract_activation_layers=LAYERS,
)
sp = SamplingParams(temperature=0, max_tokens=10, extract_activations=True)
out = llm.generate(['What is the capital of France?'], sp)
act = out[0].outputs[0].activations

assert act is not None, 'No activations returned'
assert set(act.keys()) == set(LAYERS), f'Expected {set(LAYERS)}, got {set(act.keys())}'
shapes = {k: v.shape for k, v in act.items()}
print(f'Layer shapes: {shapes}')

hidden_sizes = set(v.shape[-1] for v in act.values())
num_tokens   = set(v.shape[0] for v in act.values())
assert hidden_sizes == {HIDDEN_SIZE}, f'Expected hidden_size={HIDDEN_SIZE}, got {hidden_sizes}'
assert len(num_tokens) == 1, f'Inconsistent token counts: {num_tokens}'
assert not torch.allclose(act[0].float(), act[27].float()), 'Layer 0 and 27 are identical'
for k, v in act.items():
    assert torch.isfinite(v).all(), f'Layer {k} has NaN/Inf'
print('PASS: Qwen3 multi-layer extraction')
"
```

---

### 1.4 Per-request opt-in / opt-out

Layers are configured at startup; per-request boolean controls whether activations are returned.

```bash
# Gemma3
python -c "
from vllm import LLM
from vllm.sampling_params import SamplingParams

llm = LLM(
    model='google/gemma-3-270m-it',
    max_model_len=512,
    enforce_eager=True,
    extract_activation_layers=[5],
)

# Request WITH activations
sp_on = SamplingParams(max_tokens=5, extract_activations=True)
out_on = llm.generate(['Hello'], sp_on)

# Request WITHOUT activations
sp_off = SamplingParams(max_tokens=5, extract_activations=False)
out_off = llm.generate(['Hello'], sp_off)

assert out_on[0].outputs[0].activations is not None, 'Should have activations'
assert out_off[0].outputs[0].activations is None,    'Should NOT have activations'
print('PASS: Gemma3 per-request opt-in/out')
"
```

```bash
# Qwen3
python -c "
from vllm import LLM
from vllm.sampling_params import SamplingParams

llm = LLM(
    model='Qwen/Qwen3-0.6B',
    max_model_len=512,
    enforce_eager=True,
    extract_activation_layers=[5],
)
sp_on  = SamplingParams(max_tokens=5, extract_activations=True)
sp_off = SamplingParams(max_tokens=5, extract_activations=False)
out_on  = llm.generate(['Hello'], sp_on)
out_off = llm.generate(['Hello'], sp_off)
assert out_on[0].outputs[0].activations is not None, 'Should have activations'
assert out_off[0].outputs[0].activations is None,    'Should NOT have activations'
print('PASS: Qwen3 per-request opt-in/out')
"
```

---

### 1.5 Mixed-batch activation slicing

Multiple prompts in one batch, only some requesting activations. Verifies per-request token slicing.

```bash
# Gemma3
python -c "
from vllm import LLM
from vllm.sampling_params import SamplingParams

llm = LLM(
    model='google/gemma-3-270m-it',
    max_model_len=512,
    enforce_eager=True,
    extract_activation_layers=[5],
)

sp_on  = SamplingParams(max_tokens=8, extract_activations=True)
sp_off = SamplingParams(max_tokens=8, extract_activations=False)

outputs = llm.generate(
    ['Prompt A about cats', 'Prompt B about dogs', 'Prompt C about birds'],
    [sp_on, sp_off, sp_on],
)

assert outputs[0].outputs[0].activations is not None, 'Request 0 should have activations'
assert outputs[1].outputs[0].activations is None,     'Request 1 should NOT have activations'
assert outputs[2].outputs[0].activations is not None, 'Request 2 should have activations'

act_0 = outputs[0].outputs[0].activations[5]
act_2 = outputs[2].outputs[0].activations[5]
print(f'Request 0 activation shape: {act_0.shape}')
print(f'Request 2 activation shape: {act_2.shape}')

assert act_0.shape[-1] == 640, f'Wrong hidden size: {act_0.shape[-1]}'
assert act_2.shape[-1] == 640, f'Wrong hidden size: {act_2.shape[-1]}'
print('PASS: Gemma3 mixed-batch slicing')
"
```

```bash
# Qwen3
python -c "
from vllm import LLM
from vllm.sampling_params import SamplingParams

llm = LLM(
    model='Qwen/Qwen3-0.6B',
    max_model_len=512,
    enforce_eager=True,
    extract_activation_layers=[5],
)

sp_on  = SamplingParams(max_tokens=8, extract_activations=True)
sp_off = SamplingParams(max_tokens=8, extract_activations=False)

outputs = llm.generate(
    ['Prompt A about cats', 'Prompt B about dogs', 'Prompt C about birds'],
    [sp_on, sp_off, sp_on],
)

assert outputs[0].outputs[0].activations is not None, 'Request 0 should have activations'
assert outputs[1].outputs[0].activations is None,     'Request 1 should NOT have activations'
assert outputs[2].outputs[0].activations is not None, 'Request 2 should have activations'
act_0 = outputs[0].outputs[0].activations[5]
print(f'Request 0 activation shape: {act_0.shape}')
assert act_0.shape[-1] == 1024, f'Wrong hidden size: {act_0.shape[-1]}'
print('PASS: Qwen3 mixed-batch slicing')
"
```

---

### 1.6 Out-of-range layer index

```bash
# Gemma3 has 18 layers — index 20 is out of range
python -c "
from vllm import LLM
from vllm.sampling_params import SamplingParams

try:
    llm = LLM(
        model='google/gemma-3-270m-it',
        max_model_len=512,
        enforce_eager=True,
        extract_activation_layers=[20],
    )
    sp = SamplingParams(max_tokens=5, extract_activations=True)
    out = llm.generate(['Hello'], sp)
    act = out[0].outputs[0].activations
    if act and 20 in act:
        print('OBSERVATION: Out-of-range layer 20 returned activations (unexpected)')
    elif act is None or 20 not in act:
        print('OBSERVATION: Out-of-range layer 20 was silently ignored')
except Exception as e:
    print(f'OBSERVATION: Error raised for out-of-range layer: {type(e).__name__}: {e}')
"
```

**Expected**: Either a clear error or silent skip. Document the behavior.

---

## Phase 2 — OpenAI-Compatible Server Tests

### 2.1 Start the Gemma3 server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model google/gemma-3-270m-it \
    --max-model-len 512 \
    --enforce-eager \
    --extract-activation-layers 5 10
```

Wait for `"Uvicorn running on http://0.0.0.0:8000"`, then in a separate terminal:

### 2.2 Chat completion WITH activations

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-3-270m-it",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 10,
    "extract_activations": true
  }' | python3 -c "
import json, sys
resp = json.load(sys.stdin)
act = resp['choices'][0].get('activations')
assert act is not None, 'No activations in response'
assert '5' in act and '10' in act, f'Expected keys 5,10, got {list(act.keys())}'
print(f'Layer 5: {len(act[\"5\"])} floats')
print(f'Layer 10: {len(act[\"10\"])} floats')
assert not any(v != v for v in act['5']), 'NaN in layer 5'   # NaN != NaN
assert not any(v != v for v in act['10']), 'NaN in layer 10'
print('PASS: chat completion with activations')
"
```

### 2.3 Chat completion WITHOUT activations (default)

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-3-270m-it",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 10
  }' | python3 -c "
import json, sys
resp = json.load(sys.stdin)
act = resp['choices'][0].get('activations')
assert act is None, f'Expected no activations, got {type(act)}'
print('PASS: chat completion without activations')
"
```

### 2.4 Text completion WITH activations

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-3-270m-it",
    "prompt": "The capital of France is",
    "max_tokens": 10,
    "extract_activations": true
  }' | python3 -c "
import json, sys
resp = json.load(sys.stdin)
act = resp['choices'][0].get('activations')
assert act is not None, 'No activations in response'
assert '5' in act and '10' in act, f'Expected keys 5,10, got {list(act.keys())}'
print(f'Layer 5: {len(act[\"5\"])} floats')
print(f'Layer 10: {len(act[\"10\"])} floats')
print('PASS: text completion with activations')
"
```

### 2.5 Repeat with Qwen3

Stop the Gemma3 server, then:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-0.6B \
    --max-model-len 512 \
    --enforce-eager \
    --extract-activation-layers 5 10
```

Then repeat 2.2–2.4 with `"model": "Qwen/Qwen3-0.6B"`.

---

## Phase 3 — Correctness & Regression

### 3.1 Determinism

```bash
python -c "
import torch
from vllm import LLM
from vllm.sampling_params import SamplingParams

llm = LLM(
    model='google/gemma-3-270m-it',
    max_model_len=512,
    enforce_eager=True,
    extract_activation_layers=[5],
)
sp = SamplingParams(temperature=0, max_tokens=10, extract_activations=True)

out1 = llm.generate(['What is 2+2?'], sp)
out2 = llm.generate(['What is 2+2?'], sp)

act1 = out1[0].outputs[0].activations[5]
act2 = out2[0].outputs[0].activations[5]
assert torch.equal(act1, act2), 'Activations differ across identical runs'
print(f'Activation shape: {act1.shape}')
print('PASS: Gemma3 determinism')
"
```

### 3.2 Non-zero activations

```bash
python -c "
import torch
from vllm import LLM
from vllm.sampling_params import SamplingParams

llm = LLM(
    model='google/gemma-3-270m-it',
    max_model_len=512,
    enforce_eager=True,
    extract_activation_layers=[0, 5, 17],
)
sp = SamplingParams(temperature=0, max_tokens=10, extract_activations=True)
out = llm.generate(['Hello world'], sp)
act = out[0].outputs[0].activations

for layer_idx, tensor in act.items():
    assert tensor.abs().sum() > 0, f'Layer {layer_idx} activations are all zeros'
    print(f'Layer {layer_idx}: abs_sum={tensor.abs().sum():.2f}, shape={tensor.shape}')
print('PASS: non-zero activations')
"
```

### 3.3 Different prompts produce different activations

```bash
python -c "
import torch
from vllm import LLM
from vllm.sampling_params import SamplingParams

llm = LLM(
    model='google/gemma-3-270m-it',
    max_model_len=512,
    enforce_eager=True,
    extract_activation_layers=[5],
)
sp = SamplingParams(temperature=0, max_tokens=10, extract_activations=True)

out_a = llm.generate(['Hello'], sp)
out_b = llm.generate(['Explain quantum physics in detail'], sp)

act_a = out_a[0].outputs[0].activations[5]
act_b = out_b[0].outputs[0].activations[5]

# Shapes may differ (different prompt lengths), but values should not be identical
if act_a.shape == act_b.shape:
    assert not torch.allclose(act_a.float(), act_b.float(), atol=1e-6), \
        'Different prompts produced identical activations'
print(f'Prompt A activation shape: {act_a.shape}')
print(f'Prompt B activation shape: {act_b.shape}')
print('PASS: different prompts produce different activations')
"
```

### 3.4 Generation quality unchanged by activation extraction

```bash
python -c "
from vllm import LLM
from vllm.sampling_params import SamplingParams

# Without extraction
llm_plain = LLM(model='google/gemma-3-270m-it', max_model_len=512, enforce_eager=True)
out_plain = llm_plain.generate(
    ['What is AI?'], SamplingParams(temperature=0, max_tokens=20)
)
text_plain = out_plain[0].outputs[0].text
del llm_plain  # free GPU memory

# With extraction
llm_act = LLM(
    model='google/gemma-3-270m-it',
    max_model_len=512,
    enforce_eager=True,
    extract_activation_layers=[5],
)
sp_act = SamplingParams(temperature=0, max_tokens=20, extract_activations=True)
out_act = llm_act.generate(['What is AI?'], sp_act)
text_act = out_act[0].outputs[0].text

print(f'Without extraction: {text_plain!r}')
print(f'With extraction:    {text_act!r}')
assert text_plain == text_act, 'Activation extraction changed generation output!'
print('PASS: generation quality unchanged')
"
```

---

## Phase 4 — Edge Cases

### 4.1 Single token generation

```bash
python -c "
from vllm import LLM
from vllm.sampling_params import SamplingParams

llm = LLM(
    model='google/gemma-3-270m-it',
    max_model_len=512,
    enforce_eager=True,
    extract_activation_layers=[5],
)
sp = SamplingParams(temperature=0, max_tokens=1, extract_activations=True)
out = llm.generate(['Hello'], sp)
act = out[0].outputs[0].activations
assert act is not None, 'No activations'
tensor = act[5]
print(f'Shape: {tensor.shape}')
# First dim should be 1 (single token) or could be prompt+1
print('PASS: single token generation')
"
```

### 4.2 Long generation

```bash
python -c "
import torch
from vllm import LLM
from vllm.sampling_params import SamplingParams

llm = LLM(
    model='google/gemma-3-270m-it',
    max_model_len=512,
    enforce_eager=True,
    extract_activation_layers=[5],
)
sp = SamplingParams(temperature=0, max_tokens=256, extract_activations=True)
out = llm.generate(['Tell me a long story'], sp)
act = out[0].outputs[0].activations
assert act is not None, 'No activations'
tensor = act[5]
print(f'Shape: {tensor.shape}')
assert torch.isfinite(tensor).all(), 'Contains NaN/Inf'
print('PASS: long generation (256 tokens)')
"
```

### 4.3 First layer (layer 0)

```bash
python -c "
import torch
from vllm import LLM
from vllm.sampling_params import SamplingParams

llm = LLM(
    model='google/gemma-3-270m-it',
    max_model_len=512,
    enforce_eager=True,
    extract_activation_layers=[0],
)
sp = SamplingParams(temperature=0, max_tokens=5, extract_activations=True)
out = llm.generate(['Hello'], sp)
act = out[0].outputs[0].activations
assert act is not None and 0 in act, 'Layer 0 not in activations'
assert torch.isfinite(act[0]).all(), 'Layer 0 has NaN/Inf'
print(f'Layer 0 shape: {act[0].shape}')
print('PASS: first layer extraction')
"
```

### 4.4 Last layer (layer 17 for Gemma3)

```bash
python -c "
import torch
from vllm import LLM
from vllm.sampling_params import SamplingParams

llm = LLM(
    model='google/gemma-3-270m-it',
    max_model_len=512,
    enforce_eager=True,
    extract_activation_layers=[17],
)
sp = SamplingParams(temperature=0, max_tokens=5, extract_activations=True)
out = llm.generate(['Hello'], sp)
act = out[0].outputs[0].activations
assert act is not None and 17 in act, 'Layer 17 not in activations'
assert torch.isfinite(act[17]).all(), 'Layer 17 has NaN/Inf'
print(f'Layer 17 shape: {act[17].shape}')
print('PASS: last layer extraction')
"
```

### 4.5 All layers (0..17 for Gemma3)

```bash
python -c "
import torch
from vllm import LLM
from vllm.sampling_params import SamplingParams

ALL_LAYERS = list(range(18))
llm = LLM(
    model='google/gemma-3-270m-it',
    max_model_len=512,
    enforce_eager=True,
    extract_activation_layers=ALL_LAYERS,
)
sp = SamplingParams(temperature=0, max_tokens=5, extract_activations=True)
out = llm.generate(['Hello'], sp)
act = out[0].outputs[0].activations
assert act is not None, 'No activations'
assert set(act.keys()) == set(ALL_LAYERS), f'Missing layers: {set(ALL_LAYERS) - set(act.keys())}'
for idx in ALL_LAYERS:
    assert torch.isfinite(act[idx]).all(), f'Layer {idx} has NaN/Inf'
print(f'All {len(act)} layers extracted successfully')
print('PASS: all layers extraction')
"
```

### 4.6 Empty prompt

```bash
python -c "
from vllm import LLM
from vllm.sampling_params import SamplingParams

llm = LLM(
    model='google/gemma-3-270m-it',
    max_model_len=512,
    enforce_eager=True,
    extract_activation_layers=[5],
)
sp = SamplingParams(max_tokens=5, extract_activations=True)
try:
    out = llm.generate([''], sp)
    print(f'Generated: {out[0].outputs[0].text!r}')
    act = out[0].outputs[0].activations
    print(f'Activations: {\"present\" if act else \"None\"}')
    print('OBSERVATION: empty prompt handled without crash')
except Exception as e:
    print(f'OBSERVATION: empty prompt raised error: {type(e).__name__}: {e}')
"
```

---

## Phase 5 — Docker (if applicable)

```bash
# Build
docker build -t vllm-activation .

# Run with Gemma3 + activation layers 5 and 10
docker run --gpus all -p 8000:8000 vllm-activation \
    --model google/gemma-3-270m-it \
    --max-model-len 512 \
    --enforce-eager \
    --extract-activation-layers 5 10
```

Then run the curl tests from Phase 2 against `http://localhost:8000`.

---

## Phase 6 — Existing Test Suite

Run the project's relevant existing tests to check for regressions:

```bash
# Model-specific tests (if any exist)
python -m pytest tests/models/ -k "gemma" -x -v --timeout=600 2>&1 | tail -30

# Sampling params tests
python -m pytest tests/ -k "sampling_param" -x -v --timeout=120 2>&1 | tail -30

# Test the existing activation test script (uses old API — may need updating)
python test_activation.py --model gemma --layers 5 --max-tokens 10
```

---

## Summary Checklist

| # | Test | Gemma3 | Qwen3 |
|---|------|--------|-------|
| 1.1 | Baseline (no activations) | [ ] | [ ] |
| 1.2 | Single-layer extraction | [ ] | [ ] |
| 1.3 | Multi-layer extraction | [ ] | [ ] |
| 1.4 | Per-request opt-in/out | [ ] | [ ] |
| 1.5 | Mixed-batch slicing | [ ] | [ ] |
| 1.6 | Out-of-range layer | [ ] | [ ] |
| 2.2 | OpenAI chat completion + activations | [ ] | [ ] |
| 2.3 | OpenAI chat completion − activations | [ ] | [ ] |
| 2.4 | OpenAI text completion + activations | [ ] | [ ] |
| 3.1 | Determinism | [ ] | [ ] |
| 3.2 | Non-zero activations | [ ] | [ ] |
| 3.3 | Different prompts → different activations | [ ] | [ ] |
| 3.4 | Generation quality unchanged | [ ] | [ ] |
| 4.1 | Single token (`max_tokens=1`) | [ ] | [ ] |
| 4.2 | Long generation (`max_tokens=256`) | [ ] | [ ] |
| 4.3 | First layer (layer 0) | [ ] | [ ] |
| 4.4 | Last layer (layer N-1) | [ ] | [ ] |
| 4.5 | All layers | [ ] | [ ] |
| 4.6 | Empty prompt | [ ] | [ ] |
| 5 | Docker | [ ] | [ ] |
