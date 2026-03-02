# Activation Extraction Test Plan — Gemma3 & Qwen3

## Prerequisites

| Requirement | Detail |
|---|---|
| **GPU** | NVIDIA GPU with sufficient VRAM (8 GB+ for small models) |
| **Compilation** | Must be disabled — use `enforce_eager=True` or `CompilationConfig(mode=CompilationMode.NONE)` |
| **Models** | `google/gemma-3-270m-it` (Gemma3, 18 layers), a Qwen3 variant e.g. `Qwen/Qwen3-0.6B` (28 layers) |
| **Access** | HuggingFace token set if gated models are used |

---

## Phase 1 — Smoke Tests (Python API)

These use the offline `LLM` class directly. Run each with `enforce_eager=True`.

### 1.1 Baseline: generation without activations

Verify normal generation still works for both models.

```bash
# Gemma3
python -c "
from vllm import LLM
from vllm.sampling_params import SamplingParams
llm = LLM(model='google/gemma-3-270m-it', max_model_len=512, enforce_eager=True)
out = llm.generate(['Hello'], SamplingParams(max_tokens=10))
assert out[0].outputs[0].text, 'No output generated'
assert out[0].outputs[0].activations is None
print('PASS: Gemma3 baseline')
"

# Qwen3
python -c "
from vllm import LLM
from vllm.sampling_params import SamplingParams
llm = LLM(model='Qwen/Qwen3-0.6B', max_model_len=512, enforce_eager=True)
out = llm.generate(['Hello'], SamplingParams(max_tokens=10))
assert out[0].outputs[0].text, 'No output generated'
assert out[0].outputs[0].activations is None
print('PASS: Qwen3 baseline')
"
```

**Expected**: Both produce text, `activations` is `None`.

---

### 1.2 Single-layer activation extraction

```bash
python test_activation.py --model google/gemma-3-270m-it --layers 5 --max-tokens 10
python test_activation.py --model Qwen/Qwen3-0.6B         --layers 5 --max-tokens 10
```

**Check**:
- `activations` dict is non-empty
- Exactly one key: `5`
- Tensor shape is `[num_generated_tokens, hidden_size]` — the first dim should equal `max_tokens` (10) and the second dim should match the model's hidden size
- Values are finite (no NaN/Inf): `torch.isfinite(tensor).all()`

---

### 1.3 Multi-layer activation extraction

```bash
# Gemma3 has 18 layers (valid indices: 0–17)
python test_activation.py --model google/gemma-3-270m-it --layers 0 5 10 17 --max-tokens 10

# Qwen3-0.6B has 28 layers (valid indices: 0–27)
python test_activation.py --model Qwen/Qwen3-0.6B         --layers 0 5 14 27 --max-tokens 10
```

**Check**:
- `activations` dict has 4 keys: `{0, 5, 10, 17}` / `{0, 5, 14, 27}`
- All tensors have the same `hidden_size` dimension
- All tensors have the same `num_tokens` dimension (= `max_tokens`)
- First-layer and last-layer activations should have different values (not identical copies)

---

### 1.4 Activation opt-in per request

Verify that `extract_activations=False` suppresses extraction even when `--extract-activation-layers` is configured at startup.

```python
from vllm import LLM
from vllm.sampling_params import SamplingParams
from vllm.config.compilation import CompilationConfig, CompilationMode

llm = LLM(
    model="google/gemma-3-270m-it",
    max_model_len=512,
    compilation_config=CompilationConfig(mode=CompilationMode.NONE),
    extract_activation_layers=[5],
)

# Request WITH activations
sp_on = SamplingParams(max_tokens=5, extract_activations=True)
out_on = llm.generate(["Hello"], sp_on)

# Request WITHOUT activations
sp_off = SamplingParams(max_tokens=5, extract_activations=False)
out_off = llm.generate(["Hello"], sp_off)

assert out_on[0].outputs[0].activations is not None, "Should have activations"
assert out_off[0].outputs[0].activations is None,    "Should NOT have activations"
print("PASS: per-request opt-in/out")
```

Repeat the same test with the Qwen3 model.

---

### 1.5 Batch with mixed activation requests

Verify per-request slicing in a batch where only some requests ask for activations.

```python
from vllm import LLM
from vllm.sampling_params import SamplingParams
from vllm.config.compilation import CompilationConfig, CompilationMode

llm = LLM(
    model="google/gemma-3-270m-it",
    max_model_len=512,
    compilation_config=CompilationConfig(mode=CompilationMode.NONE),
    extract_activation_layers=[5],
)

sp_on  = SamplingParams(max_tokens=8, extract_activations=True)
sp_off = SamplingParams(max_tokens=8, extract_activations=False)

# Three prompts: only first and third request activations
outputs = llm.generate(
    ["Prompt A", "Prompt B", "Prompt C"],
    [sp_on, sp_off, sp_on],
)

assert outputs[0].outputs[0].activations is not None, "Request 0 should have activations"
assert outputs[1].outputs[0].activations is None,     "Request 1 should NOT have activations"
assert outputs[2].outputs[0].activations is not None,  "Request 2 should have activations"

# Each request's activation should be sliced to its OWN tokens, not the full batch
act_0 = outputs[0].outputs[0].activations[5]
act_2 = outputs[2].outputs[0].activations[5]
assert act_0.shape[0] == 8, f"Expected 8 tokens, got {act_0.shape[0]}"
assert act_2.shape[0] == 8, f"Expected 8 tokens, got {act_2.shape[0]}"
print("PASS: mixed-batch slicing")
```

Repeat with Qwen3.

---

### 1.6 Out-of-range layer index

Request a layer index beyond the model's layer count.

```python
# Gemma3 has 18 layers — index 20 is out of range
sp = SamplingParams(max_tokens=5, extract_activation_layers=[20])
```

**Expected**: Either a clear error at startup/request time, or the layer is silently ignored. Document whichever behavior you observe.

---

## Phase 2 — OpenAI-Compatible Server Tests

Start the server, then call it with `curl` or the `openai` Python client.

### 2.1 Start the server

```bash
# Gemma3
python -m vllm.entrypoints.openai.api_server \
    --model google/gemma-3-270m-it \
    --max-model-len 512 \
    --enforce-eager \
    --extract-activation-layers 5 10

# Qwen3
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-0.6B \
    --max-model-len 512 \
    --enforce-eager \
    --extract-activation-layers 5 10
```

### 2.2 Chat completion with activations

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-3-270m-it",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 10,
    "extract_activations": true
  }' | python -m json.tool
```

**Check**:
- Response has `choices[0].activations`
- Keys are `"5"` and `"10"` (string-typed layer indices)
- Each value is a flat list of floats
- List length = `num_generated_tokens * hidden_size`
- No NaN values in the list

### 2.3 Chat completion without activations (default)

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-3-270m-it",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 10
  }' | python -m json.tool
```

**Check**: `activations` field is absent or `null`.

### 2.4 Text completion with activations

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-3-270m-it",
    "prompt": "The capital of France is",
    "max_tokens": 10,
    "extract_activations": true
  }' | python -m json.tool
```

**Check**: Same as 2.2 — `activations` present in `choices[0]`.

### 2.5 Repeat 2.2–2.4 with Qwen3

Swap model to `Qwen/Qwen3-0.6B` and repeat. Hidden size will differ; verify the list lengths are consistent with Qwen3's hidden dimension.

---

## Phase 3 — Correctness & Regression

### 3.1 Determinism

Run the same prompt twice with `temperature=0` and activation extraction on.

**Check**: Activation tensors are bitwise identical across both runs (same model, same input → same hidden states).

### 3.2 Activation values are not all zeros

```python
for layer_idx, tensor in activations.items():
    assert tensor.abs().sum() > 0, f"Layer {layer_idx} activations are all zeros"
```

### 3.3 Different prompts produce different activations

Generate activations for two semantically distinct prompts (e.g., "Hello" vs "Explain quantum physics"). Verify the tensors differ.

### 3.4 Generation quality unchanged

Compare output text with and without `--extract-activation-layers`. The generated text for a deterministic config (`temperature=0`) should be identical — activation hooks must not alter the forward pass.

```python
# Without extraction
llm_plain = LLM(model="google/gemma-3-270m-it", max_model_len=512, enforce_eager=True)
out_plain = llm_plain.generate(["What is AI?"], SamplingParams(temperature=0, max_tokens=20))

# With extraction
llm_act = LLM(
    model="google/gemma-3-270m-it", max_model_len=512, enforce_eager=True,
    extract_activation_layers=[5],
)
sp_act = SamplingParams(temperature=0, max_tokens=20, extract_activations=True)
out_act = llm_act.generate(["What is AI?"], sp_act)

assert out_plain[0].outputs[0].text == out_act[0].outputs[0].text, \
    "Activation extraction should NOT change generation output"
```

Repeat with Qwen3.

---

## Phase 4 — Edge Cases

| # | Test | Expected |
|---|------|----------|
| 4.1 | `max_tokens=1` — single token generation | Activation shape `[1, hidden_size]` |
| 4.2 | Long generation (`max_tokens=256`) | Activation shape `[256, hidden_size]`, no OOM for small models |
| 4.3 | Layer 0 (first layer) | Valid activations returned |
| 4.4 | Last layer (index `num_layers - 1`) | Valid activations returned |
| 4.5 | All layers (every index `0..num_layers-1`) | All returned — check memory impact |
| 4.6 | Empty prompt (`""`) | Either generates normally or returns an error — should not crash |
| 4.7 | `enforce_eager=False` without `CompilationMode.NONE` | Should warn or error that hooks won't work with `torch.compile` |

---

## Phase 5 — Docker (if applicable)

Using the project Dockerfile:

```bash
docker build -t vllm-activation .
docker run --gpus all -p 8000:8000 \
    -e VLLM_EXTRACT_ACTIVATION_LAYERS=5 \
    vllm-activation
```

Then run the curl tests from Phase 2 against `http://localhost:8000`.

---

## Summary Checklist

| Test | Gemma3 | Qwen3 |
|------|--------|-------|
| Baseline (no activations) | [ ] | [ ] |
| Single-layer extraction | [ ] | [ ] |
| Multi-layer extraction | [ ] | [ ] |
| Per-request opt-in/out | [ ] | [ ] |
| Mixed-batch slicing | [ ] | [ ] |
| Out-of-range layer | [ ] | [ ] |
| OpenAI chat completions | [ ] | [ ] |
| OpenAI text completions | [ ] | [ ] |
| Determinism | [ ] | [ ] |
| Non-zero activations | [ ] | [ ] |
| Different prompts → different activations | [ ] | [ ] |
| Generation quality unchanged | [ ] | [ ] |
| Edge cases (single token, long gen, first/last layer) | [ ] | [ ] |
| Docker | [ ] | [ ] |
