# Debugging Engine Core Crash

The engine core process is dying unexpectedly. Here's a systematic approach to debug this:

## Step 1: Check if it's related to activation extraction

Since activation collection is currently disabled, if the crash still happens, it might be:
- An issue with the new fields added to dataclasses/structs
- A serialization issue with `ModelRunnerOutput.activations` field
- An issue with `Request.activations` field

### Test without any activation-related code:

1. Temporarily remove the `activations` field from `ModelRunnerOutput`:
   ```python
   # In vllm/v1/outputs.py, comment out:
   # activations: dict[str, dict[int, torch.Tensor]] = field(default_factory=dict)
   ```

2. Remove `Request.activations` field:
   ```python
   # In vllm/v1/request.py, comment out:
   # self.activations: dict[int, torch.Tensor] | None = None
   ```

3. Run the test again. If it works, the issue is with these fields.

## Step 2: Enable verbose logging

Run with debug logging to see worker process errors:

```bash
VLLM_LOGGING_LEVEL=DEBUG uv run test_activation.py 2>&1 | tee debug.log
```

Look for:
- Python exceptions in worker processes
- Serialization errors
- Import errors
- Attribute errors

## Step 3: Check worker process stderr

The worker process logs might be going to stderr. Try:

```bash
# Redirect both stdout and stderr
VLLM_LOGGING_LEVEL=DEBUG uv run test_activation.py > output.log 2>&1
cat output.log | grep -i "error\|exception\|traceback\|failed"
```

## Step 4: Test with a minimal example

Create a minimal test that doesn't use activations at all:

```python
from vllm import LLM

if __name__ == "__main__":
    llm = LLM(model="google/gemma-3-270m-it")
    outputs = llm.generate(["Hello"], max_tokens=10)
    print(outputs[0].outputs[0].text)
```

If this also crashes, the issue is in the base code changes, not activation-specific.

## Step 5: Check for serialization issues

The crash might be happening when trying to serialize `ModelRunnerOutput` with the `activations` field. 

### Check if ModelRunnerOutput is being serialized:

1. Look for where `ModelRunnerOutput` is sent between processes
2. The `activations` field contains `dict[str, dict[int, torch.Tensor]]` which might not serialize properly
3. Even if the field is empty `{}`, the type annotation might cause issues

### Potential fix:

Try making the activations field optional and ensuring it's always a simple dict:

```python
# In vllm/v1/outputs.py
activations: dict[str, dict[int, torch.Tensor]] | None = None
```

Instead of:
```python
activations: dict[str, dict[int, torch.Tensor]] = field(default_factory=dict)
```

## Step 6: Use Python debugger

Add breakpoints or print statements in critical paths:

1. In `vllm/v1/worker/gpu/model_runner.py`, `sample_tokens()` method
2. In `vllm/v1/core/sched/scheduler.py`, `update_from_output()` method
3. Check if the crash happens before or after these methods

## Step 7: Check for import errors

The new `activation_collector.py` module might have import issues:

```python
# Test if the module can be imported
python -c "from vllm.activation_collector import ActivationCollector; print('OK')"
```

## Step 8: Use trace function (SLOW but detailed)

```bash
VLLM_TRACE_FUNCTION=1 uv run test_activation.py 2>&1 | tee trace.log
```

This will log every function call (very slow, but shows exactly where it crashes).

## Step 9: Check for type annotation issues

msgspec might be strict about types. Check if `dict[int, torch.Tensor]` is causing issues.

Try changing to:
```python
activations: dict[str, Any] | None = None
```

And convert layer indices to strings when storing.

## Step 10: Isolate the problematic code

Comment out sections one by one:

1. Comment out `Request.activations` field
2. Comment out `ModelRunnerOutput.activations` field  
3. Comment out `EngineCoreOutput.activations` field
4. Comment out `CompletionOutput.activations` field

Test after each change to find which one causes the crash.

## Most Likely Issues:

1. **Serialization of nested dict with torch.Tensor**: `dict[str, dict[int, torch.Tensor]]` might not serialize through multiprocessing
2. **Type annotation incompatibility**: msgspec might not like the type annotation
3. **Empty dict vs None**: Using `field(default_factory=dict)` vs `None` might matter

## Quick Test:

Try this minimal change - make activations always None initially:

```python
# In vllm/v1/outputs.py
activations: dict[str, dict[int, torch.Tensor]] | None = None  # Always None for now
```

And in the model runner, don't set it at all. If this works, the issue is with the dict structure.

