#!/usr/bin/env python3
"""
Debug script for activation extraction issue.

This script helps debug the engine core crash by:
1. Enabling verbose logging
2. Testing without activation extraction first
3. Gradually enabling features
"""

import os
import sys
import traceback

# Enable debug logging
os.environ["VLLM_LOGGING_LEVEL"] = "INFO"
os.environ["VLLM_LOG_STATS_INTERVAL"] = "1.0"

# For CUDA debugging (if using GPU)
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from vllm import LLM
from vllm.config.compilation import CompilationConfig, CompilationMode
from vllm.sampling_params import SamplingParams


def test_without_activations():
    """Test basic generation without activation extraction."""
    print("=" * 80)
    print("TEST 1: Basic generation WITHOUT activation extraction")
    print("=" * 80)
    try:
        llm = LLM(model="google/gemma-3-270m-it")
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=50,
            extract_activation_layers=None,  # Disabled
        )
        prompts = ["Hello, how are you?"]
        outputs = llm.generate(prompts, sampling_params)
        print("✓ SUCCESS: Basic generation works")
        for output in outputs:
            print(f"Generated: {output.outputs[0].text[:50]}...")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        traceback.print_exc()
        return False


def test_with_activations():
    """Test generation WITH activation extraction."""
    print("\n" + "=" * 80)
    print("TEST 2: Generation WITH activation extraction")
    print("=" * 80)
    try:
        # Disable compilation for activation extraction (PyTorch hooks don't work with compiled models)
        compilation_config = CompilationConfig(mode=CompilationMode.NONE)
        llm = LLM(
            model="google/gemma-3-270m-it",
            compilation_config=compilation_config,
        )
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=50,
            extract_activation_layers=[0, 5, 10],
        )
        prompts = ["Hello, how are you?"]
        outputs = llm.generate(prompts, sampling_params)
        print("✓ SUCCESS: Generation with activations works")
        for output in outputs:
            for completion in output.outputs:
                if completion.activations:
                    print(
                        f"Available activation layers: {list(completion.activations.keys())}"
                    )
                    for layer_idx, activation in completion.activations.items():
                        print(f"Layer {layer_idx} activation shape: {activation.shape}")
                else:
                    print("⚠ WARNING: No activations found in output")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Starting activation extraction debugging...")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print()

    # Test 1: Without activations
    test1_passed = test_without_activations()

    if test1_passed:
        # Test 2: With activations
        test2_passed = test_with_activations()

        if test2_passed:
            print("\n" + "=" * 80)
            print("✓ ALL TESTS PASSED")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("✗ ACTIVATION EXTRACTION FAILED")
            print("=" * 80)
            print("\nDebugging suggestions:")
            print("1. Check worker process logs (they may be in stderr)")
            print("2. Try with VLLM_TRACE_FUNCTION=1 to see function call trace")
            print(
                "3. Check if the issue is with serialization by inspecting ModelRunnerOutput"
            )
            print("4. Verify the model structure supports activation extraction")
    else:
        print("\n" + "=" * 80)
        print("✗ BASIC GENERATION FAILED - Issue is not related to activations")
        print("=" * 80)
        print("\nThe crash happens even without activation extraction.")
        print(
            "This suggests the issue is in the base code changes, not activation collection."
        )
