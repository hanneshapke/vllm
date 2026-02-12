"""Test activation extraction with different model architectures.

Demonstrates that the activation extraction is model-agnostic and works
with Gemma, Llama, Nemotron, NemotronH (hybrid), and any other model that
follows the standard vLLM ForCausalLM + Model.layers pattern.

Usage:
    python test_activation.py                          # default: gemma
    python test_activation.py --model llama
    python test_activation.py --model nemotron
    python test_activation.py --model nemotron-h
    python test_activation.py --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
"""

import argparse

from vllm import LLM
from vllm.config.compilation import CompilationConfig, CompilationMode
from vllm.sampling_params import SamplingParams

MODELS = {
    "gemma": "google/gemma-3-270m-it",
    "llama": "meta-llama/Llama-3.2-1B",
    "nemotron": "nvidia/Nemotron-Mini-4B-Instruct",
    "nemotron-h": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
}


def main():
    parser = argparse.ArgumentParser(
        description="Test activation extraction across model architectures"
    )
    parser.add_argument(
        "--model", default="gemma", help="Model name or alias (gemma, llama, nemotron)"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[5],
        help="Layer indices to extract activations from",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=10, help="Maximum tokens to generate"
    )
    args = parser.parse_args()

    model_name = MODELS.get(args.model, args.model)
    print(f"Model: {model_name}")
    print(f"Activation layers: {args.layers}")

    llm = LLM(
        model=model_name,
        max_model_len=512,
        compilation_config=CompilationConfig(mode=CompilationMode.NONE),
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        extract_activation_layers=args.layers,
    )

    prompts = ["What is the capital of France?"]
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        print(f"\nPrompt: {output.prompt}")
        print(f"Generated: {output.outputs[0].text}")

        activations = output.outputs[0].activations
        if activations:
            print(f"Activation layers: {sorted(activations.keys())}")
            for layer_idx in sorted(activations.keys()):
                act = activations[layer_idx]
                print(
                    f"  Layer {layer_idx}: shape={act.shape}, "
                    f"dtype={act.dtype}, "
                    f"mean={act.float().mean():.4f}, "
                    f"std={act.float().std():.4f}"
                )
        else:
            print("No activations returned")


if __name__ == "__main__":
    main()
