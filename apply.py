import gc
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

parser = argparse.ArgumentParser(description="Apply precomputed refusal tensors to a model")
parser.add_argument(
    "--model",
    "-m",
    type=str,
    required=True,
    help="Your model directory or huggingface model ID",
)
parser.add_argument(
    "--scale-factor",
    type=float,
    default=1.0,
    help="Scale factor for ablation. Use a negative scale-factor to encourage refusal",
)
parser.add_argument(
    "--output",
    "-o",
    type=str,
    required=True,
    help="Output directory for the modified model",
)
args = parser.parse_args()

def load_refusal_dir(file_path: str) -> torch.Tensor:
    return torch.load(file_path)

def modify_tensor(
    tensor_data: torch.Tensor, refusal_dir: torch.Tensor, scale_factor: float = 1.0
) -> torch.nn.Parameter:
    if tensor_data.device != refusal_dir.device:
        refusal_dir = refusal_dir.to(tensor_data.device)
    tensor_float32 = tensor_data.to(torch.float32)
    refusal_dir_float32 = refusal_dir.to(torch.float32)
    tensor_float32 -= scale_factor * torch.matmul(
        torch.outer(refusal_dir_float32, refusal_dir_float32), tensor_float32
    )
    tensor_modified = tensor_float32.to(torch.float16)

    torch.cuda.empty_cache()
    gc.collect()

    return torch.nn.Parameter(tensor_modified)

def apply_saved_tensors(
    model: AutoModelForCausalLM, refusal_dirs: dict, scale_factor: float
) -> AutoModelForCausalLM:
    lm_model = model.model
    assert hasattr(lm_model, "layers"), "The model does not have the expected structure."
    for layer_idx, refusal_dir in refusal_dirs.items():
        lm_model.layers[layer_idx].self_attn.o_proj.weight = modify_tensor(
            lm_model.layers[layer_idx].self_attn.o_proj.weight.data,
            refusal_dir,
            scale_factor,
        )
        lm_model.layers[layer_idx].mlp.down_proj.weight = modify_tensor(
            lm_model.layers[layer_idx].mlp.down_proj.weight.data,
            refusal_dir,
            scale_factor,
        )

    torch.cuda.empty_cache()
    gc.collect()

    return model

if __name__ == "__main__":
    torch.inference_mode()
    torch.set_grad_enabled(False)

    print(f"Loading model {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, low_cpu_mem_usage=True
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("Checking for saved tensors...")
    refusal_dirs = {}
    num_layers = len(model.model.layers)

    for layer_idx in range(1, num_layers):
        tensor_file = f"refusal_tensors/{args.model.replace('/', '_')}_layer_{layer_idx}_refusal_dir.pt"
        if os.path.exists(tensor_file):
            print(f"Loading refusal tensor for layer {layer_idx}...")
            refusal_dirs[layer_idx] = load_refusal_dir(tensor_file)
        else:
            print(f"No refusal tensor found for layer {layer_idx}, skipping...")

    if not refusal_dirs:
        raise ValueError("No saved refusal tensors found. Ensure tensors are precomputed and saved.")

    print("Applying saved tensors to the model...")
    model = apply_saved_tensors(model, refusal_dirs, args.scale_factor)

    print(f"Saving modified model to {args.output}...")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("Model saved successfully.")
