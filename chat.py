from transformers import (
    TextStreamer,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from argparse import ArgumentParser
import torch
import os

parser = ArgumentParser()
parser.add_argument(
    "--model", "-m", type=str, required=True, help="Path to model directory"
)
parser.add_argument(
    "--precision",
    "-p",
    type=str,
    default="fp16",
    choices=["fp16", "bf16", "fp32"],
    help="Precision of model",
)
parser.add_argument(
    "--device",
    "-d",
    type=str,
    choices=["auto", "cuda", "cpu"],
    default="auto",
    help="Target device to process abliteration. Warning, bitsandbytes quantization DOES NOT support CPU",
)
parser.add_argument(
    "--max-new-tokens", "-n", type=int, default=256, help="Max new tokens to generate"
)
quant = parser.add_mutually_exclusive_group()
quant.add_argument(
    "--load-in-4bit",
    action="store_true",
    default=False,
    help="Load model in 4-bit precision using bitsandbytes",
)
quant.add_argument(
    "--load-in-8bit",
    action="store_true",
    default=False,
    help="Load model in 8-bit precision using bitsandbytes",
)
parser.add_argument(
    "--flash-attn", action="store_true", default=False, help="Use flash attention 2"
)
args = parser.parse_args()

def list_available_tensors(model_name):
    folder = f"refusal_tensors/{model_name.replace('/', '_')}"
    if not os.path.exists(folder):
        return []
    return [f for f in os.listdir(folder) if f.endswith("_refusal_dir.pt")]

def parse_layer_input(input_str, num_layers):
    layers = set()
    try:
        for part in input_str.split(";"):
            if "-" in part:
                start, end = map(int, part.split("-"))
                if start < 1 or end >= num_layers or start > end:
                    raise ValueError
                layers.update(range(start, end + 1))
            else:
                layer = int(part)
                if layer < 1 or layer >= num_layers:
                    raise ValueError
                layers.add(layer)
    except ValueError:
        raise ValueError("Invalid layer or range input.")
    return sorted(layers)

if __name__ == "__main__":
    if args.precision == "fp16":
        precision = torch.float16
    elif args.precision == "bf16":
        precision = torch.bfloat16
    elif args.precision == "fp32":
        precision = torch.float32

    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=precision,
            bnb_4bit_use_double_quant=True,
        )
    elif args.load_in_8bit:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_has_fp16_weight=True,
        )
    else:
        quant_config = None

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=precision,
        low_cpu_mem_usage=True,
        device_map=args.device,
        quantization_config=quant_config,
        attn_implementation="flash_attention_2" if args.flash_attn else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, device_map=args.device
    )

    model_name = args.model.replace("/", "_")
    available_tensors = list_available_tensors(model_name)

    if available_tensors:
        print("Detected ablated tensors for the model.")
        print("Available layers:")
        for tensor in available_tensors:
            print(f"  - {tensor}")

        while True:
            layer_input = input(
                "Enter the layers to load (e.g., '1', '8-18', '4-12;18-20;21-22') or press Enter to skip: "
            )
            if not layer_input:
                print("No ablated tensors loaded. Proceeding with a clean model.")
                break

            try:
                num_layers = len(model.model.layers)
                selected_layers = parse_layer_input(layer_input, num_layers)

                scale_factor = input(
                    "Enter scale factor to apply (default: 1.0): "
                )
                scale_factor = float(scale_factor) if scale_factor else 1.0

                print("Loading and applying tensors...")
                for layer in selected_layers:
                    tensor_file = f"refusal_tensors/{model_name}_layer_{layer}_refusal_dir.pt"
                    if os.path.exists(tensor_file):
                        refusal_dir = torch.load(tensor_file)
                        layer_weights = model.model.layers[layer].self_attn.o_proj.weight
                        layer_weights -= scale_factor * torch.outer(refusal_dir, refusal_dir)
                        print(f"Applied tensor to layer {layer}.")
                    else:
                        print(f"Tensor for layer {layer} not found. Skipping.")
                break
            except ValueError:
                print("Invalid input. Please enter a valid layer or range.")
    else:
        print("No ablated tensors detected. Proceeding with a clean model.")

    conversation = []
    streamer = TextStreamer(tokenizer)
    print("Type /clear to clear history, /exit to quit.")
    while True:
        prompt = input("User> ")
        if prompt == "/clear":
            conversation = []
            print("! History cleared.")
            continue
        elif prompt == "/exit":
            break
        elif prompt == "":
            print("! Please type a message.")
            continue
        conversation.append({"role": "user", "content": prompt})
        toks = tokenizer.apply_chat_template(
            conversation=conversation, add_generation_prompt=True, return_tensors="pt"
        )
        gen = model.generate(
            toks.to(model.device), streamer=streamer, max_new_tokens=args.max_new_tokens
        )
        decoded = tokenizer.batch_decode(
            gen[0][len(toks[0]) :], skip_special_tokens=True
        )
        conversation.append({"role": "assistant", "content": "".join(decoded)})
