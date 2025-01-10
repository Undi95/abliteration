from transformers import (
    TextStreamer,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from argparse import ArgumentParser
import torch
import os
from prompt_toolkit import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.containers import HSplit, Window

parser = ArgumentParser()
parser.add_argument(
    "--model", "-m", type=str, required=True, help="Path to model directory"
)
parser.add_argument(
    "--precision",
    "-p", type=str, default="fp16", choices=["fp16", "bf16", "fp32"], help="Precision of model"
)
parser.add_argument(
    "--device", "-d", type=str, choices=["auto", "cuda", "cpu"], default="auto", help="Target device to process abliteration. Warning, bitsandbytes quantization DOES NOT support CPU"
)
parser.add_argument(
    "--max-new-tokens", "-n", type=int, default=256, help="Max new tokens to generate"
)
parser.add_argument(
    "--precise-scale", action="store_true", default=False, help="Launch precise scale adjustment menu"
)
quant = parser.add_mutually_exclusive_group()
quant.add_argument(
    "--load-in-4bit", action="store_true", default=False, help="Load model in 4-bit precision using bitsandbytes"
)
quant.add_argument(
    "--load-in-8bit", action="store_true", default=False, help="Load model in 8-bit precision using bitsandbytes"
)
parser.add_argument(
    "--flash-attn", action="store_true", default=False, help="Use flash attention 2"
)
args = parser.parse_args()

def list_available_tensors(model_name):
    folder = f"refusal_tensors/"
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

def modify_tensor(tensor_data, refusal_dir, scale_factor):
    if tensor_data.device != refusal_dir.device:
        refusal_dir = refusal_dir.to(tensor_data.device)
    tensor_float32 = tensor_data.to(torch.float32)
    refusal_dir_float32 = refusal_dir.to(torch.float32)

    # Ensure the dimensions match for broadcasting
    if refusal_dir_float32.dim() > 1:
        refusal_dir_float32 = refusal_dir_float32.view(-1)
    while refusal_dir_float32.dim() < tensor_float32.dim():
        refusal_dir_float32 = refusal_dir_float32.unsqueeze(-1)

    tensor_float32 -= scale_factor * refusal_dir_float32 * refusal_dir_float32
    return torch.nn.Parameter(tensor_float32.to(tensor_data.dtype))

def reload_model_with_scale(scale_factor, refusal_dirs):
    global model
    global tokenizer
    print("Reloading model with updated scale factor...")
    del model
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=precision,
        low_cpu_mem_usage=True,
        device_map=args.device,
        quantization_config=quant_config,
        attn_implementation="flash_attention_2" if args.flash_attn else None,
    )

    if refusal_dirs:
        print("Applying refusal tensors with scale factor to all layers...")
        for layer_idx, refusal_dir in refusal_dirs.items():
            model.model.layers[layer_idx].self_attn.o_proj.weight = modify_tensor(
                model.model.layers[layer_idx].self_attn.o_proj.weight.data,
                refusal_dir,
                scale_factor,
            )
            model.model.layers[layer_idx].mlp.down_proj.weight = modify_tensor(
                model.model.layers[layer_idx].mlp.down_proj.weight.data,
                refusal_dir,
                scale_factor,
            )
        print("Model updated with scale factor applied.")

def precise_scale_menu(model, refusal_dirs, initial_scales):
    layers = sorted(refusal_dirs.keys())
    num_layers = len(layers)
    selected_layer_idx = 0

    def render_menu():
        os.system('cls' if os.name == 'nt' else 'clear')
        print("Precise Scale Adjustment Menu")
        print("Controls: [←/→] Switch Layers | [↑/↓] Adjust Scale | [Enter] Apply and Launch | [Esc] Exit")
        print("Layer Scales:")

        # Display layers in a horizontal compact format
        row_length = 8
        rows = [layers[i:i + row_length] for i in range(0, num_layers, row_length)]

        for row in rows:
            for layer_idx in row:
                indicator = "->" if layers[selected_layer_idx] == layer_idx else "  "
                print(f"{indicator} L{layer_idx}: {initial_scales[layer_idx]:.1f}", end="\t")
            print()

    bindings = KeyBindings()

    @bindings.add("up")
    def _(event):
        layer_idx = layers[selected_layer_idx]
        initial_scales[layer_idx] = min(2.0, initial_scales[layer_idx] + 0.1)
        render_menu()

    @bindings.add("down")
    def _(event):
        layer_idx = layers[selected_layer_idx]
        initial_scales[layer_idx] = max(-2.0, initial_scales[layer_idx] - 0.1)
        render_menu()

    @bindings.add("left")
    def _(event):
        nonlocal selected_layer_idx
        selected_layer_idx = max(0, selected_layer_idx - 1)
        render_menu()

    @bindings.add("right")
    def _(event):
        nonlocal selected_layer_idx
        selected_layer_idx = min(num_layers - 1, selected_layer_idx + 1)
        render_menu()

    @bindings.add("enter")
    def _(event):
        event.app.exit(result=initial_scales)

    @bindings.add("escape")
    def _(event):
        event.app.exit(result=None)

    layout = Layout(HSplit([Window(content=FormattedTextControl("Adjust scales."))]))
    app = Application(layout=layout, full_screen=False, key_bindings=bindings)
    render_menu()
    result = app.run()

    if result is None:
        print("Exited without applying changes.")
        return None
    return result

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

    refusal_dirs = {}
    scale_factor = 1.0

    if available_tensors:
        print("Detected ablated tensors for the model.")
        for tensor in available_tensors:
            layer_idx = int(tensor.split("_layer_")[1].split("_refusal_dir")[0])
            refusal_dirs[layer_idx] = torch.load(f"refusal_tensors/{tensor}")

        if args.precise_scale:
            initial_scales = {layer_idx: 1.0 for layer_idx in refusal_dirs.keys()}
            precise_scales = precise_scale_menu(model, refusal_dirs, initial_scales)
            if precise_scales is not None:
                reload_model_with_scale(1.0, refusal_dirs)
                for layer_idx, scale_factor in precise_scales.items():
                    model.model.layers[layer_idx].self_attn.o_proj.weight = modify_tensor(
                        model.model.layers[layer_idx].self_attn.o_proj.weight.data,
                        refusal_dirs[layer_idx],
                        scale_factor,
                    )
                    model.model.layers[layer_idx].mlp.down_proj.weight = modify_tensor(
                        model.model.layers[layer_idx].mlp.down_proj.weight.data,
                        refusal_dirs[layer_idx],
                        scale_factor,
                    )
                print("Model reloaded and updated with precise scales.")
        else:
            scale_factor = float(input("Enter a scale factor to apply to all layers (default: 1.0): ") or 1.0)
            reload_model_with_scale(scale_factor, refusal_dirs)
    else:
        print("No ablated tensors detected. Proceeding with a clean model.")

    conversation = []
    streamer = TextStreamer(tokenizer)
    print("Type /clear to clear history, /scale <value> to adjust scale factor, /exit to quit.")

    while True:
        prompt = input("User> ")
        if prompt == "/clear":
            conversation = []
            print("! History cleared.")
            continue
        elif prompt.startswith("/scale"):
            try:
                _, value = prompt.split()
                scale_factor = float(value)
                reload_model_with_scale(scale_factor, refusal_dirs)
            except ValueError:
                print("! Invalid scale factor. Please use: /scale <value>")
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
