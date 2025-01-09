# Abliteration

ORIGINAL REPO !!! https://github.com/Orion-zhen/abliteration.git !!!
THIS REPO (Undi95/abliteration) is a FORK.

Make abliterated models using transformers, easy and fast.

The code has been tested on Llama-3.2, Qwen2.5-Coder, Ministral-8b.

Note: abliteration is not uncensorship. Though abliterated, it doesn't necessarily mean the model is completely uncensored, it simply will not explicitly refuse you.

## Usage

**Clone the repositoty**:

```shell
git clone https://github.com/Undi95/abliteration.git
cd abliteration
```

**Install dependencies**:

```shell
pip install -r requirements.txt
```
**Optionnal : install flash-attention**:

```shell
pip install flash-attn --no-build-isolation
```

**Make your abliterations**:

```shell
python abliterate.py -m <path_to_your_model> -o <output_dir> --scan-all
```

Now your model will be abliterated and saved to `<output_dir>`. Once it finishes, you can immediately chat with your abliterated model in the terminal. For Chinese models, you can use `--deccp` to abliterate it from certain topics.

The tensors used are stored in the folder `../refusal_tensors/` where the script is, if you launch the original model with some tensors already saved, you will be able to load them on the fly and test it with precise layers.

```python
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
```

> Qwen series models are so stubborn that you might need to adjust parameters to make a good abliteration.
> You can toy with those too :

```python
lm_model.layers[layer_idx].self_attn.q_proj.weight = modify_tensor(
    lm_model.layers[layer_idx].self_attn.q_proj.weight.data,
    refusal_dir,
    scale_factor,
)
lm_model.layers[layer_idx].mlp.gate_proj.weight = modify_tensor(
    lm_model.layers[layer_idx].mlp.gate_proj.weight.data,
    refusal_dir,
    scale_factor,
)
lm_model.layers[layer_idx].mlp.up_proj.weight = modify_tensor(
    lm_model.layers[layer_idx].mlp.up_proj.weight.data,
    refusal_dir,
    scale_factor,
)
```

Available targets can be found in [transformers model architectures](https://github.com/huggingface/transformers/tree/main/src/transformers/models) and [mergekit model architectures](https://github.com/arcee-ai/mergekit/tree/main/mergekit/_data/architectures).

**Full arguments**:

```shell
usage: abliterate.py [-h] --model MODEL [--device {auto,cuda,cpu}] --output OUTPUT [--skip-begin SKIP_BEGIN] [--skip-end SKIP_END] [--layer-fraction LAYER_FRACTION]
                     [--scale-factor SCALE_FACTOR] [--flash-attn] [--deccp] [--load-in-4bit | --load-in-8bit]

Make abliterated models

options:
  -h, --help            show this help message and exit
  --model MODEL, -m MODEL
                        Your model directory or huggingface model ID
  --device {auto,cuda,cpu}, -d {auto,cuda,cpu}
                        Target device to process abliteration. Warning, bitsandbytes quantization DOES NOT support CPU
  --output OUTPUT, -o OUTPUT
                        Output directory
  --skip-begin SKIP_BEGIN
                        Number of layers to skip at the beginning. Defaults to 1 to avoid messing with the first layer
  --skip-end SKIP_END   Number of layers to skip at the end
  --layer-fraction LAYER_FRACTION
                        Fraction of layers to use for refusal_dir calculation
  --scale-factor SCALE_FACTOR
                        Scale factor for ablation. Use a negative scale-factor to encourage refusal
  --flash-attn          Use flash attention 2
  --deccp               For Chinese models, in specific topics
  --load-in-4bit        Load model in 4-bit precision using bitsandbytes
  --load-in-8bit        Load model in 8-bit precision using bitsandbytes
  --scan-all            Perform calculations for all layers. Cannot be used with --layer or --layer-fraction
  --layer               Perform calculations for a specific layer. Cannot be used with --layer-fraction or --scan-all
  --layer-fraction      Fraction of layers to use for refusal_dir calculation. Cannot be used with --layer or --scan-all
```

## Credits

- [Orion-zhen](https://github.com/Orion-zhen)
- [Sumandora/remove-refusals-with-transformers](https://github.com/Sumandora/remove-refusals-with-transformers)
- [AUGMXNT/deccp](https://github.com/AUGMXNT/deccp)
- [huihui-ai](https://huggingface.co/huihui-ai)
