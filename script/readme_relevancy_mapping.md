# LVLM-style Attention Visualization for Qwen VL

Visualizes which image regions a Qwen vision-language model attends to when generating each output token. Based on the [LVLM-Interpret](https://github.com/YinanHuang/lvlm-interpret) approach: gradient-weighted attention rollout computed during generation, one heatmap per token.

## Installation

```bash
pip install -r requirements.txt
pip install qwen-vl-utils
```

The model must be loaded with `attn_implementation="eager"` (handled automatically) — flash-attention does not expose the attention weights needed for this method.

## Basic usage

```bash
python run_qwen_lvlm_style.py \
    --image_path image.png \
    --prompt "Transcribe the text and only the text"
```

Output PNGs are saved to `script/data/rel_map_output/` by default, one per generated token (`qwen_lvlm_step000_Hello.png`, etc.). The output directory is created automatically if it does not exist.

## Key options

| Flag | Default | Description |
|---|---|---|
| `--model_name_or_path` | `Qwen/Qwen2.5-VL-3B-Instruct` | Model to use |
| `--image_path` | `image.png` | Input image |
| `--prompt` | `"Transcribe the text and only the text"` | Text prompt |
| `--max_new_tokens` | `50` | Max tokens to generate |
| `--layer_start` / `--layer_end` | `22` / `24` | Layer range for attention rollout (exclusive end) |
| `--image_size` | `350` | Resize image to this square size before processing |
| `--output_dir` | `./lvlm_output` | Where to save output images |
| `--load_4bit` / `--load_8bit` | off | Quantized model loading (reduces VRAM) |
| `--save_pt` | off | Save a `.pt` file with all relevancy maps |

## Selecting which tokens to visualize

**Single token** — show only heatmaps matching a substring:
```bash
python run_qwen_lvlm_style.py --image_path image.png --target_token "1847"
```

**Combined tokens** — average relevancy across multiple tokens into one plot:
```bash
python run_qwen_lvlm_style.py --image_path image.png --combine_tokens "January,1847"
```

**List tokens only** — print generated tokens without producing any images:
```bash
python run_qwen_lvlm_style.py --image_path image.png --list_tokens
```
Use this first to see what tokens are available before selecting specific ones.

## Notes

- Each token requires a full forward + backward pass, so generation is slow (expect several seconds per token on a single GPU).
- `pixel_values` are re-encoded on every step because KV caching is incompatible with the per-step backward pass.
- Relevancy maps can optionally be saved to a `.pt` file with `--save_pt` and reloaded with `torch.load()` for further analysis.
