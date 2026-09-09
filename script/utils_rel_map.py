"""
Relevancy mapping utilities for Qwen LVLM models.
Combines model loading/input processing with relevancy extraction and visualization.
"""
import logging
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import BitsAndBytesConfig

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    logging.getLogger(__name__).warning(
        "qwen_vl_utils not found. Please install: pip install qwen-vl-utils"
    )
    process_vision_info = None

logger = logging.getLogger(__name__)
cmap = plt.get_cmap('jet')


def _enable_qwen_gradients(model):
    """Enable gradients on _sample so attention gradients can be captured."""
    func = '_sample'
    if hasattr(model, func):
        setattr(
            model.__class__,
            func,
            torch.enable_grad()(getattr(model.__class__, func))
        )
    return model


def get_qwen_processor_model(args):
    """
    Load Qwen VL processor and model.

    Args:
        args: Namespace with model_name_or_path, load_4bit, load_8bit, device_map

    Returns:
        tuple: (processor, model)
    """
    model_name = args.model_name_or_path

    if "qwen" not in model_name.lower():
        logger.warning(f"Model {model_name} may not be a Qwen model")

    processor = AutoProcessor.from_pretrained(model_name)

    if args.load_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
    elif args.load_8bit:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quant_config = None

    torch_dtype = (
        torch.float16
        if (quant_config or torch.cuda.is_available())
        else torch.float32
    )

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        device_map=args.device_map,
        torch_dtype=torch_dtype,
        quantization_config=quant_config,
        low_cpu_mem_usage=True,
        attn_implementation="eager"  # Required for attention extraction
    )

    model = _enable_qwen_gradients(model)

    logger.info(f"Loaded Qwen model: {model_name}")
    logger.info(f"Model dtype: {torch_dtype}, quantization: {quant_config is not None}")

    return processor, model


def process_qwen_inputs(processor, messages, device="cuda"):
    """
    Process inputs for Qwen VL model.

    Args:
        processor: Qwen processor
        messages: List of message dicts:
            [{"role": "user", "content": [
                {"type": "image", "image": PIL_image},
                {"type": "text", "text": "question"}
            ]}]
        device: Device to move tensors to

    Returns:
        tuple: (inputs, text_prompt, tokens)
    """
    if process_vision_info is None:
        raise ImportError("qwen_vl_utils is required. Install with: pip install qwen-vl-utils")

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)

    tokens = processor.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    return inputs, text, tokens


def get_qwen_image_token_indices(tokens):
    """
    Find start and end indices of image tokens in Qwen tokenization.

    Args:
        tokens: List of tokens from processor.tokenizer.convert_ids_to_tokens()

    Returns:
        tuple: (start_idx, end_idx)
    """
    try:
        start = tokens.index("<|vision_start|>")
        end = tokens.index("<|vision_end|>")
        return start, end
    except ValueError as e:
        logger.error(f"Could not find vision tokens: {e}")
        logger.debug(f"Available tokens: {tokens}")
        raise


def compute_qwen_image_grid_size(num_tokens, image_height, image_width):
    """
    Compute vision-token grid size based on image aspect ratio.

    Args:
        num_tokens: Number of image tokens
        image_height: Original image height
        image_width: Original image width

    Returns:
        tuple: (grid_h, grid_w)
    """
    aspect_ratio = image_height / image_width
    grid_h = int(math.sqrt(num_tokens * aspect_ratio))
    grid_w = math.ceil(num_tokens / grid_h)
    return grid_h, grid_w



def reshape_qwen_relevancy_to_grid(relevancy_scores, grid_h, grid_w):
    """
    Reshape flat relevancy scores into a 2D grid matching the image layout.

    Args:
        relevancy_scores: 1D array of relevancy scores
        grid_h: Grid height
        grid_w: Grid width

    Returns:
        np.ndarray: 2D relevancy grid
    """
    num_tokens = len(relevancy_scores)
    target_size = grid_h * grid_w

    if target_size > num_tokens:
        pad_arr = np.zeros(target_size - num_tokens, dtype=np.float32)
        relevancy_scores = np.concatenate([relevancy_scores, pad_arr])
    elif target_size < num_tokens:
        relevancy_scores = relevancy_scores[:target_size]

    return relevancy_scores.reshape(grid_h, grid_w)


def draw_qwen_heatmap_on_image(relevancy_grid, image, normalize=True, sigma=1.2, alpha_scale=1.8):
    """
    Overlay a relevancy heatmap on the original image.

    Args:
        relevancy_grid: 2D array of relevancy scores
        image: PIL Image
        normalize: Whether to min-max normalise relevancy values
        sigma: Gaussian smoothing sigma
        alpha_scale: Scale factor for alpha channel transparency

    Returns:
        PIL.Image: Image with relevancy heatmap overlay
    """
    if isinstance(relevancy_grid, torch.Tensor):
        relevancy_grid = relevancy_grid.cpu().numpy()

    heatmap = gaussian_filter(relevancy_grid.astype(np.float32), sigma=sigma)

    heatmap_tensor = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    heatmap_up = F.interpolate(
        heatmap_tensor,
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False
    ).squeeze().cpu().numpy()

    if normalize:
        heatmap_up = (heatmap_up - heatmap_up.min()) / (heatmap_up.max() - heatmap_up.min() + 1e-8)

    alpha_map = np.clip(heatmap_up * alpha_scale, 0, 1)

    heatmap_colored = cmap(np.clip(heatmap_up, 0, 1))
    heatmap_rgba = (heatmap_colored * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap_rgba, mode='RGBA')

    alpha_channel = (alpha_map * 255).astype(np.uint8)
    heatmap_arr = np.array(heatmap_img)
    heatmap_arr[:, :, 3] = alpha_channel
    heatmap_img = Image.fromarray(heatmap_arr, mode='RGBA')

    img_overlay = image.copy()
    img_overlay.paste(heatmap_img, (0, 0), heatmap_img)

    return img_overlay
