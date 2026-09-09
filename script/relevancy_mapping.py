"""
LVLM-style attention visualization for Qwen models.
Computes relevancy per generated token during generation (not after).
Based on: "LVLM-Interpret: An Interpretability Tool for Large Vision-Language Model" (https://arxiv.org/abs/2404.03118)"
Note that this code requires a significant amount of GPU memory (at least 24GB VRAM, preferably more).
Our testing was done on a 128GB DGX Spark, these parameters allow for it to be run on a 24GB GPU. 
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from utils_rel_map import (
    get_qwen_processor_model,
    process_qwen_inputs,
    get_qwen_image_token_indices,
    compute_qwen_image_grid_size,
    reshape_qwen_relevancy_to_grid,
    draw_qwen_heatmap_on_image,
)


def compute_qwen_relevancy_per_token(model, inputs, processor, max_new_tokens=50, layer_start=22, layer_end=24):
    """
    Generate tokens and compute relevancy map for each generated token.
    This follows the LVLM-Interpret approach.
    
    Args:
        model: The Qwen model
        inputs: The processed inputs
        processor: The Qwen processor
        max_new_tokens: Maximum number of tokens to generate
        layer_start: Start layer for attention computation (inclusive)
        layer_end: End layer for attention computation (exclusive)
    """
    device = next(model.parameters()).device
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # Get input tokens
    input_tokens = processor.tokenizer.convert_ids_to_tokens(input_ids[0])
    start_idx, end_idx = get_qwen_image_token_indices(input_tokens)
    num_image_tokens = end_idx - start_idx - 1
    
    generated_ids = []
    generated_tokens = []
    relevancy_maps = {}  # (token, step) -> relevancy map (to handle duplicate tokens)
    
    print(f"Starting generation with relevancy computation...")
    
    for step in tqdm(range(max_new_tokens), desc="Generating tokens"):
        # Forward pass with attention
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=inputs.get('pixel_values'),
            image_grid_thw=inputs.get('image_grid_thw'),
            output_attentions=True,
            return_dict=True
        )
        
        # Retain gradients for attentions
        for attn in outputs.attentions:
            attn.retain_grad()
        
        # Get next token logits
        next_token_logits = outputs.logits[0, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)
        
        # Check for EOS
        if next_token_id == processor.tokenizer.eos_token_id:
            break
        
        # Create one-hot for the selected token
        token_one_hot = torch.nn.functional.one_hot(
            next_token_id, 
            num_classes=next_token_logits.size(-1)
        ).float()
        # Don't add extra dimension - keep shape matching logits
        token_one_hot.requires_grad_(True)
        
        # Backward pass to get gradients
        model.zero_grad()
        next_token_logits.backward(gradient=token_one_hot, retain_graph=True)
        
        # Get attention weights and gradients for this step, keeping pairs aligned
        attentions = outputs.attentions
        paired = [(a, a.grad) for a in attentions if a.grad is not None]

        if not paired:
            # Attentions need to retain gradients
            print(f"Warning: No gradients at step {step}")
            # Store empty relevancy with step number to handle duplicates
            token_str = processor.tokenizer.decode([next_token_id.item()])
            relevancy_maps[(token_str, step)] = np.zeros((num_image_tokens,))
        else:
            # Compute relevancy rollout for this token
            # Use specified layers for stability
            subset = paired[layer_start:min(layer_end, len(paired))]
            att_subset = [a.detach() for a, g in subset]
            grad_subset = [g.detach() for a, g in subset]
            
            # Simple gradient-weighted rollout
            rollout = torch.eye(att_subset[0].size(-1)).to(device)
            for attn, grad in zip(att_subset, grad_subset):
                # Average heads, apply gradient weighting
                attn_avg = attn[0].mean(0)  # [seq, seq]
                grad_avg = grad[0].mean(0)  # [seq, seq]
                
                # Gradient weighting (positive gradients only)
                weighted_attn = attn_avg * grad_avg.clamp(min=0)
                
                # Add residual connection
                weighted_attn = weighted_attn + torch.eye(weighted_attn.size(0)).to(device)
                
                # Normalize
                weighted_attn = weighted_attn / weighted_attn.sum(dim=-1, keepdim=True)
                
                # Accumulate
                rollout = torch.matmul(rollout, weighted_attn)
            
            # Extract relevancy to image tokens from last token
            relevancy = rollout[-1, start_idx+1:end_idx].cpu().numpy()
            relevancy = np.maximum(relevancy, 0)
            
            # Normalize
            if relevancy.max() > 0:
                relevancy = relevancy / relevancy.max()
            
            # Store with step number to handle duplicate tokens
            token_str = processor.tokenizer.decode([next_token_id.item()])
            relevancy_maps[(token_str, step)] = relevancy
        
        # Append token and update for next iteration
        generated_ids.append(next_token_id.item())
        generated_tokens.append(processor.tokenizer.decode([next_token_id.item()]))
        
        # Update input for next step
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)
        attention_mask = torch.cat([
            attention_mask, 
            torch.ones((1, 1), dtype=attention_mask.dtype, device=device)
        ], dim=1)
        
        # Clear memory after processing each token
        del outputs
        del attentions
        del paired
        del next_token_logits
        del token_one_hot
        if 'att_subset' in locals():
            del att_subset
        if 'grad_subset' in locals():
            del grad_subset
        if 'rollout' in locals():
            del rollout
        if 'weighted_attn' in locals():
            del weighted_attn
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    return {
        'generated_tokens': generated_tokens,
        'generated_ids': generated_ids,
        'relevancy_maps': relevancy_maps,
        'vision_start': start_idx,
        'vision_end': end_idx,
        'num_image_tokens': num_image_tokens
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen VL LVLM-style Relevancy Visualization")
    parser.add_argument("--model_name_or_path", type=str, 
                        default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="Qwen model name or path")
    parser.add_argument("--image_path", type=str, default="image.png",
                        help="Path to input image")
    parser.add_argument("--prompt", type=str, default="Transcribe the text and only the text",
                        help="Question prompt")
    parser.add_argument("--target_token", type=str, default=None,
                        help="Specific token to visualize (optional, shows all if not specified)")
    parser.add_argument("--combine_tokens", type=str, default=None,
                        help="Comma-separated list of tokens to combine into one plot (e.g., 'a,dog,.'). Use quotes for tokens with spaces.")
    parser.add_argument("--list_tokens", action="store_true",
                        help="Only list the generated tokens and exit (no visualization)")
    parser.add_argument("--layer_start", type=int, default=22,
                        help="Start layer for relevancy computation (inclusive, default: 22)")
    parser.add_argument("--layer_end", type=int, default=24,
                        help="End layer for relevancy computation (exclusive, default: 24)")
    parser.add_argument("--image_size", type=int, default=350,
                        help="Resize image to this size (square)")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="Maximum tokens to generate")
    parser.add_argument("--output_dir", type=str, default="script/data/rel_map_output",
                        help="Directory to save outputs")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--load_4bit", action="store_true",
                        help="Load model in 4-bit")
    parser.add_argument("--load_8bit", action="store_true",
                        help="Load model in 8-bit")
    parser.add_argument("--device_map", type=str, default="auto",
                        help="Device map for model")
    parser.add_argument("--save_pt", action="store_true",
                        help="Save a .pt file with all relevancy maps")
    return parser.parse_args()


def main():
    args = parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model_name_or_path}")
    processor, model = get_qwen_processor_model(args)
    
    # Load image
    print(f"Loading image: {args.image_path}")
    image = Image.open(args.image_path).convert("RGB")
    image = image.resize((args.image_size, args.image_size))
    
    # Prepare inputs
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": args.prompt},
        ]}
    ]
    
    print("Processing inputs...")
    inputs, text_prompt, input_tokens = process_qwen_inputs(processor, messages, device=args.device)
    
    # Generate with per-token relevancy
    print(f"Using layers {args.layer_start} to {args.layer_end} for relevancy computation")
    results = compute_qwen_relevancy_per_token(
        model, inputs, processor, 
        max_new_tokens=args.max_new_tokens,
        layer_start=args.layer_start,
        layer_end=args.layer_end
    )
    
    generated_text = ''.join(results['generated_tokens'])
    print(f"\nGenerated text: {generated_text}")
    print(f"Generated {len(results['generated_tokens'])} tokens")
    
    # Handle combined tokens option
    if args.combine_tokens:
        # Parse the comma-separated tokens
        tokens_to_combine = [t.strip() for t in args.combine_tokens.split(',')]
        print(f"\nLooking for tokens to combine: {tokens_to_combine}")
        # Extract unique tokens from (token, step) keys
        available_tokens = sorted(set(tok for tok, step in results['relevancy_maps'].keys()))
        print(f"Available tokens: {available_tokens}")
        
        # Find matching tokens in the generated output
        combined_relevancy = None
        matched_tokens = []
        
        for target_token in tokens_to_combine:
            # Find exact matches (try with and without spaces)
            matches = []
            for (tok, step), rel in results['relevancy_maps'].items():
                # Try exact match with stripped version
                if tok.strip() == target_token or tok == target_token:
                    matches.append((tok, step, rel))
                    break
                # Try with space prefix (common for tokens)
                elif tok == f" {target_token}" or tok.strip() == target_token:
                    matches.append((tok, step, rel))
                    break
            
            if matches:
                # Use the first match for each target token
                token, step, relevancy = matches[0]
                matched_tokens.append((token, step))
                
                if combined_relevancy is None:
                    combined_relevancy = relevancy.copy()
                else:
                    # Combine by averaging (you could also use max or sum)
                    combined_relevancy = combined_relevancy + relevancy
                
                print(f"  Found: '{token}' for target '{target_token}'")
            else:
                print(f"  Warning: Token '{target_token}' not found in generated output!")
                print(f"    Hint: Check available tokens above. Tokens may have leading spaces.")
        
        if combined_relevancy is not None and len(matched_tokens) > 0:
            # Normalize the combined relevancy
            combined_relevancy = combined_relevancy / len(matched_tokens)
            if combined_relevancy.max() > 0:
                combined_relevancy = combined_relevancy / combined_relevancy.max()
            
            # Create combined token label
            combined_label = ' + '.join([f"'{tok.strip()}' (step {step})" for tok, step in matched_tokens])
            
            # Compute grid size
            grid_h, grid_w = compute_qwen_image_grid_size(
                results['num_image_tokens'],
                image.height,
                image.width
            )
            
            # Reshape and draw
            heatmap = reshape_qwen_relevancy_to_grid(combined_relevancy, grid_h, grid_w)
            img_with_heatmap = draw_qwen_heatmap_on_image(heatmap, image)
            background_with_heatmap = draw_qwen_heatmap_on_image(
                heatmap, Image.new("RGB", image.size, (255, 255, 255))
            )
            
            # Plot
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            
            axs[0].imshow(image)
            axs[0].set_title("Original Image", fontsize=12)
            axs[0].axis("off")
            
            axs[1].imshow(img_with_heatmap)
            axs[1].set_title(f"Combined Relevance: {combined_label}", fontsize=12)
            axs[1].axis("off")
            
            plt.tight_layout()
            
            # Save
            safe_label = '_'.join([f"{tok.strip().replace('/', '_').replace(' ', '_')}_step{step}" for tok, step in matched_tokens])
            output_path = f"{args.output_dir}/qwen_lvlm_combined_{safe_label}.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"\nSaved combined visualization: {output_path}")
            plt.close()
        else:
            print(f"\nNo tokens found to combine!")
        
        # Early return - don't visualize individual tokens
        if args.save_pt:
            save_path = f"{args.output_dir}/qwen_lvlm_relevancy_maps.pt"
            torch.save(results, save_path)
            print(f"\nSaved all relevancy maps to: {save_path}")
        return
    
    # Find target token or use all
    if args.target_token:
        # Find matching tokens (now keys are (token, step) tuples)
        matches = [((tok, step), rel) for (tok, step), rel in results['relevancy_maps'].items() 
                   if args.target_token.lower() in tok.lower()]
        if not matches:
            print(f"Warning: Token '{args.target_token}' not found in generated output!")
            available_tokens = sorted(set(tok for tok, step in results['relevancy_maps'].keys()))
            print(f"Available tokens: {available_tokens}")
            return
        tokens_to_viz = dict(matches)
    else:
        # Visualize all tokens
        tokens_to_viz = results['relevancy_maps']
    
    # Compute grid size
    grid_h, grid_w = compute_qwen_image_grid_size(
        results['num_image_tokens'],
        image.height,
        image.width
    )
    
    # Visualize each token (keys are now (token, step) tuples)
    for (token, step), relevancy in tokens_to_viz.items():
        # Reshape and draw
        heatmap = reshape_qwen_relevancy_to_grid(relevancy, grid_h, grid_w)
        img_with_heatmap = draw_qwen_heatmap_on_image(heatmap, image)
        background_with_heatmap = draw_qwen_heatmap_on_image(
            heatmap, Image.new("RGB", image.size, (255, 255, 255))
        )

        
        # Plot
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        
        axs[0].imshow(image)
        axs[0].set_title("Original Image", fontsize=12)
        axs[0].axis("off")
        
        axs[1].imshow(img_with_heatmap)
        axs[1].set_title(f"Relevance for '{token.strip()}' (step {step})", fontsize=12)
        axs[1].axis("off")
        
        plt.tight_layout()
        
        # Save with step number to differentiate duplicate tokens
        safe_token = token.strip().replace('/', '_').replace(' ', '_')
        output_path = f"{args.output_dir}/qwen_lvlm_step{step:03d}_{safe_token}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
        plt.close()

    # Save all data
    if args.save_pt:
        save_path = f"{args.output_dir}/qwen_lvlm_relevancy_maps.pt"
        torch.save(results, save_path)
        print(f"\nSaved all relevancy maps to: {save_path}")


if __name__ == "__main__":
    main()
