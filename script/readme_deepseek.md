# Reproducing the Experiment: DeepSeek-OCR Setup Guide

This guide walks through how to set up DeepSeek-OCR for the batch inference pipeline used in this experiment. The original model code requires several modifications before it can be used programmatically. Follow the steps below in order.

---

## Step 1: Obtain the model code

Download the model from HuggingFace: [deepseek-ai/DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR). The file you need to modify is `modeling_deepseekocr.py`, which you will find in your HuggingFace cache folder after downloading the model.

---

## Step 2: Apply required modifications

The following changes are **required** for the batch inference pipeline to work. Apply them to `modeling_deepseekocr.py`.

### 2.1 Extend `load_image()` to accept PIL Image objects

The original function only accepts file paths. Replace it with the version below, which also handles pre-loaded `PIL.Image` objects. This is necessary because of the change in step 2.2.

```python
def load_image(image):
    if isinstance(image, Image.Image):
        return ImageOps.exif_transpose(image)
    try:
        pil_image = Image.open(image)
        return ImageOps.exif_transpose(pil_image)
    except Exception as e:
        print(f"error: {e}")
        return None
```

### 2.2 Change the `load_pil_images` argument in `infer()`

The original passes the full conversation list to `load_pil_images`. Change it to pass `image_file` directly:

```python
# Original:
images = load_pil_images(conversation)

# Change to:
images = load_pil_images(image_file)
```

### 2.3 Disable output directory creation in `infer()`

Comment out the two `os.makedirs` calls near the top of `infer()`. We do not need an actual output on disk.

```python
#os.makedirs(output_path, exist_ok=True)
#os.makedirs(f'{output_path}/images', exist_ok=True)
```

### 2.4 Strip the `save_results` branch to a pure return

The original `save_results` block writes files, draws bounding boxes, and renders matplotlib charts to disk. Remove all of this and replace the block with a simple return statement:

```python
return outputs
```

This makes `infer()` return the decoded text string directly, so the calling code can handle output collection and storage.

---

## Step 3: Optional modifications

These changes are not strictly required but are recommended for clean batch runs.

### 3.1 Reduce `max_new_tokens`

The default is `8192`, which is suited for full-document conversion. For line-level transcription as used in this experiment, reduce it in both `generate()` calls inside `infer()`:

```python
max_new_tokens=256
```

### 3.2 Remove debug prints from `DeepseekOCRModel.forward()`

The original logs intermediate feature shapes to stdout. Remove the four `print` statements referencing `BASE`, `PATCHES`, and `NO PATCHES` to keep batch output readable.

### 3.3 Silence `NoEOSTextStreamer`

Comment out the `print` in `on_finalized_text` to suppress per-token streaming output:

```python
#print(text, flush=True, end="")
```

### 3.4 Remove `test_compress` diagnostic prints

Remove the print block in the `test_compress` branch that reports image size, token counts, and compression ratio.

---

## Result

After these modifications, `infer()` is a stateless function that accepts an image and a prompt, and returns the decoded transcription as a plain string — with no filesystem writes and no stdout output. It can be called directly in a batch loop.
