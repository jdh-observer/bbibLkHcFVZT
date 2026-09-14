# Easy as 1-2-3? Vision-Language Models for Historical Text Recognition

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jdh-observer/bbibLkHcFVZT/main?filepath=article.ipynb)


## Abstract
This article examines how vision–language models (VLMs) – AI systems that process both images and text – can be used for historical text recognition. We test these models on company records collected from the Belgian Official Gazette (1873-1991), collected as part of the BelHisFirm project. While VLMs recognize text more accurately than traditional systems, historians lack clear standards for using them. Researchers need to understand how these models generate text, yet practices for selecting models, designing prompts, understanding and validating results vary widely. We address this gap by comparing tree VLMs (DeepSeek-OCR, Qwen-VL, CHURRO) with two traditional systems (Tesseract, Kraken) and one Vision Transformer (TrOCR). By visualising model attention and token probabilities, we examine how visual input and prompt design influence VLM behavior and error patterns. The comparison reveals a critical trade-off: VLMs excel at reading poor-quality prints and irregular layouts, but they produce “hallucinations”, text that looks plausible but doesn’t match the original document. Traditional systems make predictable character-level errors, which can be corrected using dictionaries or rules. In contrast, VLM hallucinations are often hard to detect and filter. We argue that incorporating VLMs into historical text recognition requires adapting traditional ATR workflows to address these challenges.

## Keywords
Vision-language models (VLMs), Historical text recognition, Belgian Official Gazette, Economic history, AI hallucinations

## Python version
3.13.7

