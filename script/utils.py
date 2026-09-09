import re 
import unicodedata
import os
import distance
from lxml import etree, html
from apted import APTED, Config
from apted.helpers import Tree
from collections import deque

def text_normalisation(text):
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200B", "")   # zero-width space
    text = text.replace("\u00A0", " ")  # non-breaking space
    text = re.sub(r"[''´`'']", "'", text)      # apostrophes / accents → '
    text = re.sub(r'["]', '', text)            # smart quotes → "
    text = re.sub(r"[‐‑‒–—―−]", "-", text)     # hyphen/dash variants → -
    text = re.sub(r"[…]", "...", text)         # ellipsis normalization
    text = re.sub(r"[·•●]", ".", text)         # bullet points → dot
    text = re.sub(r"[«»]", '"', text)          # guillemets → quotes
    text = re.sub(r"\s+", " ", text)           # normalize whitespace
    text = re.sub(r'-{2,}', '-', text)         # multiple hyphens → single hyphen
    text = re.sub(r'\.(?:\s+\.)+', '.', text)  # long lines of dots
    
    if text.endswith('.'):
        text = text[:-1]
        
    text = text.strip()
    text = text.lower()
    
    if text.startswith('_'):
        text = text[1:].lstrip()
    return text

def crop_to_line_image(image, coords, iskraken=False, output="temp_kraken"):
    x = coords["hpos"]
    y = coords["vpos"]
    w = coords["width"]
    h = coords["height"]

    # Add padding
    padding = 5
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = w + 2 * padding
    h = h + 2 * padding
    
    img_width, img_height = image.size
    x2 = min(x + w, img_width)
    y2 = min(y + h, img_height)

    if iskraken:
        if not os.path.exists(output):
            os.makedirs(output)
        cropped_img = image.crop((x, y, x2, y2))
        cropped_img.save(os.path.join(output, f"{coords['page_id']}_line_{coords['line_index']}.png"))
        return os.path.join(output, f"{coords['page_id']}_line_{coords['line_index']}.png")
    else:
        return image.crop((x, y, x2, y2))
    
import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter, affine_transform, convolve, binary_dilation
import io


def _load_image_input(image):
    if isinstance(image, str):
        with Image.open(image) as pil_image:
            return pil_image.copy()
    if isinstance(image, Image.Image):
        return image.copy()
    return image


def _as_uint8_array(image):
    image = _load_image_input(image)

    if isinstance(image, Image.Image):
        image = np.array(image)

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array, PIL image, or image path")

    if image.dtype == np.uint8:
        return image

    if np.issubdtype(image.dtype, np.floating):
        max_value = float(np.nanmax(image)) if image.size else 0.0
        if max_value <= 1.0:
            image = image * 255.0
        return np.clip(image, 0, 255).astype(np.uint8)

    if np.issubdtype(image.dtype, np.integer):
        max_value = int(image.max()) if image.size else 0
        if max_value <= 255:
            return image.astype(np.uint8)
        scale = 255.0 / max_value if max_value else 1.0
        return np.clip(image.astype(np.float32) * scale, 0, 255).astype(np.uint8)

    return np.clip(image, 0, 255).astype(np.uint8)

# This function is responsible for the degradation of the image

def degrade_image(image, noise_sigma=100, blur_sigma=0.9, dark_spot_coverage=0.95, 
                  bright_spot_coverage=1, distortion_strength=0.000):
    input_was_pil_like = isinstance(image, (str, Image.Image))
    image = _load_image_input(image)

    if isinstance(image, Image.Image):
        image = np.array(image)

    image = _as_uint8_array(image)

    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB) if input_was_pil_like else cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if input_was_pil_like else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim != 2:
        raise ValueError("degrade_image expects a 2D grayscale image or a 3-channel image")
    
    # Gaussian noise
    noise = np.random.normal(0, noise_sigma, image.shape)
    noisy = image.astype(np.float32) + noise

    h, w = image.shape[:2]
    
    # Dark spots (simulate paper tears) - irregular, sharp-edged
    rand_field_dark = np.random.rand(h, w)
    blobs_dark = gaussian_filter(rand_field_dark, sigma=40)  # Smaller sigma for more defined spots
    mask_dark = blobs_dark > np.quantile(blobs_dark, dark_spot_coverage)
    
    # Create irregular tear-like shapes by erosion/dilation
    kernel_tear = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=np.uint8)

    for _ in range(np.random.randint(1, 3)):
        mask_dark = binary_dilation(mask_dark, structure=kernel_tear)
    
    tear_noise = np.random.rand(h, w) > 0.7
    mask_dark = mask_dark & tear_noise
    
    # Bright spots (hide letters by making them white)
    rand_field_bright = np.random.rand(h, w)
    blobs_bright = gaussian_filter(rand_field_bright, sigma=60)
    mask_bright = blobs_bright > np.quantile(blobs_bright, bright_spot_coverage)
    
    # Apply varying intensity to spots
    # Tears are very dark (like holes)
    dark_intensity = np.random.uniform(-200, -150, mask_dark.shape)
    spots_dark = mask_dark * dark_intensity
    bright_intensity = np.random.uniform(80, 150, mask_bright.shape)
    spots_bright = mask_bright * bright_intensity
    spots = spots_dark + spots_bright

    noisy = noisy + spots

    # Blur (simulate ink diffusion)
    blurred = gaussian_filter(noisy, sigma=blur_sigma)

    # Geometric distortion
    matrix = [[1, distortion_strength], [distortion_strength / 5, 1]]
    warped = affine_transform(blurred, matrix)

    # Ink bleed kernel
    kernel = np.array([[0.02, 0.05, 0.02],
                       [0.05, 1.0, 0.05],
                       [0.02, 0.05, 0.02]])

    bleed = convolve(warped, kernel)
    
    result = np.clip(bleed, 0, 255).astype(np.uint8)
    
    result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))


# This function is responsible for compressing the image as if it was a jpeg

def apply_jpeg_compression(image, jpeg_quality=95, jpeg_passes=1, chroma_subsampling=2,
                           add_blockiness=True, block_scale=0.24):
    input_was_pil_like = isinstance(image, (str, Image.Image))
    image = _load_image_input(image)

    # Convert to PIL
    if isinstance(image, Image.Image):
        img = image.copy()
    elif isinstance(image, np.ndarray) and image.ndim == 2:
        img = Image.fromarray(image)
    else:
        image = _as_uint8_array(image)
        if image.ndim == 2:
            img = Image.fromarray(image)
        else:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB) if input_was_pil_like else cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            img = Image.fromarray(image if input_was_pil_like else cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if img.mode == "I;16":
        img = img.point(lambda value: value * (255.0 / 65535.0)).convert("L")
    elif img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # Optional macroblock exaggeration
    if add_blockiness:
        w, h = img.size
        sw = max(1, int(w * block_scale))
        sh = max(1, int(h * block_scale))
        img = img.resize((sw, sh), Image.NEAREST)
        img = img.resize((w, h), Image.NEAREST)

    #JPEG recompression
    for _ in range(max(1, jpeg_passes)):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(jpeg_quality),
                 subsampling=int(chroma_subsampling), optimize=False, progressive=False)
        buf.seek(0)
        img = Image.open(buf)
        img.load()

    return img

# TEDS: Tree Edit Distance-based Similarity (IBM / PubTables-1M)
# Source adapted from: https://github.com/ibm-aur-nlp/Table-Structure-Recognition

class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)


class CustomConfig(Config):
    @staticmethod
    def maximum(*sequences):
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        return float(distance.levenshtein(*sequences)) / self.maximum(*sequences)

    def rename(self, node1, node2):
        if (node1.tag != node2.tag) or (node1.colspan != node2.colspan) or (node1.rowspan != node2.rowspan):
            return 1.0
        if node1.tag == "td":
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.0


class TEDS:
    """Tree Edit Distance-based Similarity metric."""
    def __init__(self, structure_only=False, n_jobs=1, ignore_nodes=None):
        self.structure_only = structure_only
        self.n_jobs = n_jobs
        self.ignore_nodes = ignore_nodes
        self.__tokens__ = []

    def tokenize(self, node):
        self.__tokens__.append(f"<{node.tag}>")
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in node.getchildren():
            self.tokenize(n)
        if node.tag != "unk":
            self.__tokens__.append(f"</{node.tag}>")
        if node.tag != "td" and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        if node.tag == "td":
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            new_node = TableTree(
                node.tag,
                int(node.attrib.get("colspan", "1")),
                int(node.attrib.get("rowspan", "1")),
                cell,
                *deque(),
            )
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != "td":
            for n in node.getchildren():
                self.load_html_tree(n, new_node)
        if parent is None:
            return new_node

    def evaluate(self, pred_html, true_html):
        """Compute TEDS score between predicted and ground truth HTML strings."""
        if not pred_html or not true_html:
            return 0.0

        parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
        pred = html.fromstring(pred_html, parser=parser)
        true = html.fromstring(true_html, parser=parser)

        pred = pred.xpath("//table")[0]
        true = true.xpath("//table")[0]

        if self.ignore_nodes:
            etree.strip_tags(pred, *self.ignore_nodes)
            etree.strip_tags(true, *self.ignore_nodes)

        n_nodes = max(len(pred.xpath(".//*")), len(true.xpath(".//*")))
        tree_pred = self.load_html_tree(pred)
        tree_true = self.load_html_tree(true)

        dist = APTED(tree_pred, tree_true, CustomConfig()).compute_edit_distance()
        return 1.0 - float(dist) / n_nodes


def teds_score(pred_path, gt_path, structure_only=False):
    """Compute TEDS score between two HTML table files.

    Args:
        pred_path: path to predicted HTML file
        gt_path: path to ground truth HTML file
        structure_only: if True, ignore cell content and compare structure only

    Returns:
        float between 0 and 1
    """
    pred_html = open(pred_path, encoding="utf-8").read()
    gt_html = open(gt_path, encoding="utf-8").read()
    return TEDS(structure_only=structure_only).evaluate(pred_html, gt_html)
