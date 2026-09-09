import os
import shutil
import xml.etree.ElementTree as ET
import torch

from PIL import Image
from .utils import text_normalisation, crop_to_line_image


class Models:

    def __init__(self, image_path="script/data/image.tif", xml_path="script/data/truth.xml"):
        self.image_path = image_path
        try:
            self.image = Image.open(self.image_path)
        except FileNotFoundError as e:
            print(f"Error opening image {self.image_path}: {e}")
            self.image = None
        try:
            with open(xml_path, "r") as f:
                self.xml_path = xml_path
                tree = ET.parse(xml_path)
                root = tree.getroot()

                ALTO_NS = {"alto": "http://www.loc.gov/standards/alto/ns-v2#"}

                def get_textline_text(textline):
                    """Extract text from ALTO TextLine element."""
                    strings = textline.findall("alto:String", ALTO_NS)
                    return " ".join(s.get("CONTENT", "") for s in strings if s.get("CONTENT"))

                def extract_line_coords(root):
                    """Yield data about each text line in the ALTO file."""
                    global_line_index = 0
                    
                    for page in root.findall(".//alto:Page", ALTO_NS):
                        page_id = page.get("ID", "")
                        page_num = page.get("PHYSICAL_IMG_NR", "")

                        for block in page.findall(".//alto:TextBlock", ALTO_NS):
                            block_id = block.get("ID", "")

                            for line in block.findall("alto:TextLine", ALTO_NS):
                                global_line_index += 1
                                yield {
                                    "page_id": page_id,
                                    "page_num": page_num,
                                    "block_id": block_id,
                                    "line_index": global_line_index,
                                    "hpos": int(line.get("HPOS", 0)),
                                    "vpos": int(line.get("VPOS", 0)),
                                    "width": int(line.get("WIDTH", 0)),
                                    "height": int(line.get("HEIGHT", 0)),
                                    "text": get_textline_text(line),
                                }
                
                self.data = []
                for row in extract_line_coords(root):
                    cropped = crop_to_line_image(self.image, row)
                    ref = text_normalisation(row["text"])
                    self.data.append((cropped, ref))


        except FileNotFoundError as e:
                            print(f"Error opening XML {self.xml_path}: {e}")
                            self.xml_path = None

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Some models require a GPU to run.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")
        

    # Selector function
    def inference(self, model, prompt=None):
        prompt = prompt or "Transcribe the text in the image."
        if self.image:
            handlers = {
                "kraken": self.kraken_inference,
                "tesseract": self.tesseract_inference,
                "churro": lambda: self.churro_inference(prompt),
                "qwen": lambda: self.qwen_inference(prompt),
                "deepseek": self.deepseek_inference,
                "trocr": self.trocr_inference,
            }

            handler = handlers.get(model)
            if handler is None:
                print(f"Error: Unknown model '{model}'")
                return [], []

            try:
                outputs = handler()
            except Exception as e:
                print(f"Error during inference with model '{model}': {e}")
                return [], []

            if outputs is None:
                return [], []

            # Normalize model outputs to two lists: references and predictions.
            predictions = []
            references = []
            for prediction, reference in outputs:
                predictions.append(text_normalisation(prediction))
                references.append(text_normalisation(reference))

            return references, predictions
    

    def kraken_inference(self):
        # Kraken is a special case, is it takes files as input, not images in memory. So we have to save the cropped line images to disk first, then run Kraken on them, and finally read the output back into memory.
        if shutil.which("kraken") is None:
            print("Error: Kraken is not installed or not available in PATH.")
            return None

        kraken_outputs = []

        for cropped, ref in self.data:
            cropped.save("temp_line.png")
            cmd = (f"kraken -i 'temp_line.png' 'temp_kraken.txt' ocr -m 'catmus-print-fondue-large.mlmodel' -s")
            os.system(cmd)
    
            try:
                with open("temp_kraken.txt", "r") as f:
                    output = f.read().strip()
                kraken_outputs.append((output, ref))
                os.remove("temp_line.png")
                os.remove("temp_kraken.txt")
            except Exception as e:
                print(f"Error reading Kraken output: {e}")
                return None
        return kraken_outputs
        

    def tesseract_inference(self):
        import pytesseract
        try:
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError as e:
            print(f"Error: Tesseract is not installed or not available in PATH.")
            return None

        tesseract_outputs = []
        for cropped, ref in self.data:
            output = pytesseract.image_to_string(cropped)
            tesseract_outputs.append((output, ref))
        return tesseract_outputs

    def churro_inference(self, prompt=None):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        prompt = prompt or "Transcribe the text in the image."
        
        try: 
            processor = AutoProcessor.from_pretrained("stanford-oval/churro-3B", min_pixels=512 * 28 * 28, max_pixels=5120 * 28 * 28, trust_remote_code=True)
            model = AutoModelForImageTextToText.from_pretrained("stanford-oval/churro-3B", dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
            model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load Churro processor: {e}")

        churro_outputs = []
        for cropped, ref in self.data:
            if hasattr(Image, "Resampling"):
                resample_filter = Image.Resampling.LANCZOS
            else:
                resample_filter = Image.LANCZOS

            # This resizing is optimized for this specific model
            width, height = cropped.size
            if width > 2500 or height > 2500:
                scale = min(2500 / width, 2500 / height)
                new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
                cropped = cropped.resize(new_size, resample=resample_filter)

            conversation = [
                {"role": "system", "content": [{"type": "text", "text": prompt}]},
                {"role": "user", "content": [{"type": "image", "image": cropped}]},
            ]

            chat_prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            encoded = processor(text=[chat_prompt], images=[cropped], return_tensors="pt")
            encoded = {k: v.to(model.device) for k, v in encoded.items() if isinstance(v, torch.Tensor)}

            input_length = encoded["input_ids"].shape[1]
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=512,
                    do_sample=False,
                )

            new_tokens = generated[0, input_length:]
            output = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            churro_outputs.append((output, ref))

        return churro_outputs

    def qwen_inference(self, prompt=None):
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        prompt = prompt or "Transcribe the text in the image."
        
        try:
            processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
            model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-8B-Instruct",dtype=torch.bfloat16,device_map=self.device)
            model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load Qwen model or processor: {e}")

        qwen_outputs = []
        for cropped, ref in self.data:

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": cropped},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

            input_length = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=512, do_sample=False)

            new_tokens = generated[0, input_length:]
            output = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            qwen_outputs.append((output, ref))

        return qwen_outputs


    def deepseek_inference(self, prompt=None):
        print("Unfortunatly, DeepSeek requires some internal changes to the model code for working in this pipeline. Therefor, it can not be part of the testing environment. See readme_deepseek.md to reproduce the results.")
        return []




    def trocr_inference(self):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        try:
            processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
            model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
            model.to(self.device)
            model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load TrOCR model or processor: {e}")

        trocr_outputs = []
        for cropped, ref in self.data:
            if cropped.mode != "RGB":
                cropped = cropped.convert("RGB")

            pixel_values = processor(images=cropped, return_tensors="pt").pixel_values.to(model.device)
            with torch.inference_mode():
                generated = model.generate(pixel_values, max_new_tokens=512)

            output = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
            trocr_outputs.append((output, ref))

        return trocr_outputs


if __name__ == "__main__":
    model = Models()
    print(model.inference("trocr", "Transcribe the text in the image."))
        




   