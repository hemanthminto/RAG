import json
import re
import os
import uuid
import traceback
import base64
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from bs4 import BeautifulSoup
from unstructured.partition.pdf import partition_pdf
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
def extract_and_store_pdf_elements(pdf_path: str, output_json_path: str, figures_dir: str = "figures"):
    print(f"Processing PDF: {pdf_path}")
    os.makedirs(figures_dir, exist_ok=True)
    result = {
        "text_elements": [],
        "table_elements": [],
        "image_elements": [],
        "formula_elements": [],
        "url_elements": [],
        "other_elements": []
    }
    print("Loading image captioning model...")
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    max_length = 20
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    formula_patterns = [
        r'\$.*?\$',
        r'\\\(.*?\\\)',
        r'\\\[.*?\\\]',
        r'\\begin\{equation\}.*?\\end\{equation\}',
        r'\\frac\{.*?\}\{.*?\}',
        r'\\sum',
        r'\\int',
        r'\\alpha|\\beta|\\gamma|\\delta|\\epsilon',
        r'\\sqrt\{.*?\}',
        r'\\mathbb\{.*?\}',
        r'\\mathcal\{.*?\}',
        r'\\mathrm\{.*?\}',
        r'\\partial',
        r'\\nabla',
        r'\\infty',
        r'\\lim',
    ]
    formula_pattern = re.compile('|'.join(formula_patterns))
    def predict_caption(image):
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
            pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)
            output_ids = model.generate(pixel_values, **gen_kwargs)
            preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            return preds[0].strip()
        except Exception as e:
            print(f"Error in predict_caption: {str(e)}")
            return "Unable to generate caption due to an error."
    def save_image_from_bytes(image_bytes, prefix="img", format="png"):
        try:
            image = Image.open(BytesIO(image_bytes))
            filename = f"{prefix}_{uuid.uuid4().hex}.{format}"
            filepath = os.path.join(figures_dir, filename)
            image.save(filepath)
            return filepath, image
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            return None, None
    def render_formula_with_matplotlib(formula_text, formula_id):
        try:
            if formula_text.startswith('$') and formula_text.endswith('$'):
                formula_text = formula_text[1:-1]
            elif formula_text.startswith('\\(') and formula_text.endswith('\\)'):
                formula_text = formula_text[2:-2]
            elif formula_text.startswith('\\[') and formula_text.endswith('\\]'):
                formula_text = formula_text[2:-2]
            fig = plt.figure(figsize=(12, 3), dpi=300)
            fig.patch.set_facecolor('white')
            plt.text(0.5, 0.5, f"${formula_text}$", 
                     fontsize=20, 
                     ha='center', va='center', 
                     transform=fig.transFigure)
            plt.axis('off')
            filename = f"formula_{formula_id}.png"
            filepath = os.path.join(figures_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close(fig)
            return filepath
        except Exception as e:
            print(f"Error rendering formula with matplotlib: {str(e)}")
            return render_formula_fallback(formula_text, formula_id)
    def render_formula_fallback(formula_text, formula_id):
        try:
            width, height = 1500, 300
            image = Image.new('RGB', (width, height), color='white')
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 36)
            except:
                try:
                    font = ImageFont.truetype("Arial.ttf", 36)
                except:
                    font = ImageFont.load_default()
            draw = ImageDraw.Draw(image)
            text_width = draw.textlength(formula_text, font=font)
            x = (width - text_width) / 2
            draw.text((x, height/2 - 18), formula_text, fill='black', font=font)
            filename = f"formula_fallback_{formula_id}.png"
            filepath = os.path.join(figures_dir, filename)
            image.save(filepath, quality=100)
            return filepath
        except Exception as e:
            print(f"Error in formula fallback rendering: {str(e)}")
            return None
    def save_and_caption_table(table_element, table_data=None, image_data=None):
        table_element_with_paths = table_element.copy()
        if image_data:
            try:
                image_bytes = base64.b64decode(image_data)
                table_image_path, _ = save_image_from_bytes(image_bytes, "table", "png")
                if table_image_path:
                    table_element_with_paths["table_image_path"] = table_image_path
                    print(f"Saved table image to {table_image_path}")
            except Exception as e:
                print(f"Error saving table image: {str(e)}")
        if table_data:
            try:
                table_data_path = os.path.join(figures_dir, f"table_data_{uuid.uuid4().hex}.json")
                with open(table_data_path, 'w', encoding='utf-8') as f:
                    json.dump(table_data, f, indent=2)
                table_element_with_paths["table_data_path"] = table_data_path
                print(f"Saved table data to {table_data_path}")
            except Exception as e:
                print(f"Error saving table data: {str(e)}")
        return table_element_with_paths
    def fetch_url_content(url):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                for script in soup(["script", "style"]):
                    script.extract()
                text = soup.get_text(separator=' ', strip=True)
                return text[:5000]  
            return f"Failed to fetch content: Status code {response.status_code}"
        except Exception as e:
            return f"Error fetching URL content: {str(e)}"
    print("Extracting elements from PDF...")
    try:
        elements = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_images_in_pdf=True,
        )
        print(f"Successfully extracted {len(elements)} elements")
    except Exception as e:
        print(f"Error extracting PDF elements: {str(e)}")
        traceback.print_exc()
        elements = []
    image_count = sum(1 for element in elements if type(element).__name__ == "Image")
    table_count = sum(1 for element in elements if type(element).__name__ == "Table")
    print(f"Found {image_count} image elements and {table_count} table elements in PDF")
    for element in elements:
        element_type = type(element).__name__
        print(f"Processing element of type: {element_type}")
        metadata = {}
        image_data = None
        if hasattr(element, "metadata"):
            try:
                metadata = element.metadata.to_dict()
                print(f"Metadata keys: {list(metadata.keys())}")
                if "coordinates" in metadata:
                    del metadata["coordinates"]
                if "last_modified" in metadata:
                    del metadata["last_modified"]
                if "image_base64" in metadata:
                    image_data = metadata["image_base64"]
                    print("Found image_base64 in metadata")
            except Exception as e:
                print(f"Error processing metadata: {str(e)}")
                metadata = {}
        element_dict = {
            "element_id": element.id if hasattr(element, "id") else str(uuid.uuid4()),
            "element_type": element_type,
            "text": str(element),
            "metadata": metadata
        }
        text_content = str(element)
        urls = url_pattern.findall(text_content)
        formula_matches = formula_pattern.findall(text_content)
        has_formula = len(formula_matches) > 0
        if not has_formula:
            has_formula = (
                ('=' in text_content and any(c in text_content for c in '+-*/^')) or
                (text_content.count('_') > 1 and any(c.isalpha() for c in text_content)) or
                (text_content.count('^') > 0 and any(c.isalpha() for c in text_content)) or
                ('∫' in text_content or '∑' in text_content or '∏' in text_content) or
                ('lim' in text_content and '→' in text_content)
            )
        if urls:
            for url in urls:
                url_element = element_dict.copy()
                url_element["element_type"] = "url"
                url_element["original_text"] = text_content
                url_element["url"] = url
                url_content = fetch_url_content(url)
                url_element["text"] = url_content
                result["url_elements"].append(url_element)
        if element_type == "Table":
            print("Processing table element...")
            table_data = None
            if hasattr(element, "value") and element.value is not None:
                try:
                    if hasattr(element.value, "to_dict"):
                        table_data = element.value.to_dict()
                    elif hasattr(element.value, "to_list"):
                        table_data = element.value.to_list()
                    else:
                        table_data = str(element.value)
                except Exception as e:
                    print(f"Error extracting table data: {str(e)}")
            enhanced_table_element = save_and_caption_table(
                element_dict, 
                table_data=table_data, 
                image_data=image_data
            )
            result["table_elements"].append(enhanced_table_element)
            print("Finished processing table element")
        elif element_type == "Image":
            print("Processing image element...")
            image_element = element_dict.copy()
            image_obj = None
            if image_data:
                try:
                    print("Decoding image from base64...")
                    image_bytes = base64.b64decode(image_data)
                    image_path, image_obj = save_image_from_bytes(image_bytes, "image")
                    if image_path:
                        image_element["image_path"] = image_path
                        print(f"Saved image to {image_path}")
                except Exception as e:
                    print(f"Error processing image_base64: {str(e)}")
            if not image_obj and hasattr(element, "image"):
                try:
                    print("Getting image from element.image...")
                    image_obj = element.image
                    image_path = os.path.join(figures_dir, f"image_{uuid.uuid4().hex}.png")
                    image_obj.save(image_path)
                    image_element["image_path"] = image_path
                    print(f"Saved image to {image_path}")
                except Exception as e:
                    print(f"Error saving image from element.image: {str(e)}")
            if not image_obj and "image_path" in metadata:
                try:
                    print(f"Using image path from metadata: {metadata['image_path']}")
                    image_path = metadata["image_path"]
                    if os.path.exists(image_path):
                        image_obj = Image.open(image_path)
                        image_element["image_path"] = image_path
                except Exception as e:
                    print(f"Error loading image from metadata path: {str(e)}")
            if image_obj:
                try:
                    print("Generating caption for image...")
                    caption = predict_caption(image_obj)
                    if caption:
                        formatted_caption = f"{caption}"
                        image_element["caption"] = formatted_caption
                        image_element["text"] = formatted_caption
                        print(f"Generated caption: {formatted_caption}")
                    else:
                        default_caption = "Image contains visual content that could not be described"
                        image_element["caption"] = default_caption
                        image_element["text"] = default_caption
                except Exception as e:
                    print(f"Error generating caption: {str(e)}")
                    image_element["caption"] = "Error generating caption"
                    image_element["text"] = "Error generating caption"
            else:
                print("WARNING: No valid image data found for captioning")
                default_caption = "This element was identified as an image but no image data could be processed"
                image_element["caption"] = default_caption
                image_element["text"] = default_caption  
            result["image_elements"].append(image_element)
            print("Finished processing image element")
        elif element_type == "Formula" or has_formula:
            print("Processing formula element...")
            if formula_matches:
                for i, formula in enumerate(formula_matches):
                    formula_element = element_dict.copy()
                    formula_element["text"] = formula
                    formula_element["original_text"] = text_content
                    formula_id = f"{uuid.uuid4().hex}_{i}"
                    formula_path = render_formula_with_matplotlib(formula, formula_id)
                    if formula_path:
                        formula_element["formula_path"] = formula_path
                        print(f"Created formula image at {formula_path}")
                    result["formula_elements"].append(formula_element)
            else:
                formula_id = uuid.uuid4().hex
                formula_path = render_formula_with_matplotlib(text_content, formula_id)
                if formula_path:
                    element_dict["formula_path"] = formula_path
                    print(f"Created formula image at {formula_path}")
                result["formula_elements"].append(element_dict)
            print("Finished processing formula element")
        elif element_type in ["NarrativeText", "Title", "Text", "ListItem"]:
            if not has_formula and not urls:
                result["text_elements"].append(element_dict)
        else:
            result["other_elements"].append(element_dict)
    print("Post-processing all image elements...")
    for i, image_element in enumerate(result["image_elements"]):
        if "caption" not in image_element or not image_element["caption"] or image_element["caption"].startswith("No "):
            if "image_path" in image_element:
                try:
                    image_path = image_element["image_path"]
                    if os.path.exists(image_path):
                        print(f"Retrying caption generation for image {i} from path {image_path}")
                        image = Image.open(image_path)
                        caption = predict_caption(image)
                        if caption:
                            formatted_caption = f"{caption}"
                            image_element["caption"] = formatted_caption
                            image_element["text"] = formatted_caption
                            print(f"Generated caption on retry: {formatted_caption}")
                except Exception as e:
                    print(f"Error in post-processing image caption: {str(e)}")
    captioned_images = sum(1 for img in result["image_elements"] if "caption" in img and not img["caption"].startswith("No "))
    print(f"Successfully captioned {captioned_images} out of {len(result['image_elements'])} images")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Extraction complete. Results saved to: {output_json_path}")
    print(f"Summary:")
    print(f"- Text elements: {len(result['text_elements'])}")
    print(f"- Table elements: {len(result['table_elements'])}")
    print(f"- Image elements: {len(result['image_elements'])}")
    print(f"- Formula elements: {len(result['formula_elements'])}")
    print(f"- URL elements: {len(result['url_elements'])}")
    print(f"- Other elements: {len(result['other_elements'])}")
    print(f"- Figures saved to: {figures_dir}")
if __name__ == "__main__":
    pdf_file_path = "/Users/hemanthreddy/Desktop/Mixed_data/formula/allmix.pdf"
    json_output_path = "update_final.json"
    extract_and_store_pdf_elements(pdf_file_path, json_output_path)
