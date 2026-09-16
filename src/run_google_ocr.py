import io
import os
import json
import re
import argparse  
from google.cloud import vision

def clean_text(text):
    """Regex Cleaning function"""
    if not text: return ""
    text = text.replace('\u200b', '').replace('\ufeff', '')
    text = re.sub(r'[|¦_—]+', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def extract_number(filename):
    """Helper function to sort files by page number (e.g., page_1, page_2, page_10)"""
    s = re.findall(r'\d+', filename)
    return int(s[0]) if s else 0

def process_cleaned_images_to_json(input_dir, output_path, config_path):
    print("Starting Cleaned Images -> JSON OCR process...")
    
    # ==========================================
    # 1. Read values from config.json
    # ==========================================
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    # Extract settings into variables
    cert_path = config["google_cloud"]["cert_path"]
    header_margin_ratio = config["ocr_settings"]["header_margin_ratio"]
    footer_margin_ratio = config["ocr_settings"]["footer_margin_ratio"]
    min_confidence = config["ocr_settings"]["min_confidence"]

    # ==========================================
    # 2. Configure Google Cloud Certificate
    # ==========================================
    if os.path.exists(cert_path):
        print(f"Found custom certificate at: {cert_path}")
        os.environ['SSL_CERT_FILE'] = cert_path
        os.environ['REQUESTS_CA_BUNDLE'] = cert_path
        os.environ['GRPC_DEFAULT_SSL_ROOTS_FILE_PATH'] = cert_path
    else:
        print(f"Warning: Certificate file not found at {cert_path}")

    # ==========================================
    # 3. Initialize Google Client
    # ==========================================
    try:
        client = vision.ImageAnnotatorClient()
        print("Google Client Initialized")
    except Exception as e:
        print(f"Error initializing Google Client: {e}")
        return

    # ==========================================
    # 4. Start processing images
    # ==========================================
    print(f"Reading images from: {input_dir}")
    if not os.path.exists(input_dir):
        print(f"Directory not found: {input_dir}")
        return

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort(key=extract_number)

    if not image_files:
        print("Warning: No image files found in this directory — saving empty OCR JSON")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([], f)
        return

    print(f"Found {len(image_files)} image pages in total")
    final_result = []

    for i, filename in enumerate(image_files):
        file_page_num = extract_number(filename)
        page_id = file_page_num if file_page_num > 0 else i + 1
        
        print(f"   ...Processing: {filename} (Page ID: {page_id})")
        file_path = os.path.join(input_dir, filename)

        try:
            with io.open(file_path, 'rb') as image_file:
                content = image_file.read()
            image_vision = vision.Image(content=content)
        except Exception as e:
            print(f"      Read File Error: {e}")
            continue

        try:
            response = client.document_text_detection(image=image_vision)
            if response.error.message:
                raise Exception(f'{response.error.message}')
        except Exception as e:
            print(f"      OCR Error: {e}")
            continue

        cleaned_lines = []
        
        if response.full_text_annotation and response.full_text_annotation.pages:
            page = response.full_text_annotation.pages[0]
            img_height = page.height
            
            # Use variables extracted from Config
            header_threshold_y = img_height * header_margin_ratio
            footer_threshold_y = img_height * footer_margin_ratio
            
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    
                    # Use variables extracted from Config
                    if paragraph.confidence < min_confidence:
                        continue 
                        
                    ys = [vertex.y for vertex in paragraph.bounding_box.vertices]
                    if not ys: continue
                    avg_y = sum(ys) / len(ys)
                    
                    if avg_y < header_threshold_y or avg_y > footer_threshold_y:
                        continue
                        
                    para_text = ""
                    for word in paragraph.words:
                        word_text = "".join([symbol.text for symbol in word.symbols])
                        para_text += word_text + " "
                        
                    cleaned = clean_text(para_text)
                    if cleaned:
                        cleaned_lines.append(cleaned)
        
        full_text_cleaned = "\n".join(cleaned_lines)
        
        page_obj = {
            "page_id": page_id,
            "filename": filename,
            "full_text": full_text_cleaned,
            "lines": cleaned_lines
        }
        
        final_result.append(page_obj)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=4)

    print(f"\nCompleted! JSON file (from cleaned images) saved at:\n{output_path}")

if __name__ == "__main__":
    # Setup argparse to receive commands from main.py
    parser = argparse.ArgumentParser(description="Process cleaned images with Google OCR")
    parser.add_argument("--input_dir", required=True, help="Path to cleaned images directory")
    parser.add_argument("--output_json", required=True, help="Path to save the OCR JSON")
    parser.add_argument("--config", required=True, help="Path to config.json file")
    
    args = parser.parse_args()
    
    # Pass all variables to the function
    process_cleaned_images_to_json(args.input_dir, args.output_json, args.config)