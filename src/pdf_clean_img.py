import fitz  # PyMuPDF
import os
import json
import argparse
from pdf2image import convert_from_path
from PIL import Image, ImageDraw

def preprocess_pdf_extract_and_clean(pdf_path, output_base_dir, meta_path):
    print(f"Starting image extraction and cleaning process: {pdf_path}")
    
    # 1. Prepare storage directories
    clean_img_dir = os.path.join(output_base_dir, "cleaned_pages_for_ocr") # Store cleaned pages (masked) for OCR
    cropped_img_dir = os.path.join(output_base_dir, "extracted_images")     # Store cropped images
    
    os.makedirs(clean_img_dir, exist_ok=True)
    os.makedirs(cropped_img_dir, exist_ok=True)
    
    # Log image metadata (page, path, coordinates)
    extraction_metadata = [] 

    # 2. Open PDF file
    try:
        doc = fitz.open(pdf_path)
        # Convert to image at 400 DPI
        pil_images = convert_from_path(pdf_path, dpi=400)
    except Exception as e:
        print(f"❌ Error loading PDF: {e}")
        return

    # 3. Loop through each page
    for i, pil_image in enumerate(pil_images):
        page_num = i + 1
        pdf_page = doc[i]
        
        # Calculate scale (Pixels vs Points)
        pdf_w, pdf_h = pdf_page.rect.width, pdf_page.rect.height
        img_w, img_h = pil_image.size
        scale_x, scale_y = img_w / pdf_w, img_h / pdf_h
        
        # Find image locations
        image_list = pdf_page.get_images(full=True)
        draw = ImageDraw.Draw(pil_image)
        
        print(f"Page {page_num}: Found {len(image_list)} image(s)")
        
        for img_idx, img in enumerate(image_list):
            xref = img[0]
            rects = pdf_page.get_image_rects(xref)
            
            for rect_idx, rect in enumerate(rects):
                # Convert coordinates to Pixels
                box = (
                    int(rect.x0 * scale_x),
                    int(rect.y0 * scale_y),
                    int(rect.x1 * scale_x),
                    int(rect.y1 * scale_y)
                )
                
                # ------------------------------------------------
                # STEP 1: CROP & SAVE (Extract image)
                # ------------------------------------------------
                # Generate unique filename
                crop_filename = f"p{page_num}_img{img_idx}_{rect_idx}.jpg"
                crop_filepath = os.path.join(cropped_img_dir, crop_filename)
                
                try:
                    # Crop the image based on bounding box
                    cropped_img = pil_image.crop(box)
                    cropped_img.save(crop_filepath, "JPEG")
                    
                    # Save metadata for later use
                    extraction_metadata.append({
                        "page_id": page_num,
                        "image_id": f"{img_idx}_{rect_idx}",
                        "file_path": crop_filepath,
                        "coordinates": box # (left, top, right, bottom)
                    })
                except Exception as e:
                    print(f"      ⚠️ Crop Error: {e}")

                # ------------------------------------------------
                # STEP 2: MASK (White-out the original image)
                # ------------------------------------------------
                draw.rectangle(box, fill="white", outline="white")

        # 4. Save the cleaned page (for OCR)
        clean_page_path = os.path.join(clean_img_dir, f"page_{page_num}_clean.jpg")
        pil_image.save(clean_page_path, "JPEG")

    # 5. Save metadata as JSON
    # Use the meta_path passed from main.py
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(extraction_metadata, f, ensure_ascii=False, indent=4)

    print("\n✅ Process completed successfully!")
    print(f"📂 1. OCR-ready images (Cleaned): {clean_img_dir}")
    print(f"📂 2. Cropped images (Crops): {cropped_img_dir}")
    print(f"📄 3. Image metadata (JSON): {meta_path}")

if __name__ == "__main__":
    # Set up external arguments (passed from main.py)
    parser = argparse.ArgumentParser(description="Process PDF and clean images")
    parser.add_argument("--input_pdf", required=True, help="Path to the input PDF file")
    parser.add_argument("--output_dir", required=True, help="Base output directory")
    parser.add_argument("--meta_json", required=True, help="Path to save extracted images metadata JSON")
    
    args = parser.parse_args()
    
    # Pass the received arguments to the main function
    preprocess_pdf_extract_and_clean(args.input_pdf, args.output_dir, args.meta_json)