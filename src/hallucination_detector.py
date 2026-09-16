import os
import sys
import json

# Import step functions directly — no subprocess needed
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from importlib import import_module as _imp

# Dynamically import modules (updated to match the new filenames in src/)
_step1 = _imp("pdf_clean_img")
_step2 = _imp("run_google_ocr")
_step3 = _imp("split_md_pages")
_step4 = _imp("detect_hallu_omiss")

preprocess_pdf_extract_and_clean = _step1.preprocess_pdf_extract_and_clean
process_cleaned_images_to_json   = _step2.process_cleaned_images_to_json
md_to_json_by_comment            = _step3.md_to_json_by_comment
detect                           = _step4.detect


class HallucinationDetector:
    
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        print("🤖 Detector initialized")

    def predict(self, pdf_path, md_path, output_dir="output"):
        """
        Run the full hallucination detection pipeline.

        Args:
            pdf_path:   Path to the source PDF (ground truth)
            md_path:    Path to the converted Markdown file
            output_dir: Directory to write intermediate files and final report.
                        Defaults to "output/" for backward compatibility.

        Returns:
            list: Per-page report dicts, or dict with "error" key on failure.
        """
        print(f"\n🚀 Starting analysis process...")
        print(f"📄 PDF: {pdf_path}")
        print(f"📝 MD:  {md_path}")

        # Prepare output directory
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        run_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(run_output_dir, exist_ok=True)

        # Define paths for required intermediate files
        clean_img_dir = os.path.join(run_output_dir, "cleaned_pages_for_ocr")
        meta_json     = os.path.join(run_output_dir, "extracted_images_metadata.json")
        ocr_json      = os.path.join(run_output_dir, "google_ocr.json")
        split_md_json = os.path.join(run_output_dir, f"{base_name}_split_md.json")
        final_report  = os.path.join(run_output_dir, f"{base_name}_final_report.json")

        try:
            # Step 1: Extract images and apply visual masking
            preprocess_pdf_extract_and_clean(pdf_path, run_output_dir, meta_json)

            # Step 2: Extract Ground Truth via Google Cloud Vision (OCR)
            process_cleaned_images_to_json(clean_img_dir, ocr_json, self.config_path)

            # Step 3: Split Markdown page-by-page (supports x2md <section id="page" /> and legacy <!-- --> comments)
            md_to_json_by_comment(md_path, split_md_json)

            # Step 4: Detect Hallucinations and Omissions (Scoring Engine)
            report_data = detect(
                ocr_json=ocr_json,
                md_json=split_md_json,
                meta_json=meta_json,
                output_report=final_report,
                config_path=self.config_path,
            )
            return report_data

        except Exception as e:
            return {"error": str(e)}