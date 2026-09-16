import os
import sys
import argparse
import subprocess

def run_step(script_name, args_list):
    """Function to execute scripts located in the same directory as main.py"""
    # Get the absolute path of the current script (src/ directory)
    src_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(src_dir, script_name)
    
    command = [sys.executable, script_path] + args_list
    print(f"\n▶[Running]: {' '.join(command)}")
    
    result = subprocess.run(command)
    
    if result.returncode != 0:
        print(f"\nError occurred in {script_name} (Pipeline aborted)")
        sys.exit(1)
    print(f"{script_name} executed successfully!\n" + "-"*60)

def main():
    # ==========================================
    # 1. Configure argparse for input parameters
    # ==========================================
    parser = argparse.ArgumentParser(description="PDF vs Markdown Audit Pipeline Orchestrator")
    
    parser.add_argument("--pdf", required=True, help="Path to the original PDF file (Raw PDF)")
    parser.add_argument("--md", required=True, help="Path to the original Markdown file (Raw MD)")
    parser.add_argument("--config", default="config.json", help="Path to the configuration JSON file")
    
    args = parser.parse_args()
    
    input_pdf = args.pdf
    input_md = args.md
    config_path = args.config

    if not os.path.exists(input_pdf):
        print(f"PDF file not found at: {input_pdf}")
        sys.exit(1)
    if not os.path.exists(input_md):
        print(f"Markdown file not found at: {input_md}")
        sys.exit(1)

    print("============================================================")
    print("Starting Document Audit Pipeline")
    print(f"📄 Input PDF: {input_pdf}")
    print(f"📝 Input MD:  {input_md}")
    print("============================================================")

    # ==========================================
    # 2. Auto-generate paths for output directory
    # ==========================================
    base_name = os.path.splitext(os.path.basename(input_pdf))[0]
    
    # Ensure output is generated in the root directory
    output_base_dir = os.path.join(os.getcwd(), "output", base_name)
    os.makedirs(output_base_dir, exist_ok=True)

    clean_img_dir = os.path.join(output_base_dir, "cleaned_pages_for_ocr")
    metadata_json = os.path.join(output_base_dir, "extracted_images_metadata.json")
    ocr_json = os.path.join(output_base_dir, "google_ocr.json")
    md_json = os.path.join(output_base_dir, f"{base_name}_split_md.json")
    final_report = os.path.join(output_base_dir, f"{base_name}_final_report.json")

    # ==========================================
    # 3. Execute Pipeline Steps Sequentially
    # ==========================================
    
    # Step 1: Extract and mask images
    run_step("pdf_clean_img.py", [
        "--input_pdf", input_pdf,
        "--output_dir", output_base_dir,
        "--meta_json", metadata_json
    ])

    # Step 2: Generate Google OCR Ground Truth
    run_step("run_google_ocr.py", [
        "--input_dir", clean_img_dir,
        "--output_json", ocr_json,
        "--config", config_path
    ])

    # Step 3: Split Markdown page-by-page
    run_step("split_md_pages.py", [
        "--input_md", input_md,
        "--output_json", md_json
    ])

    # Step 4: Detect Hallucinations and Omissions
    run_step("detect_hallu_omiss.py", [
        "--ocr_json", ocr_json,
        "--md_json", md_json,
        "--meta_json", metadata_json,
        "--output_report", final_report,
        "--config", config_path
    ])

    print("\nPipeline completed successfully!")
    print(f"View the final audit report at: {final_report}")

if __name__ == "__main__":
    main()