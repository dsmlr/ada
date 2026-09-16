import json
import os
import re
import difflib
import argparse
import numpy as np

# Import library for chrF++
import sacrebleu

# Import Google Cloud libraries
import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from google.oauth2 import service_account

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def clean_text_nospace(text):
    """Clean text and remove spaces (for Jaccard and diff inspection)"""
    if not text: return ""
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'["“”\'’‘]', '', text)
    text = re.sub(r'[<>\\/#*|\[\]()_\-+\.`!•o0?]+', '', text)
    text = re.sub(r'ฝนสป\d+.*', '', text)  # Remove specific Thai document noise
    text = re.sub(r'วันที่\d+.*?\d{4}', '', text)
    text = text.replace('\u200b', '').replace('\ufeff', '')
    text = re.sub(r'[\n\r\t\s]+', '', text) 
    return text

def clean_text_for_chrf(text):
    """Clean text for chrF (remove all spaces similar to Jaccard)"""
    if not text: return ""
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text) 
    text = re.sub(r'["“”\'’‘]', '', text)
    text = re.sub(r'[<>\\/#*|\[\]()_\-+\.`!•o0?]+', '', text) 
    text = re.sub(r'ฝนสป\d+.*', '', text)
    text = re.sub(r'วันที่\d+.*?\d{4}', '', text)
    text = text.replace('\u200b', '').replace('\ufeff', '')
    text = re.sub(r'[\n\r\t\s]+', '', text)
    return text

def clean_text_for_semantic(text):
    """Clean text for Cosine similarity (preserve spaces for AI readability)"""
    if not text: return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_ngrams(text, n):
    if len(text) < n: return set([text]) if text else set()
    return set([text[i:i+n] for i in range(len(text)-n+1)])

def get_chrf_score(truth, prediction):
    """Calculate chrF score (Pure Character) using sacrebleu"""
    if not truth or not prediction: return 0.0
    try:
        chrf = sacrebleu.sentence_chrf(prediction, [truth], word_order=0)
        return chrf.score / 100.0
    except Exception as e:
        print(f"chrF Error: {e}")
        return 0.0

def get_vertex_cosine_score(text_a, text_b, semantic_model):
    if not text_a or not text_b: return 0.0
    try:
        inputs = [TextEmbeddingInput(text_a), TextEmbeddingInput(text_b)]
        embeddings = semantic_model.get_embeddings(inputs)
        vec_a = np.array(embeddings[0].values)
        vec_b = np.array(embeddings[1].values)
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0: return 0.0
        return max(0.0, float(dot_product / (norm_a * norm_b)))
    except Exception as e:
        print(f"Embedding Error: {e}")
        return 0.0

# Function to check image area coverage
# PAGE_AREA default = 3308 * 4676 (A4 at 400 DPI) — overridden by config["audit_thresholds"]["page_area"]
_DEFAULT_PAGE_AREA = 3308 * 4676

def is_image_heavy_page(page_id, metadata, threshold_ratio, page_area=None):
    images_on_page = [img for img in metadata if img.get('page_id') == page_id]
    if not images_on_page: return False, 0.0

    total_img_area = 0
    for img in images_on_page:
        coords = img.get('coordinates', [0,0,0,0])
        width = coords[2] - coords[0]
        height = coords[3] - coords[1]
        total_img_area += (width * height)

    area = page_area if page_area else _DEFAULT_PAGE_AREA
    ratio = total_img_area / area
    return ratio > threshold_ratio, ratio

# ==========================================
# CORE ANALYSIS LOGIC
# ==========================================

def analyze_page_hybrid(page_id, truth, prediction, config, semantic_model, is_img_heavy=False):

    # Stage 1: Preprocessing - Clean text
    truth_clean = clean_text_for_chrf(truth)
    pred_clean = clean_text_for_chrf(prediction)
    
    audit_cfg = config["audit_thresholds"]
    ngram_size = audit_cfg["ngram_size"]
    
    pass_score = audit_cfg["pass_chrf"]
    review_score = audit_cfg["review_chrf"]
    pass_recall = audit_cfg["pass_recall"]
    pass_cosine = audit_cfg["pass_cosine"]
    sparse_limit = audit_cfg.get("sparse_text_limit", 100)
    min_len = audit_cfg.get("min_len", 150)

    # Check Empty
    if not truth_clean and not pred_clean:
        return {"jaccard": 1.0, "chrf": 1.0, "cosine": 1.0, "recall": 1.0, "omission": [], "hallucination": [], "status": "Empty OK", "note": "Blank page"}
    if not truth_clean:
        status = "⚠️ Skipped (Image)" if is_img_heavy else "❌ Hallucination Only"
        return {"jaccard": 0.0, "chrf": 0.0, "cosine": 0.0, "recall": 0.0, "omission": [], "hallucination": [pred_clean], "status": status, "note": "PDF Empty"}
    if not pred_clean:
        return {"jaccard": 0.0, "chrf": 0.0, "cosine": 0.0, "recall": 0.0, "omission": ["(Entire page content missing)"], "hallucination": [], "status": "Missing Page", "note": "MD Empty"}

    # Stage 2: 3D Scoring (Jaccard, chrF, Cosine)
    gt_ngrams = get_ngrams(truth_clean, ngram_size)
    pred_ngrams = get_ngrams(pred_clean, ngram_size)
    intersection = gt_ngrams.intersection(pred_ngrams)
    union = gt_ngrams.union(pred_ngrams)
    
    jaccard_score = len(intersection) / len(union) if union else 0.0
    recall_score = len(intersection) / len(gt_ngrams) if gt_ngrams else 1.0
    
    chrf_score = get_chrf_score(truth_clean, pred_clean)

    truth_sem = clean_text_for_semantic(truth)
    pred_sem = clean_text_for_semantic(prediction)
    cosine_score = get_vertex_cosine_score(truth_sem, pred_sem, semantic_model)

    # Stage 3: Diff Inspection - Find differences >= min_len characters
    matcher = difflib.SequenceMatcher(None, truth_clean, pred_clean)
    raw_omissions, raw_hallucinations = [], []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'delete':
            if (i2-i1) >= min_len: raw_omissions.append(truth_clean[i1:i2])
        elif tag == 'insert':
            if (j2-j1) >= min_len: raw_hallucinations.append(pred_clean[j1:j2])
        elif tag == 'replace':
            if (i2-i1) >= min_len: raw_omissions.append(truth_clean[i1:i2])
            if (j2-j1) >= min_len: raw_hallucinations.append(pred_clean[j1:j2])

    # Verify using N-gram Coverage
    genuine_omissions = []
    for om in raw_omissions:
        chunk_ngrams = get_ngrams(om, ngram_size)
        if chunk_ngrams:
            coverage = len(chunk_ngrams.intersection(pred_ngrams)) / len(chunk_ngrams)
            if coverage < 0.75:
                genuine_omissions.append(f"[omiss {(1-coverage):.0%}]: {om}")

    genuine_hallucinations = []
    for hal in raw_hallucinations:
        chunk_ngrams = get_ngrams(hal, ngram_size)
        if chunk_ngrams:
            coverage = len(chunk_ngrams.intersection(gt_ngrams)) / len(chunk_ngrams)
            if coverage < 0.75:
                genuine_hallucinations.append(f"[hallu {(1-coverage):.0%}]: {hal}")

    # Stage 4: Decision Logic
    status = "Processed"
    note = ""

    if is_img_heavy:
        # For image-heavy pages (> image_heavy_ratio threshold)

        # Check text length in OCR (Sparse Text Check)
        # If < sparse_limit chars, treat as "Image Page" where text metrics are unreliable
        is_sparse_text = len(truth_clean) < sparse_limit
        
        if is_sparse_text:
            # For sparse text: Do not strictly fail, MD might contain image captions
            if len(genuine_hallucinations) > 0 or len(pred_clean) > len(truth_clean):
                status = "⚠️ Review (Vision)"
                note = "Sparse Text + AI Caption (Check Vision)"
            else:
                # PDF and MD text match perfectly
                status = "✅ Pass"
                note = "Sparse Text matched"
                
        else:
            # For image-heavy pages with significant text
            issues_found = []
            is_critical_fail = False # True if it's a critical failure

            if recall_score < pass_recall:
                # Is main content missing?
                issues_found.append("Omission (Low Recall)")
                is_critical_fail = True
                
            if cosine_score < pass_cosine:
                # If semantic shifts but original content remains (Check Recall), treat as image caption
                if recall_score >= pass_recall:
                    # Original content is preserved
                    issues_found.append("Context Shift (Likely Image Caption)")
                else:
                    # MD hallucinates and main content (Recall) drops
                    issues_found.append("Context Mismatch (Low Cosine)")
                    is_critical_fail = True
                
            if len(genuine_hallucinations) > 0:
                # Hallucinations > min_len flag as Extra Text (needs manual vision check)
                issues_found.append("Extra Text (Check Vision)")

            # Summarize decision
            if len(issues_found) > 0:
                if is_critical_fail:
                    status = "❌ Fail" 
                else:
                    status = "⚠️ Review (Vision)" 
                note = " | ".join(issues_found) 
                
            elif chrf_score >= pass_score: 
                # No critical failures and high chrF score
                status = "✅ Pass"
                note = "Perfect match"
            else:
                # Low chrF score forgiven due to image content
                status = "⚠️ Skipped (Image)"
                note = "Low score ignored (Image Content)"  
    else:
        # For Text-only pages (reliant on chrF)
        if chrf_score >= pass_score:
            status = "✅ Pass"
        elif chrf_score >= review_score:
            status = "⚠️ Review"
        else:
            status = "❌ Fail"

    return {
        "jaccard": jaccard_score,
        "chrf": chrf_score,
        "cosine": cosine_score,
        "recall": recall_score,
        "omission": genuine_omissions,
        "hallucination": genuine_hallucinations,
        "status": status,
        "note": note
    }

def detect(ocr_json, md_json, meta_json, output_report, config_path):
    """
    Core detection function. Importable directly by HallucinationDetector facade.
    
    Args:
        ocr_json:      Path to OCR JSON (from step 2)
        md_json:       Path to split MD JSON (from step 3)
        meta_json:     Path to image metadata JSON (from step 1)
        output_report: Path to save final JSON report
        config_path:   Path to config.json
    
    Returns:
        list: Per-page report dicts
    """
    print("🚀 Starting evaluation process (Metrics: chrF + N-gram Subsets + Cosine)...")

    # Load Config
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    gcp_cfg = config["google_cloud"]
    audit_cfg = config["audit_thresholds"]
    page_area = audit_cfg.get("page_area", None)  # None = use _DEFAULT_PAGE_AREA

    # Setup Certificate
    cert_path = gcp_cfg["cert_path"]
    if os.path.exists(cert_path):
        os.environ['SSL_CERT_FILE'] = cert_path
        os.environ['REQUESTS_CA_BUNDLE'] = cert_path
        os.environ['GRPC_DEFAULT_SSL_ROOTS_FILE_PATH'] = cert_path

     # Initialize Vertex AI (using separate SA Key file)
    auth_path = gcp_cfg["auth_config_path"]
    model_name = gcp_cfg["embedding_model"]

    # Load SA Config from keys folder
    with open(auth_path, "r", encoding="utf-8") as f:
        auth_cfg = json.load(f)
        
    project_id = auth_cfg["project_id"]
    location = auth_cfg["location"]
    sa_key_file = auth_cfg["service_account_key_file"]

    # Create Credentials and connect to Vertex AI
    credentials = service_account.Credentials.from_service_account_file(sa_key_file)
    vertexai.init(project=project_id, location=location, credentials=credentials)
    
    # Verify Key
    print("\n" + "="*60)
    print("🔍 [DEBUG] System Authentication Check")
    print(f"🏢 Active Project : {project_id}")
    # Check if loaded Credentials contain SA email
    if hasattr(credentials, 'service_account_email'):
        print(f"🔑 Using SA Key   : {credentials.service_account_email}")
    else:
        print("⚠️ Using Default/Other Credentials (Not SA File)")
    print("="*60 + "\n")

    semantic_model = TextEmbeddingModel.from_pretrained(model_name)

    # 1. Load Metadata
    image_metadata = []
    if os.path.exists(meta_json):
        with open(meta_json, 'r', encoding='utf-8') as f:
            image_metadata = json.load(f)

    # 2. Load Google OCR
    with open(ocr_json, 'r', encoding='utf-8') as f:
        google_data = json.load(f)
    gt_map = {}
    if isinstance(google_data, list):
        for item in google_data:
            pid = item.get('page_id') or item.get('page')
            text = item.get('full_text') or item.get('text_content') or "".join(item.get('lines', []))
            if pid: gt_map[pid] = gt_map.get(pid, "") + str(text) 

    # 3. Load MD
    with open(md_json, 'r', encoding='utf-8') as f:
        md_data = json.load(f)
    pred_map = {item.get('page_id'): item.get('md_text', '') for item in md_data if isinstance(item, dict) and item.get('page_id')}

    all_pages = sorted(set(gt_map.keys()) | set(pred_map.keys()))
    final_report = []
    
    # Update Print Table 
    print("-" * 155)
    print(f"{'Page':<5} | {'Jaccard':<8} | {'chrF':<8} | {'Cosine':<8} | {'Recall':<8} | {'Issues (Om/Hal)':<16} | {'Status':<20} | {'Note'}")
    print("-" * 155)

    for pid in all_pages:
        gt_text = gt_map.get(pid, "")
        pred_text = pred_map.get(pid, "")
        
        is_heavy, img_ratio = is_image_heavy_page(pid, image_metadata, audit_cfg["image_heavy_ratio"], page_area=page_area)
        
        result = analyze_page_hybrid(pid, gt_text, pred_text, config, semantic_model, is_img_heavy=is_heavy)
        
        final_report.append({
            "page_id": pid,
            "is_image_heavy": is_heavy,
            "image_ratio": img_ratio,
            "scores": {
                "jaccard_score": result['jaccard'],
                "chrf_score": result['chrf'],
                "cosine_score": result['cosine'],
                "recall_score": result['recall']
            },
            "status": result['status'],
            "note": result['note'],
            "details": {
                "omissions": result['omission'],
                "hallucinations": result['hallucination']
            }
        })

        note_display = result['note']
        if is_heavy: 
            note_display += f" (Img: {img_ratio:.0%})"
        
        issue_summary = f"Om: {len(result['omission'])}, Hal: {len(result['hallucination'])}"

        print(f"{pid:<5} | {result['jaccard']:>7.2%} | {result['chrf']:>7.2%} | {result['cosine']:>7.2%} | {result['recall']:>7.2%} | {issue_summary:<16} | {result['status']:<20} | {note_display}")

    os.makedirs(os.path.dirname(os.path.abspath(output_report)), exist_ok=True)
    with open(output_report, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, ensure_ascii=False, indent=4)
    print(f"\n💾 Saved report to: {output_report}")

    return final_report


def main(args):
    """CLI entry point — thin wrapper around detect()."""
    detect(
        ocr_json=args.ocr_json,
        md_json=args.md_json,
        meta_json=args.meta_json,
        output_report=args.output_report,
        config_path=args.config,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect Hallucination and Omission")
    parser.add_argument("--ocr_json", required=True, help="Path to OCR JSON")
    parser.add_argument("--md_json", required=True, help="Path to Markdown JSON")
    parser.add_argument("--meta_json", required=True, help="Path to Metadata JSON")
    parser.add_argument("--output_report", required=True, help="Path to save the final report")
    parser.add_argument("--config", required=True, help="Path to config.json file")
    
    args = parser.parse_args()
    main(args)