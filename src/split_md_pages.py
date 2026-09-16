import re
import json
import os
import argparse

def md_to_json_by_comment(md_file_path, output_json_path=None):
    print(f"🚀 Starting Markdown segmentation: {md_file_path}")
    
    # Read the markdown file
    with open(md_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Combine both patterns (Split whenever either is found)
    # Supports both x2md style: <section id="page-N" /> and legacy style: <!-- comment -->
    COMBINED_PATTERN = r'<section id="[^"]+" />|<!--.*?-->'

    # 1. Find all page delimiters to determine total expected tags
    delimiters = re.findall(COMBINED_PATTERN, content, flags=re.DOTALL)
    num_tags = len(delimiters)
    print(f"=== Found {num_tags} page delimiter tags/comments ===")
    
    # 2. Split content using the combined pattern
    parts = re.split(COMBINED_PATTERN, content, flags=re.DOTALL)

    pages_text = []
    
    # --- LOGIC Alignment Fix ---
    # Normally, if a Tag is at the very top, parts[0] is empty whitespace.
    # However, if there is text before the first Tag, parts[0] will contain content,
    # and len(parts) will equal num_tags + 1.
    
    if len(parts) > 0 and parts[0].strip() != "":
        print(f"⚠️ Detected loose content before the first page tag ({len(parts)} chunks found) -> Merging header content.")
        
        # Merge the first chunk (index 0) with the expected first page (index 1)
        # If there is only one chunk (no tags found), it will process just that chunk.
        first_page_content = parts[0].strip()
        if len(parts) > 1:
            first_page_content += "\n\n" + parts[1].strip()
            
        pages_text.append(first_page_content)
        
        # Process subsequent pages (starting from index 2)
        for p in parts[2:]:
            pages_text.append(p.strip())
    else:
        # Standard case: Tag is at the top, or split returns expected results.
        # We skip parts[0] which contains empty values.
        for p in parts:
            if p.strip():
                pages_text.append(p.strip())

    # 3. Format into JSON structure
    final_pages = []
    for i, text in enumerate(pages_text):
        final_pages.append({
            "page_id": i + 1,
            "md_text": text
        })

    print(f"Total extracted pages: {len(final_pages)} (Formatting complete)\n")

    # If output path is provided -> Save to file
    if output_json_path:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(final_pages, f, ensure_ascii=False, indent=4)
        print(f"💾 Successfully saved JSON output to: {output_json_path}")

    return final_pages

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split Markdown pages by HTML comments and section tags")
    parser.add_argument("--input_md", required=True, help="Path to the input Markdown file")
    parser.add_argument("--output_json", required=True, help="Path to save the splitted JSON file")
    
    args = parser.parse_args()
    result = md_to_json_by_comment(args.input_md, args.output_json)
    print("✅ Process completed successfully!")