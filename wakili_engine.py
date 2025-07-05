import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os
import json
import re
from tqdm import tqdm
import google.generativeai as genai
from PyPDF2 import PdfReader
import io
import docx

# --- CONFIGURATION ---
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set!")
genai.configure(api_key=GOOGLE_API_KEY)

JSON_DIR = "data/caselaw"
FAILURE_PHRASES = [
    "No summary or judgment body found.",
    "No meaningful text found",
    "AI summary generation failed",
    "AI summary failed",
    "AI summary was blocked"
]
summarization_model = genai.GenerativeModel('gemini-2.5')


# --- Helper Functions ---

def get_ai_summary_from_text(text):
    """Uses Gemini to summarize a given block of text."""
    if not text or len(text) < 150:  # Increased minimum length
        return "No meaningful text found in source document."

    safety_settings = {
        'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE', 'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
        'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE', 'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
    }

    prompt = f"Summarize the following Kenyan court ruling in one clear paragraph. Focus on the core legal issue, the final decision, and the primary reason for that decision: {text[:15000]}"
    try:
        response = summarization_model.generate_content(prompt, safety_settings=safety_settings,
                                                        request_options={'timeout': 120})
        return response.text if response.parts else "AI summary was blocked by safety filters."
    except Exception as e:
        return f"AI summary failed: {e}"


def get_universal_content(soup):
    """
    Finds the best possible text content from the page using a comprehensive fallback chain.
    Returns a tuple: (text_content, needs_ai_summary_boolean)
    """
    # Priority 1: Pre-written summaries
    order_dt = soup.find('dt', string=re.compile(r'\s*Order\s*'))
    if order_dt and order_dt.find_next_sibling('dd'):
        summary = order_dt.find_next_sibling('dd').get_text(separator='\n', strip=True)
        if len(summary) > 20: return summary, False

    abstract_div = soup.find('div', class_='judgment-abstract')
    if abstract_div:
        summary = abstract_div.get_text(separator='\n', strip=True)
        if len(summary) > 20: return summary, False

    holdings_div = soup.find('div', class_='akn-div holdings')
    if holdings_div:
        summary = holdings_div.get_text(separator='\n', strip=True)
        if len(summary) > 20: return summary, False

    # Priority 2: Standard text containers that need summarization
    content_container = soup.find('div', id='judgmentBody') or soup.find('div', id='decision')
    if content_container:
        return content_container.get_text(separator='\n', strip=True), True

    # Priority 3: Direct HTML content for very old cases
    direct_html_content = soup.find('div', class_='content__html')
    if direct_html_content:
        return direct_html_content.get_text(separator='\n', strip=True), True

    # Priority 4: PDF/DOCX as the last resort
    pdf_div = soup.find('div', {'data-pdf': True})
    if pdf_div and pdf_div.get('data-pdf'):
        pdf_url = "https://new.kenyalaw.org" + pdf_div['data-pdf']
        pdf_response = requests.get(pdf_url, timeout=60)
        if pdf_response.ok and pdf_response.content:
            text = ""
            with io.BytesIO(pdf_response.content) as pdf_file:
                try:
                    reader = PdfReader(pdf_file)
                    if reader.is_encrypted: reader.decrypt('')
                    for page in reader.pages: text += page.extract_text() or ""
                    if len(text) > 100: return text, True
                except Exception:
                    pass

    docx_link = soup.find('a', href=re.compile(r'\.docx'))
    if docx_link:
        docx_url = docx_link['href']
        if not docx_url.startswith('http'):
            docx_url = "https://kenyalaw-website-media.s3.amazonaws.com" + docx_url
        docx_response = requests.get(docx_url, timeout=60)
        if docx_response.ok and docx_response.content:
            text = ""
            with io.BytesIO(docx_response.content) as doc_file:
                document = docx.Document(doc_file)
                text = "\n".join([para.text for para in document.paragraphs])
            if len(text) > 100: return text, True

    return "No content found on page.", False


# --- Main Processing Function ---
def process_failed_files():
    print("--- Starting UNIVERSAL Pass: Finding and fixing all failed cases ---")
    if not os.path.exists(JSON_DIR):
        print(f"Directory '{JSON_DIR}' not found. Cannot proceed.")
        return

    files_to_fix = []
    all_files = os.listdir(JSON_DIR)
    print(f"--- Scanning {len(all_files)} total JSON files to find failures... ---")
    for filename in tqdm(all_files, desc="Scanning JSONs"):
        if not filename.endswith('.json'): continue
        filepath = os.path.join(JSON_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if any(phrase.lower() in data.get("content", "").lower() for phrase in FAILURE_PHRASES):
                files_to_fix.append(filepath)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    print(f"\n--- Found {len(files_to_fix)} files to fix. Starting processing. ---")
    if not files_to_fix:
        print("--- ✅ No files needed fixing. ---")
        return

    for json_filepath in tqdm(files_to_fix, desc="Fixing Files"):
        try:
            with open(json_filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            response = requests.get(data['source_url'], timeout=30)
            if not response.ok: continue
            soup = BeautifulSoup(response.content, 'html.parser')

            extracted_text, needs_ai_summary = get_universal_content(soup)

            final_content = extracted_text
            if needs_ai_summary:
                final_content = get_ai_summary_from_text(extracted_text)

            data['content'] = final_content
            data['last_updated'] = time.strftime("%Y-%m-%d")

            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)

            time.sleep(1)
        except Exception as e:
            print(f"\n[ERROR] processing {os.path.basename(json_filepath)}: {e}")
            continue

    print("\n--- ✅ UNIVERSAL Processing Pass Complete ---")


if __name__ == "__main__":
    process_failed_files()