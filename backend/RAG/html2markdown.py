import html2text
from bs4 import BeautifulSoup
import pytesseract
from PIL import Image
import os
import glob
import shutil
import chromadb
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="blog_rag")
embedding_model = SentenceTransformer("bert-base-uncased")  # CLIP for images & text

# Ensure image directory exists
IMAGE_SAVE_DIR = "images"
MARKDOWN_SAVE_DIR = "markdown"
if not os.path.exists(IMAGE_SAVE_DIR):
    os.makedirs(IMAGE_SAVE_DIR)
if not os.path.exists(MARKDOWN_SAVE_DIR):
    os.makedirs(MARKDOWN_SAVE_DIR)

def html_to_markdown_with_math_images(html_content, html_file_path):
    """ Converts HTML to Markdown while preserving math and images """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Convert HTML to Markdown
    converter = html2text.HTML2Text()
    converter.body_width = 0
    markdown_text = converter.handle(str(soup))

    # Extract MathJax LaTeX equations
    math_expressions = []
    for script in soup.find_all("script", {"type": "math/tex"}):
        math_expressions.append(f"${script.text}$")  # Inline math

    for script in soup.find_all("script", {"type": "math/tex; mode=display"}):
        math_expressions.append(f"\n$$\n{script.text}\n$$\n")  # Block math

    # Extract MathML
    mathml_expressions = [str(mathml) for mathml in soup.find_all("math")]

    # Extract Images and convert to Markdown
    image_markdown = []
    for img in soup.find_all("img"):
        img_src = img.get("src")
        img_alt = img.get("alt", "Image")  # Default alt text

        if img_src:
            # Save a copy of the image if local
            img_filename = os.path.basename(img_src)
            img_path = os.path.join(IMAGE_SAVE_DIR, img_filename)

            if os.path.exists(img_src):  # If the image exists locally, copy it
                shutil.copy(img_src, img_path)
                img_markdown_path = f"./{IMAGE_SAVE_DIR}/{img_filename}"
            else:
                img_markdown_path = img_src  # Use original URL if not local
            
            image_markdown.append(f"![{img_alt}]({img_markdown_path})")

    # Combine everything
    final_markdown = markdown_text + "\n" + "\n".join(math_expressions + mathml_expressions + image_markdown)
    
    # Save to markdown file
    markdown_filename = MARKDOWN_SAVE_DIR + "/" +  os.path.splitext(os.path.basename(html_file_path))[0] + ".md"
    with open(markdown_filename, "w", encoding="utf-8") as f:
        f.write(final_markdown)

    return final_markdown, markdown_filename

def extract_text_from_image(image_path):
    """ Extracts text from images using OCR """
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text.strip()

def store_data_in_chroma(markdown_text, markdown_filename, image_dir):
    """ Stores text, math, and images in ChromaDB for retrieval """
    # Store Markdown text as an embedding
    text_embedding = embedding_model.encode(markdown_text)
    collection.add(ids=[markdown_filename], embeddings=[text_embedding], documents=[markdown_text])

    # Store images with extracted OCR text
    for img_file in glob.glob(os.path.join(image_dir, "*")):
        ocr_text = extract_text_from_image(img_file)
        img_embedding = embedding_model.encode(ocr_text)  # Image-text embedding
        collection.add(ids=[img_file], embeddings=[img_embedding], documents=[ocr_text])

def process_html_files(html_directory):
    """ Processes all HTML files, converts to Markdown, extracts math/images, and stores in ChromaDB """
    html_files = glob.glob(os.path.join(html_directory, "*.html"))
    
    for html_file in html_files:
        with open(html_file, "r", encoding="utf-8") as f:
            html_content = f.read()
        
        markdown_text, markdown_filename = html_to_markdown_with_math_images(html_content, html_file)
        store_data_in_chroma(markdown_text, markdown_filename, IMAGE_SAVE_DIR)

# Process all HTML files in the "blogs" directory
process_html_files("../../blogs")

print("✅ All HTML files processed, stored as Markdown, and indexed in ChromaDB!")
