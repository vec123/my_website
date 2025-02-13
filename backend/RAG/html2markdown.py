import html2text
from bs4 import BeautifulSoup
import pytesseract
from PIL import Image
import os
import glob
import shutil
import chromadb
import re
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from llama_parse import LlamaParse
import re

# Initialize Models
summarizer = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
embedding_model = SentenceTransformer("all-mpnet-base-v2")  # All-MPNet model for embeddings

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="blog_rag")

# Ensure directories exist
IMAGE_SAVE_DIR = "images"
MARKDOWN_SAVE_DIR = "markdown"
CHUNK_SAVE_DIRC= "chunks"
if not os.path.exists(IMAGE_SAVE_DIR):
    os.makedirs(IMAGE_SAVE_DIR)
if not os.path.exists(MARKDOWN_SAVE_DIR):
    os.makedirs(MARKDOWN_SAVE_DIR)
if not os.path.exists(CHUNK_SAVE_DIRC):
    os.makedirs(CHUNK_SAVE_DIRC)

def split_document_into_chunks(markdown_text, max_tokens=500):
    """ Splits the markdown document into chunks that fit within the token limit of the model """
    chunks = []
    current_chunk = ""
    for paragraph in markdown_text.split("\n"):
        if len(current_chunk.split()) + len(paragraph.split()) > max_tokens:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            current_chunk += "\n" + paragraph
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def sanitize_filename(filename):
    """Sanitizes the filename by replacing invalid characters with underscores."""
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def add_latex_placeholder(text):
    """Adds a placeholder text to LaTeX formulas."""
    # Regex to detect $$...$$ and \[...\] LaTeX formulas
    latex_pattern = r'(\$\$.*?\$\$|\\\[.*?\\\])'
    return re.sub(latex_pattern, r'\g<0> (latex formula, embed accordingly and with the related context.)', text)

def parse_markdown_hierarchy(markdown_text):
    """
    Parse the markdown text into a hierarchical structure based on headers.
    """
    lines = markdown_text.split("\n")
    structure = []
    current_section = None
    
    for line in lines:
        # Match headers with one or more hashes
        match = re.match(r'^(#+)\s+(.*)', line.strip())
        
        if match:
            level = len(match.group(1))  # Number of hashes indicates header level
            caption = match.group(2).strip()

            # Create a new section or subsection
            if level == 1:
                # Main section
                current_section = {"caption": caption, "subsections": []}
                structure.append(current_section)
            elif level == 2:
                # Subsection of the current section
                if current_section is None:
                    raise ValueError("Current section is None. Something went wrong in the structure.")
                if "subsections" not in current_section:
                    current_section["subsections"] = []  # Initialize subsections list
                current_section["subsections"].append({"caption": caption, "subsections": []})
            elif level == 3:
                # Subsubsection of the current subsection
                if current_section is None or "subsections" not in current_section or len(current_section["subsections"]) == 0:
                    raise ValueError("No subsections available to append to at level 3.")
                current_section["subsections"][-1]["subsections"].append({"caption": caption, "subsections": []})
            elif level == 4:
                # Handle deeper levels similarly
                if current_section is None or "subsections" not in current_section or len(current_section["subsections"]) == 0:
                    raise ValueError("No subsections available to append to at level 4.")
                if len(current_section["subsections"][-1]["subsections"]) == 0:
                    current_section["subsections"][-1]["subsections"].append({"caption": caption, "subsections": []})
                else:
                    current_section["subsections"][-1]["subsections"][-1]["subsections"].append({"caption": caption, "subsections": []})
            elif level == 5:
                # Handle deeper levels similarly
                if current_section is None or "subsections" not in current_section or len(current_section["subsections"]) == 0:
                    raise ValueError("No subsections available to append to at level 5.")
                if len(current_section["subsections"][-1]["subsections"]) == 0:
                    current_section["subsections"][-1]["subsections"].append({"caption": caption, "subsections": []})
                else:
                    current_section["subsections"][-1]["subsections"][-1]["subsections"].append({"caption": caption, "subsections": []})

    return structure

def split_document_into_chunks_and_save(markdown_text, max_tokens=500, output_dir="chunks"):
    """Splits the markdown document into chunks at each caption, ensures no chunk exceeds the token limit, and saves each chunk to a Markdown file with the full caption structure."""
    chunks = []
    current_chunk = ""
    chunk_counter = 1

    # Parse the markdown structure based on the headers
    caption_structure = parse_markdown_hierarchy(markdown_text)

    # Split the markdown into lines for processing
    lines = markdown_text.split("\n")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Variable to keep track of the chunk index
    for line in lines:
        match = re.match(r'^(#{1,6})\s+(.*)$', line.strip())
        
        if match:
            # If current_chunk is not empty and exceeds the max_tokens limit, save it
            if current_chunk:
                # Sanitize the caption for use as a filename
                sanitized_caption = sanitize_filename("default_caption")
                chunk_filename = f"{sanitized_caption}_{chunk_counter}.md"
                chunk_filepath = os.path.join(output_dir, chunk_filename)
                
                # Ensure the directory exists
                os.makedirs(os.path.dirname(chunk_filepath), exist_ok=True)

                # Write the chunk to file
                with open(chunk_filepath, "w", encoding="utf-8") as f:
                    f.write(current_chunk.strip())
                
                # Append the chunk with the full caption structure
                chunks.append({
                    "content": current_chunk.strip(), 
                    "caption_structure": caption_structure
                })
                chunk_counter += 1
                current_chunk = line  # Start a new chunk with the current caption
            else:
                # Start the chunk with caption context
                current_chunk = line  # Start the chunk with structure

        else:
            # Add the line to the current chunk
            current_chunk += "\n" + add_latex_placeholder(line)  # Add placeholder for LaTeX formulas
    
    # Don't forget to add the last chunk if it exists
    if current_chunk:
        # Sanitize the caption for use as a filename
        sanitized_caption = sanitize_filename("default_caption")
        chunk_filename = f"{sanitized_caption}_{chunk_counter}.md"
        chunk_filepath = os.path.join(output_dir, chunk_filename)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(chunk_filepath), exist_ok=True)

        # Write the last chunk to file
        with open(chunk_filepath, "w", encoding="utf-8") as f:
            f.write(current_chunk.strip())
        
        # Append the last chunk with the full caption structure
        chunks.append({
            "content": current_chunk.strip(),
            "caption_structure": caption_structure
        })
    
    return chunks


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

    # Convert LaTeX to readable text
    for i in range(len(math_expressions)):
        math_expressions[i] = latex_to_text(math_expressions[i])

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
    final_markdown = markdown_text + "\n" + "\n".join(math_expressions + image_markdown)
    
    # Save to markdown file
    markdown_filename = MARKDOWN_SAVE_DIR + "/" +  os.path.splitext(os.path.basename(html_file_path))[0] + ".md"
    with open(markdown_filename, "w", encoding="utf-8") as f:
        f.write(final_markdown)

    return final_markdown, markdown_filename

def preprocess_with_caption_boost(chunk, caption=None):
    """ Preprocess the chunk and add the caption with higher importance. """
    if caption:
        # Prepend the caption (or repeat it) to the chunk to give it more importance
        chunk = caption + " " + caption + " " + caption +" "+ chunk  # Optionally, you can repeat the caption if needed
    return chunk

def store_data_in_chroma(markdown_text, markdown_filename, image_dir):
    """ Store text chunks, math, and images in ChromaDB for retrieval """
    chunks = split_document_into_chunks(markdown_text)
    
    # Assume you extract captions (you can use regex or HTML parsing to identify them)
    captions = extract_captions_from_markdown(markdown_text)
    
    for i,chunk in enumerate(chunks):
        # Extract relevant caption for this chunk (you can enhance the logic based on the structure)
        chunk_caption = captions.get(chunk, None)  # Assuming you have a method to map chunks to captions
        
        # Preprocess chunk to give the caption more importance
        chunk_with_caption_boost = preprocess_with_caption_boost(chunk, caption=chunk_caption)
            
        # Save chunks to markdown file
        chunks_filename = "chunks/{}_{}.md".format(chunk_caption,i)
        with open(chunks_filename, "w", encoding="utf-8") as f:
            f.write(chunk_with_caption_boost)
            
        # Embed the chunk (now including boosted captions)
        text_embedding = embedding_model.encode(chunk_with_caption_boost)
        
        # Store chunk in ChromaDB
        collection.add(
            ids=[markdown_filename], 
            embeddings=[text_embedding], 
            documents=[chunk_with_caption_boost]
        )
    
    # Extract and embed math formulas separately
    #math_formulas = extract_math_formulas(markdown_text)
    #for formula in math_formulas:
    #    math_embedding = math_embedding_model.encode(formula)
    #    collection.add(
    #        ids=[markdown_filename + "_math_" + str(math_formulas.index(formula))], 
    #        embeddings=[math_embedding], 
    #        documents=[formula]
    #    )

def summarize_text(text):
    """ Summarize the given text using T5 """
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = summarizer.generate(inputs, max_length=150, num_beams=2, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def process_html_files(html_directory):
    """ Processes all HTML files, converts to Markdown, extracts math/images, and stores in ChromaDB """
    html_files = glob.glob(os.path.join(html_directory, "*.html"))
    print("html_files", html_files)
    for html_file in html_files:
        with open(html_file, "r", encoding="utf-8") as f:
            html_content = f.read()
        
        markdown_text, markdown_filename = html_to_markdown_with_math_images(html_content, html_file)
        split_document_into_chunks_and_save(markdown_text, output_dir="chunks")
       # store_data_in_chroma(markdown_text, markdown_filename, IMAGE_SAVE_DIR)

def summarize_markdown_files():
    """ Processes all Markdown files and generates summaries """
    markdown_directory = "markdown/"
    markdown_files = glob.glob(os.path.join(markdown_directory, "*.md"))

    for markdown_file in markdown_files:
        with open(markdown_file, "r", encoding="utf-8") as f:
                markdown_content = f.read()
        summarized_markdown_text = summarize_text(markdown_content)

        # Save to markdown file
        print("Summarizing:", markdown_file)
        markdown_filename = "summarized_markdown/" +  os.path.splitext(os.path.basename(markdown_file))[0] + "_summary.md"
        with open(markdown_filename, "w", encoding="utf-8") as f:
            f.write(summarized_markdown_text)

# Execute the functions

# 1. Process all HTML files (from your specific directory)
html_directory = "../../blogs/"  # Specify your directory containing HTML files
process_html_files(html_directory)

# 2. Summarize all processed markdown files
#summarize_markdown_files()

print("✅ Script execution complete!")
