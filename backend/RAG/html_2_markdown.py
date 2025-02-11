import html2text
from bs4 import BeautifulSoup
import os
import shutil

def html_to_markdown_with_math(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Extract regular text as Markdown
    converter = html2text.HTML2Text()
    converter.body_width = 0  # Prevents line wrapping
    markdown_text = converter.handle(str(soup))

    # Extract KaTeX / MathJax LaTeX expressions
    math_expressions = []
    for script in soup.find_all("script", {"type": "math/tex"}):
        math_expressions.append(f"${script.text}$")  # Inline math

    for script in soup.find_all("script", {"type": "math/tex; mode=display"}):
        math_expressions.append(f"\n$$\n{script.text}\n$$\n")  # Block math

    # Extract MathML (optional, convert to LaTeX manually if needed)
    mathml_expressions = []
    for mathml in soup.find_all("math"):
        mathml_expressions.append(f"\n[MathML] {mathml}\n")  # Placeholder for manual conversion

    # Append extracted math expressions
    final_markdown = markdown_text + "\n" + "\n".join(math_expressions + mathml_expressions)
    
    return final_markdown


def html_to_markdown_with_images(html_content, image_save_dir="images"):
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Convert general HTML to Markdown
    converter = html2text.HTML2Text()
    converter.body_width = 0
    markdown_text = converter.handle(str(soup))

    # Extract images
    image_links = []
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)  # Ensure image directory exists

    for img in soup.find_all("img"):
        img_src = img.get("src")
        img_alt = img.get("alt", "Image")  # Default alt text
        
        if img_src:
            image_links.append(f"![{img_alt}]({img_src})")
            
            # Download and save images if they're local
            img_filename = os.path.basename(img_src)
            img_path = os.path.join(image_save_dir, img_filename)
            
            if os.path.exists(img_src):  # If image is local, copy it
                shutil.copy(img_src, img_path)

    # Append extracted images to Markdown
    final_markdown = markdown_text + "\n" + "\n".join(image_links)

    return final_markdown




# Example usage
with open("../../blogs/basis_splines.html", "r", encoding="utf-8") as f:
    html_content = f.read()

markdown_output = html_to_markdown_with_images(html_content)

# Save to a .md file
with open("basis_splines_w_images.md", "w", encoding="utf-8") as f:
    f.write(markdown_output)

print("Markdown file created successfully!")