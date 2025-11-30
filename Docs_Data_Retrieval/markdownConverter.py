import pdfplumber

def pdf_to_markdown(input_pdf, output_md):
    markdown_text = ""

    with pdfplumber.open(input_pdf) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                markdown_text += text + "\n\n"

    with open(output_md, "w", encoding="utf-8") as file:
        file.write(markdown_text)

    print(f"Markdown file saved as: {output_md}")

# Example Usage
pdf_to_markdown("input.pdf", "data/output.md")
