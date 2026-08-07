import pymupdf4llm


def convert_pdf_to_md():
    """
    PDF を markdown に変換する関数
    """
    md = pymupdf4llm.to_markdown("telework_guide.pdf")
    file_path = "data/processed/telework_guide.md"
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(md)

def __init__():
    convert_pdf_to_md()