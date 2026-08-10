from pathlib import Path

import pymupdf4llm


def convert_pdf_to_md(pdf_file_name: Path, md_file_path: Path) -> None:
    """PDF を markdown に変換する関数"""
    md = pymupdf4llm.to_markdown(pdf_file_name)
    with open(md_file_path, "w", encoding="utf-8") as file:
        file.write(md)

if __name__ == "__main__":
    convert_pdf_to_md("data/raw/telework_guide.pdf",
                      "data/md/telework_guide.md")
