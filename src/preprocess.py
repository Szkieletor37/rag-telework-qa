import os
from pathlib import Path

import pymupdf4llm
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_API_KEY_FREE = os.getenv('GEMINI_API_KEY_FREE')

def convert_pdf_to_md(pdf_file_name: Path, md_file_path: Path) -> None:
    """PDF を markdown に変換する関数"""
    md = pymupdf4llm.to_markdown(pdf_file_name)
    with open(md_file_path, "w", encoding="utf-8") as file:
        file.write(md)

    format_md(md)

def format_md(md):
    """変換した markdown を LLM で整形する関数"""
    load_dotenv()

    client = genai.Client(api_key=GEMINI_API_KEY_FREE)


    prompt = f"""
        このmarkdownを整形して: {md}
    """

    model = "gemini-3.6-flash"

    interaction = client.interactions.create(
        model=model,
        input=prompt,
    )

    print(interaction.output_text)

if __name__ == "__main__":
    convert_pdf_to_md("data/raw/telework_guide.pdf",
                      "data/md/telework_guide.md")


