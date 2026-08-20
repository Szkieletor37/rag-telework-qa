import os
from logging import DEBUG, getLogger
from pathlib import Path

import pymupdf4llm
from dotenv import load_dotenv
from google import genai

load_dotenv()

logger = getLogger(__name__)
logger.setLevel(DEBUG)

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_API_KEY_FREE = os.getenv('GEMINI_API_KEY_FREE')

def convert_pdf_to_md(pdf_file_name) -> None:
    """PDF を markdown に変換する関数"""
    md_file_name = os.path.splitext(os.path.basename(pdf_file_name))[0] + ".md"
    pdf_file_path = "data/raw_pdf/" + pdf_file_name
    raw_md_file_path = "data/raw_md/" + f"raw_{md_file_name}"
    cleaned_md_file_path = "data/cleaned_md/" + f"cleaned_{md_file_name}"

    if not os.path.exists(raw_md_file_path):
        raw_md = pymupdf4llm.to_markdown(pdf_file_path)
        with open(raw_md_file_path) as file:
            file.write(raw_md)

    if not os.path.exists(cleaned_md_file_path):
        cleaned_md = clean_markdown(raw_md)

        with open(cleaned_md_file_path,
                "w", encoding="utf-8") as file:
            file.write(cleaned_md)


def clean_markdown(md_content):
    """変換した markdown を LLM で整形する関数"""
    client = genai.Client(api_key=GEMINI_API_KEY_FREE)

    prompt = f"""
このmarkdownを整形して: {md_content}
    """

    model = "gemini-3.6-flash"

    interaction = client.interactions.create(
        model=model,
        input=prompt,
    )

    print(interaction.output_text)

if __name__ == "__main__":
    convert_pdf_to_md("telework_guide.pdf")

