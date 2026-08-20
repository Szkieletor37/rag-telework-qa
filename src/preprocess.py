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

def convert_pdf_to_md(raw_file_path: Path, interim_file_path: Path,
                      processed_file_path: Path) -> None:
    """PDF を markdown に変換し、整形する関数"""
    #md_file_name = os.path.splitext(os.path.basename(pdf_file_name))[0] + ".md"
    #pdf_file_path = "data/raw_pdf/" + pdf_file_name

    if not raw_file_path.exists():
        logger.error(f"変換元ファイルがないよ: {raw_file_path}")
        return

    if not interim_file_path.exists():
        logger.info(f"PDFからMarkdownへの変換を開始します: {raw_file_path}")
        interim_md = pymupdf4llm.to_markdown(raw_file_path)
        interim_file_path.write_text(interim_md, encoding="utf-8")
        logger.info(f"変換後の.mdファイルを保存しました。: {interim_file_path}")
    else:
        logger.info(f"既存のinterim .mdファイルを利用します: {interim_file_path}")
        interim_md = interim_file_path.read_text(encoding="utf-8")


    logger.info("MarkdownをGeminiで整形します...")
    if processed_file_path.exists():
        logger.info("整形後の.mdファイルは既に存在するため、上書きされます...")

    processed_md = clean_markdown(interim_md)
    processed_file_path.write_text(processed_md, encoding="utf-8")
    
    logger.info(f"整形後の.mdファイルを作成しました: {processed_file_path}")


def clean_markdown(md_content):
    """変換した markdown を LLM で整形する関数"""
    client = genai.Client(api_key=GEMINI_API_KEY_FREE)

    prompt = f"""
このmarkdownを整形して: {md_content}
    """

    model = "gemini-3.6-flash"
    
    # token数のカウント
    total_tokens = client.models.count_tokens(
        model=model,
        contents=prompt
    )
    logger.info(f"total入力tokens: {total_tokens.total_tokens}")


"""
    interaction = client.interactions.create(
        model=model,
        input=prompt,
    )
    logger.info(f"使用したtoken数: {interaction.usage}")

    return interaction.output_text

if __name__ == "__main__":
    convert_pdf_to_md(Path("data/raw_pdf/telework_guide.pdf"), Path("data/raw_md/telework_guide.md"), Path("data/cleaned_md/telework_guide.md"))

