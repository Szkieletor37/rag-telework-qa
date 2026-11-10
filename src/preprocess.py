import os
from logging import DEBUG, getLogger
from pathlib import Path

from config import PREPROCESS_PROMPT_DIR, CLEANING_MODEL

import pymupdf4llm
from dotenv import load_dotenv
from google import genai
from jinja2 import Template

load_dotenv()

logger = getLogger(__name__)
logger.setLevel(DEBUG)

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_API_KEY_FREE = os.getenv('GEMINI_API_KEY_FREE')

def build_cleaning_prompt(prompt_template:str, raw_md_content: str) -> str:
    """テンプレートプロンプトと変換後（整形前）のmarkdownから、cleaning用プロンプトを組み立てる関数"""

    template = Template(prompt_template)

    return template.render(
        raw_md_content=raw_md_content
    )


def convert_pdf_to_md(raw_pdf_path: Path, raw_md_path: Path) -> None:
    """PDF を markdown に変換し、整形する関数"""
    #md_file_name = os.path.splitext(os.path.basename(pdf_file_name))[0] + ".md"
    #pdf_file_path = "data/raw_pdf/" + pdf_file_name

    if not raw_pdf_path.exists():
        logger.error(f"変換元ファイルがないよ: {raw_pdf_path}")
        return

    if not raw_md_path.exists():
        logger.info(f"新たに{raw_md_path}に raw .md を作成します……")
    else:
        logger.info("既存の raw .md ファイルは上書きされます……")

    raw_md_content = pymupdf4llm.to_markdown(raw_pdf_path)
    raw_md_path.write_text(raw_md_content, encoding="utf-8")
    logger.debug(f"変換後の.mdファイルを保存しました。: {raw_md_path}")

def clean_markdown(raw_md_path: Path, cleaned_md_path: Path) -> None:
    """変換した markdown を LLM で整形する関数"""

    if cleaned_md_path.exists():
        logger.debug("既に存在する整形後の.mdファイルは、上書きされます...")

    client = genai.Client(api_key=GEMINI_API_KEY_FREE)

    raw_md_content = raw_md_path.read_text(encoding="utf-8")

    prompt = build_cleaning_prompt(PREPROCESS_PROMPT_DIR / "cleaning_template.md", raw_md_content)

    # token数のカウント
    total_tokens = client.models.count_tokens(
        model=CLEANING_MODEL,
        contents=prompt
    )
    logger.debug(f"total入力tokens: {total_tokens.total_tokens}")

    # 整形処理
    interaction = client.interactions.create(
        model=CLEANING_MODEL,
        input=prompt,
    )
    logger.debug(f"使用したtoken数: {interaction.usage}")

    cleaned_md_content = interaction.output_text

    # ファイル保存
    cleaned_md_path.write_text(cleaned_md_content, encoding="utf-8")

    return