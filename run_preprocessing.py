from logging import DEBUG, basicConfig, getLogger
from pathlib import Path

from src.config import setup_environment, RAW_MD_DIR, RAW_PDF_DIR, CLEANED_MD_DIR, TARGET_DOCUMENT_NAME
from src.preprocess import convert_pdf_to_md

setup_environment()

logger = getLogger(__name__)
logger.setLevel(DEBUG)

def main():

    # TODO: src.config から Path を import すべき
    # TODO: convert_pdf_to_md() は convert と cleaning に分割すべき
    convert_pdf_to_md(Path("data/raw_pdf/telework_guide.pdf"), Path("data/raw_md/telework_guide.md"), Path("data/cleaned_md/telework_guide.md"))
    convert_pdf_to_md(RAW_PDF_DIR / (TARGET_DOCUMENT_NAME + ".pdf"), RAW_MD_DIR / (TARGET_DOCUMENT_NAME + ".md"))
    logger.debug("完了: raw PDF -> raw Markdownの変換")

    logger.debug("raw MarkdownのGeminiで整形し、最終参照 Markdown を生成します")
    clean_markdown(RAW_MD_DIR / (TARGET_DOCUMENT_NAME + ".md"), CLEANED_MD_DIR / (TARGET_DOCUMENT_NAME + ".md"))

    logger.debug("完了: raw MarkdownのGeminiで整形し、最終参照 Markdown を生成")

if __name__ == "__main__":
    main()