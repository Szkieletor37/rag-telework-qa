from logging import DEBUG, basicConfig, getLogger
from pathlib import Path

from preprocess import convert_pdf_to_md

logger = getLogger(__name__)
logger.setLevel(DEBUG)

def main():
    logger.info("Hello from rag-telework-qa!")

    convert_pdf_to_md(Path("data/raw_pdf/telework_guide.pdf"), Path("data/raw_md/telework_guide.md"), Path("data/cleaned_md/telework_guide.md"))

if __name__ == "__main__":
    # logging のフォーマット
    FORMATTER = '[%(levelname)s %(asctime)s] - %(message)s'
    # basicConfig は何も設定されてないときのみ有効
    basicConfig(format=FORMATTER)

    main()
