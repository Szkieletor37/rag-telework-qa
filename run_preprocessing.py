from logging import DEBUG, basicConfig, getLogger
from pathlib import Path

from src.config import setup_environment
from src.preprocess import convert_pdf_to_md

setup_environment()

logger = getLogger(__name__)
logger.setLevel(DEBUG)

def main():

    # TODO: src.config から Path を import すべき
    # TODO: convert_pdf_to_md() は convert と cleaning に分割すべき
    convert_pdf_to_md(Path("data/raw_pdf/telework_guide.pdf"), Path("data/raw_md/telework_guide.md"), Path("data/cleaned_md/telework_guide.md"))

if __name__ == "__main__":
    main()