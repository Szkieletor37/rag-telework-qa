import os
from logging import basicConfig, getLogger
from pathlib import Path

from dotenv import load_dotenv


# プロジェクトルートの定義
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 各ディレクトリの定義
DATA_DIR = PROJECT_ROOT / "data"
CLEANED_MD_DIR = DATA_DIR / "cleaned_md"
QUESTION_DIR = DATA_DIR / "question"
PROMPT_DIR = PROJECT_ROOT / "prompt"

# 環境変数
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_API_KEY_FREE = os.getenv('GEMINI_API_KEY_FREE')


def setup_environment() -> None:
    """環境変数の設定とルートロガーの設定を行う関数"""
    load_dotenv()

    # ロガーの設定
    FORMATTER = '[%(levelname)s %(asctime)s] - %(message)s'
    # basicConfig は何も設定されてないときのみ有効
    basicConfig(format=FORMATTER)


