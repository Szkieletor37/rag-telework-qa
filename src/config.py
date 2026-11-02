import os
from logging import basicConfig, getLogger
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

# プロジェクトルートの定義
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 各ディレクトリの定義
DATA_DIR = PROJECT_ROOT / "data"
CLEANED_MD_DIR = DATA_DIR / "cleaned_md"
QUESTION_DIR = DATA_DIR / "question"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
DB_DIR = Path("./chroma_db")
CLEANED_MD_PATH = Path("./data/cleaned_md/telework_guide.md")

# 環境変数
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_API_KEY_FREE = os.getenv('GEMINI_API_KEY_FREE')

# モデルの指定
EMBEDDING_MODEL = "nomic-embed-text"
INFERENCE_MODEL = "gemini-3.5-flash-lite"

def setup_environment() -> None:
    """環境変数の設定とルートロガーの設定を行う関数"""

    # ロガーの設定
    FORMATTER = '[%(levelname)s %(asctime)s] [%(funcName)s] - %(message)s'
    # basicConfig は何も設定されてないときのみ有効
    basicConfig(format=FORMATTER)


