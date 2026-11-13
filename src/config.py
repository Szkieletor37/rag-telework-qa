import hashlib
from logging import basicConfig, getLogger
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

# プロジェクトルートの定義
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 各ディレクトリの定義
DATA_DIR = PROJECT_ROOT / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdf"
RAW_MD_DIR = DATA_DIR / "raw_md"
CLEANED_MD_DIR = DATA_DIR / "cleaned_md"
QUESTION_DIR = DATA_DIR / "question"
RESULTS_DIR = DATA_DIR / "results"
RESULTS_TMP_DIR = RESULTS_DIR / "tmp"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
INFERENCE_PROMPT_DIR = PROMPTS_DIR / "inference"
PREPROCESS_PROMPT_DIR = PROMPTS_DIR / "preprocess"

# ターゲット資料のファイル名
# 一部はソースコードのどこかでハードコードされているはず。要修正
# もし、ファイルをアップロード可能にする場合は動的に命名するように変更する
TARGET_DOCUMENT_NAME = "telework_guide"

# DBのパスの定義
DB_DIR = PROJECT_ROOT / "chroma_db"

# 推論用プロンプトファイルのパス
INFERENCE_PROMPT_PATH = INFERENCE_PROMPT_DIR / "prompt_template.md"

# 環境変数
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_API_KEY_FREE = os.getenv('GEMINI_API_KEY_FREE')

# 前処理時のパラメータ
CLEANING_MODEL = "gemini-3.6-flash"

# 推論時のパラメータ
INFERENCE_MODEL = "gemini-3.5-flash-lite"
INFERENCE_MODEL_TEMPERATURE = 0.0

# 検索関連のパラメータ
TOP_K = 3
EMBEDDING_MODEL = "nomic-embed-text"

# 固定チャンク長のときのチャンク関連のパラメータ
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50

def get_prompt_metadata(prompt_path: Path) -> dict:
    content = prompt_path.read_text(encoding="utf-8")
    # 内容のSHA-256ハッシュを計算（最初の12桁を使用）
    prompt_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return {
        "template_id": prompt_path.name,
        "template_hash": prompt_hash,
        "system_prompt_template": content,
    }


def get_default_config() -> dict:

    return {
        "generation": {
            "model_name": INFERENCE_MODEL,
            #"temperature": INFERENCE_MODEL_TEMPERATURE,
            "prompt": get_prompt_metadata(INFERENCE_PROMPT_PATH)
        },
        "chunking": {
            "strategy": "fixed_size",
            "params": {
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
            }
        },
        "retrieval": {
            "top_k": TOP_K,
            "embedding_model": EMBEDDING_MODEL,
        },
    }


def setup_environment() -> None:
    """環境変数の設定とルートロガーの設定を行う関数"""

    # ロガーの設定
    FORMATTER = '[%(levelname)s %(asctime)s] [%(funcName)s] - %(message)s'
    # basicConfig は何も設定されてないときのみ有効
    basicConfig(format=FORMATTER)


