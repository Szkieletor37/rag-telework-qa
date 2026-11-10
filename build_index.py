import subprocess

from src.config import setup_environment, CLEANED_MD_DIR, TARGET_DOCUMENT_NAME
from src.utils import get_win_host
from src.rag.vector_store import build_index


setup_environment()

if __name__ == "__main__":
    # 埋め込みベクトル作成用の Ollama 起動のための情報を取得
    win_host = get_win_host()

    # インデックス作成
    build_index(win_host, md_path=(CLEANED_MD_DIR / (TARGET_DOCUMENT_NAME + ".md")))