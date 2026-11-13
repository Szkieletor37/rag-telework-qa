import subprocess

from src.config import setup_environment, CLEANED_MD_DIR, TARGET_DOCUMENT_NAME, get_indexing_config
from src.utils import get_win_host
from src.rag.vector_store import build_index


setup_environment()

if __name__ == "__main__":
    # 埋め込みベクトル作成用の Ollama 起動のための情報を取得
    win_host = get_win_host()

    indexing_config = get_indexing_config()

    # インデックス作成
    build_index(win_host, (CLEANED_MD_DIR / (TARGET_DOCUMENT_NAME + ".md")), indexing_config)