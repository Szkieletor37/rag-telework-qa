"""
RAGの検索のためのベクトル化、インデックス作成等をまとめたモジュール。
"""

from pathlib import Path

def generate_index(cleaned_md: str):
    """検索対象の資料のvector indexを構築して、chroma_dbへ格納する関数"""

    # TODO: chunking, vectorize, chroma_dbへの格納を実装
    pass

def get_embeddings(text: str):
    """埋め込みベクトルを生成・取得する関数"""
    # TODO: udemy3.py::ollama_embed()の内容を実装する
    pass

def init_collection(db_path: Path):
    """ chroma_clientを初期化して、collectionを作成する関数"""

    # chroma_client の初期化
    chroma_client = chromadb.PersistentClient(db_path)

    # collection の作成
    chroma_collection = chroma_client.get_or_create_collection(
        name="local_md"
    )

    return chroma_collection