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

    # TODO: return chroma_client.get_or_create_collection()
    pass