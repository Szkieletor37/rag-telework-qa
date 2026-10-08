"""
RAGの検索のためのベクトル化、インデックス作成等をまとめたモジュール。
"""

from pathlib import Path

def generate_index(file_path: Path):
    """.mdファイルから ollama を用いて埋め込みベクトルを作成する関数"""

    # TODO: udemy3.py::ollama_embed()の内容を実装する
    pass

def init_collection(db_path: Path):
    """ chroma_clientを初期化して、collectionを作成する関数"""

    # TODO: return chroma_client.get_or_create_collection()
    pass