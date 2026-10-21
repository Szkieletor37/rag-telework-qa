"""
RAGの検索のためのベクトル化、インデックス作成等をまとめたモジュール。
"""

from pathlib import Path
from typing import Any

import chromadb

def split_text(md: str):
    """チャンキングを行う関数
    """
    chunk_size = 200
    overlap = 50
    chunks = []
    start = 0
    while start < len(md):
        end = start + chunk_size
        chunks.append(md[start:end])
        start += chunk_size - overlap

    return chunks

def generate_index(chroma_collection: Any, md_path: Path):
    """検索対象の資料のvector indexを構築して、chroma_dbへ格納する関数"""

    # markdown の読み込み
    md = md_path.read_text(encoding="UTF-8")

    # チャンクに分割
    chunks = split_text(md)

    # 各チャンクごとに処理
    for i, chunk in enumerate(chunks):

        # ベクタ化
        embed = get_embeddings(chunk, win_host)

        # collection に格納
        chroma_collection.add(
            documents=[chunk],
            embeddings=[embed],
            ids=[f"telework_guide_{i}"]
        )
    
    pass

def get_embeddings(text: str):
    """埋め込みベクトルを生成・取得する関数"""
    # TODO: udemy3.py::ollama_embed()の内容を実装する
    pass

def init_chroma_collection(db_path: Path):
    """ chroma_clientを初期化して、collectionを作成する関数"""

    # chroma_client の初期化
    chroma_client = chromadb.PersistentClient(db_path)

    # collection の作成
    chroma_collection = chroma_client.get_or_create_collection(
        name="local_md"
    )

    return chroma_collection