"""
RAGの検索のためのベクトル化、インデックス作成等をまとめたモジュール。
"""

from logging import DEBUG, getLogger
from pathlib import Path
from typing import Any

import chromadb
import requests
from tqdm import tqdm

logger = getLogger(__name__)
logger.setLevel(DEBUG)

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

def get_embeddings(text: str, model: Any, win_host: Any):
    """短いテキストについての埋め込みベクトルを生成・取得する関数"""

    r = requests.post(
        f"http://{win_host}:12000/api/embeddings",
        json={"model": model, "prompt": text}
    )
    data = r.json()

    return data["embedding"]


def generate_index(chroma_collection: Any, win_host: Any, embedding_model:Any, md_path: Path):
    """検索対象の資料のvector indexを構築して、chroma_dbへ格納する関数"""

    # markdown の読み込み
    logger.info("資料のMarkdownファイルの読み込みを開始")
    md = md_path.read_text(encoding="UTF-8")
    logger.info("完了: 資料のMarkdownファイルの読み込み")

    # チャンクに分割
    logger.info("チャンクの分割の開始")
    chunks = split_text(md)

    # debug 用に、chunks (list) の長さを制限
    # 本番時は None または十分大きな値にする
    DEBUG_CHUNK_LEN_LIMIT = 3
    if DEBUG_CHUNK_LEN_LIMIT is not None:
        chunks = chunks[:DEBUG_CHUNK_LEN_LIMIT]
        logger.info(f"注意: デバッグ用に、資料のチャンク数は{DEBUG_CHUNK_LEN_LIMIT}に制限しています")


    for i, item in enumerate(chunks):
            logger.debug(f"--- チャンク No.{i} ---")
            logger.debug(item)
    logger.info("完了: チャンクの分割")
    logger.debug(f"\t チャンク数: {len(chunks)}")

    # 各チャンクごとに処理
    # TODO: tqdm で進捗バーを表示できると便利
    for i, chunk in enumerate(tqdm(chunks)):

        logger.debug(f"--- チャンク No.{i}を処理 ---")

        # 埋め込みベクタ化
        logger.info("チャンクの埋め込みベクタ化を開始")
        embed = get_embeddings(chunk, embedding_model, win_host)
        logger.info("完了: チャンクの埋め込みベクタ化")

        # collection に格納
        chroma_collection.add(
            documents=[chunk],
            embeddings=[embed],
            ids=[f"telework_guide_{i}"]
        )

    logger.info("完了: 全てのチャンクをべクタ化して Collection に格納")
    
    return

def init_chroma_collection(db_path: Path):
    """ chroma_clientを初期化して、collectionを作成する関数"""

    # chroma_client の初期化
    chroma_client = chromadb.PersistentClient(db_path)

    # collection の作成
    chroma_collection = chroma_client.get_or_create_collection(
        name="local_md"
    )

    return chroma_collection