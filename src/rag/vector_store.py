"""
RAGの検索のためのベクトル化、インデックス作成等をまとめたモジュール。
"""

from logging import DEBUG, getLogger
from pathlib import Path
from typing import Any

import chromadb
import requests
from tqdm import tqdm

from src.config import DB_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

logger = getLogger(__name__)
logger.setLevel(DEBUG)

def split_text(md: str, indexing_config:dict):
    """チャンキングを行う関数
    """

    if indexing_config["chunking"]["strategy"] == "fixed_size":
        chunk_size = indexing_config["chunking"]["chunk_size"]
        chunk_overlap = indexing_config["chunking"]["chunk_overlap"]
        chunks = []
        start = 0
        while start < len(md):
            end = start + chunk_size
            chunks.append(md[start:end])
            start += chunk_size - chunk_overlap
    else:
        raise ValueError("固定チャンキングにしか config が対応していないよ")

    return chunks

def get_embeddings(text: str, win_host: Any, embedding_model: Any):
    """短いテキストについての埋め込みベクトルを生成・取得する関数"""

    r = requests.post(
        f"http://{win_host}:12000/api/embeddings",
        json={"model": embedding_model, "prompt": text}
    )
    data = r.json()

    return data["embedding"]


def generate_index(chroma_collection: Any, win_host: Any, md_path: Path, indexing_config:dict, debug_chunk_len_limit=None):
    """検索対象の資料のvector indexを構築して、chroma_dbへ格納する関数"""

    # markdown の読み込み
    logger.debug("資料のMarkdownファイルの読み込みを開始")
    md = md_path.read_text(encoding="UTF-8")
    logger.debug("完了: 資料のMarkdownファイルの読み込み")

    # チャンクに分割
    logger.debug("チャンクの分割の開始")
    chunks = split_text(md, indexing_config)

    # debug 用に、chunks (list) の長さを制限
    # 本番時は None または十分大きな値にする
    DEBUG_CHUNK_LEN_LIMIT = debug_chunk_len_limit
    if DEBUG_CHUNK_LEN_LIMIT is not None:
        chunks = chunks[:DEBUG_CHUNK_LEN_LIMIT]
        logger.debug(f"注意: デバッグ用に、資料のチャンク数は{DEBUG_CHUNK_LEN_LIMIT}に制限しています")


    #for i, item in enumerate(chunks):
            #logger.debug(f"--- チャンク No.{i} ---")
            #logger.debug(item)
    logger.debug("完了: チャンクの分割")
    logger.debug(f"\t チャンク数: {len(chunks)}")

    # 各チャンクごとに処理
    for i, chunk in enumerate(tqdm(chunks)):

        logger.debug(f"--- チャンク No.{i}を処理 ---")

        # 埋め込みベクタ化
        logger.debug("チャンクの埋め込みベクタ化を開始")
        embed = get_embeddings(chunk, win_host, indexing_config["embedding"]["model_name"])
        logger.debug("完了: チャンクの埋め込みベクタ化")

        # collection に格納
        # TODO: ループ内では list に append して、ループ外で1回の .add() にすることもできる
        chroma_collection.add(
            documents=[chunk],
            embeddings=[embed],
            ids=[f"telework_guide_{i}"]
        )

    logger.debug("完了: 全てのチャンクをべクタ化して Collection に格納")
    
    return

def init_chroma_client(db_path: Path):
    """ chroma_clientを初期化する関数"""

    # chroma_client の初期化
    chroma_client = chromadb.PersistentClient(db_path)

    return chroma_client


def get_or_create_chroma_collection(chroma_client: Any, indexing_config):
    """ chroma_collection を作成または取得する関数 """
    # TODO: 本当は name をハードコードするのやめるべきで、引数で与えるべき？
    chunking_strategy = indexing_config["chunking"]["strategy"]
    chroma_collection = chroma_client.get_or_create_collection(
        name=f"local_md_{chunking_strategy}"
    )

    return chroma_collection

def delete_chroma_collection(chroma_client: Any):
    """ chroma_collection を削除する関数"""

    chroma_client.delete_collection(name="local_md")

    return

def build_index(win_host: Any, md_path: Any, indexing_config: dict):

    logger.debug("chroma_client の初期化を開始")
    chroma_client = init_chroma_client(db_path=DB_DIR)

    #delete_chroma_collection(chroma_client)

    chroma_collection = get_or_create_chroma_collection(chroma_client, indexing_config)

    logger.debug(chroma_collection)
    logger.debug("完了: Collection の初期化")

    # インデックスの作成
    logger.debug("資料のインデックス化を開始")
    generate_index(chroma_collection, win_host, md_path, indexing_config)
    logger.debug("完了: 資料のインデックス化")

