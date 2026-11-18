"""
RAGの検索のためのベクトル化、インデックス作成等をまとめたモジュール。
"""

from logging import DEBUG, getLogger
from pathlib import Path
from typing import Any

import chromadb
from flatten_dict import flatten
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
import requests
from tqdm import tqdm

from src.config import DB_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

logger = getLogger(__name__)
logger.setLevel(DEBUG)

def _split_fixed_length_custom(text: str, chunk_size: int, chunk_overlap: int):
    """
    固定長チャンキング（自前実装）で text を分割する関数
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def _split_recursive_character(text: str, chunk_size: int, chunk_overlap: int):
    """
    langchain-text-splitterの実装で、区切り文字 separators で再帰的にテキスト分割を行う関数
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n", # 段落区切り
            "\n",   # 改行
            "。",   # 句点
            "！",   # 感嘆符
            "？",   # 疑問符
            "、",   # 読点
            " ",    # 空白
            "",     # 文字単位（優先順位最低）
        ]
    )

    return text_splitter.split_text(text)


def _split_markdown_header(text: str, chunk_size: int, chunk_overlap: int):
    """
    langchain-text-splitter の実装で、Markdown の Header に基づいて text を分割する関数
    """

    # ターゲットとする telework_guide.md の見出しは最大5レベルまで
    # "#", Header 1 は文書全体のタイトルであり、今回は単一文書のみ扱うため除外
    #   複数文書を扱う際は Header 1 も含めたほうがよい
    headers_to_split_on = [
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)

    chunks = markdown_splitter.split_text(text)
    logger.debug(f"_split_markdown_header: chunks: {chunks}")

    return chunks


def split_text(text: str, indexing_config: dict):
    """
    設定された strategy に基づいてチャンキングを行う関数
    """
    strategy = indexing_config["chunking"]["strategy"]
    chunk_size = indexing_config["chunking"]["params"]["chunk_size"]
    chunk_overlap = indexing_config["chunking"]["params"]["chunk_overlap"]

    if strategy == "custom_fixed-length":
        return _split_fixed_length_custom(text, chunk_size, chunk_overlap)
    elif strategy == "langchain_recursive-character":
        return _split_recursive_character(text, chunk_size, chunk_overlap)
    elif strategy == "langchain_markdown-header":
        return _split_markdown_header(text, indexing_config)
    else:
        raise ValueError(f"Unknown Chunking strategy: {strategy}")


def get_embeddings(text: str, win_host: Any, embedding_model: Any):
    """短いテキストについての埋め込みベクトルを生成・取得する関数"""

    r = requests.post(
        f"http://{win_host}:12000/api/embeddings",
        json={"model": embedding_model, "prompt": text}
    )
    data = r.json()

    return data["embedding"]


def generate_index(chroma_collection: Any, win_host: Any, md_path: Path, indexing_config: dict):
    """検索対象の資料のvector indexを構築して、chroma_dbへ格納する関数"""

    # markdown の読み込み
    logger.debug("資料のMarkdownファイルの読み込みを開始")
    md = md_path.read_text(encoding="UTF-8")
    logger.debug("完了: 資料のMarkdownファイルの読み込み")

    # チャンクに分割
    logger.debug("チャンクの分割の開始")
    chunks = split_text(md, indexing_config)

    if indexing_config["debug"] == True:
        chunks = chunks[:5]
        logger.debug(f"注意: デバッグ用に、資料のチャンク数は 5 に制限しています")


    #for i, item in enumerate(chunks):
            #logger.debug(f"--- チャンク No.{i} ---")
            #logger.debug(item)
    logger.debug("完了: チャンクの分割")
    logger.debug(f"\t チャンク数: {len(chunks)}")

    # 各チャンクごとに処理
    for i, chunk in enumerate(tqdm(chunks)):

        # 埋め込みベクタ化
        embed = get_embeddings(chunk, win_host, indexing_config["embedding"]["model_name"])

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


def create_chroma_collection(chroma_client: Any, collection_name: str, metadata: dict):
    """chroma_collection を作成する関数"""

    chroma_collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata=metadata
    )
    logger.debug(f"CREATE: name: {collection_name}\nmetadata:{metadata}")

    return chroma_collection

def get_chroma_collection(chroma_client: Any, collection_name: str):
    """chroma_collection を取得する関数"""
    chroma_collection = chroma_client.get_or_create_collection(
        name=collection_name
    )

    logger.debug(f"GET: name={collection_name}\nmetadata: {chroma_collection.metadata}")

    return chroma_collection

def delete_chroma_collection(chroma_client: Any, collection_name: str):
    """ chroma_collection を削除する関数"""

    chroma_client.delete_collection(name=collection_name)

    return

def build_index(win_host: Any, md_path: Any, indexing_config: dict):

    logger.debug("chroma_client の初期化を開始")
    chroma_client = init_chroma_client(db_path=DB_DIR)

    collection_name = f"local_md_{indexing_config["chunking"]["strategy"]}"

    if collection_name in chroma_client.list_collections():
        delete_chroma_collection(chroma_client, collection_name)

    # 参照時のために metadata を付加
    # nested dict を受け付けないので flatten する
    metadata = flatten(indexing_config, reducer='dot')
    chroma_collection = create_chroma_collection(chroma_client, collection_name, metadata)

    logger.debug(chroma_collection)
    logger.debug("完了: Collection の初期化")

    # インデックスの作成
    logger.debug("資料のインデックス化を開始")
    generate_index(chroma_collection, win_host, md_path, indexing_config)
    logger.debug("完了: 資料のインデックス化")

