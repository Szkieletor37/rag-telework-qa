"""
RAGの主要な処理（生成、検索、保存）を行うモジュール
"""

from pathlib import Path
from typing import Any

from vector_store import generate_index, get_embeddings, init_chroma_collection

DB_DIR = Path("./chroma_db")
CLEANED_MD_PATH = Path("./data/cleaned_md/telework_guide.md")

def retrieve_chunks(query_embeddings: Any, top_k: int) -> Any:
    """ベクトル検索を行う関数"""

    # TODO: collection.query()を用いて検索を行う？
    # TODO: 引数の型がわからん
    pass

def generate_answer(model: Any, retrieved_contexts: Any, system_prompt: str, base_prompt: str)->Any:
    """modelに回答を生成させる関数"""

    # TODO: 型が適当すぎる
    # TODO: Gemini APIを用いる

    pass

def save_experiment_result(result: Any):
    """実験結果（質問、回答、context, 設定）を、JSONファイルとして保存する"""
    pass


def rag_core_pipeline(win_host: Any, query: str)->Any:
    """
    """

    # TODO: 
    #   * DB_DIR はどのファイルで宣言すべき？
    #   * DB_DIR は rag_core_pipeline()は引数として受け取るべき？

    # chroma_collection の初期化
    chroma_collection = init_chroma_collection(db_path=DB_DIR)

    # インデックスの作成
    generate_index(chroma_collection, win_host, md_path=CLEANED_MD_PATH)

    # 質問クエリを埋め込みベクトル化
    query_embedded = get_embeddings(query, win_host)

    # 検索
    retrieved_contexts = retrieve_chunks(query_embedded)

    # モデルに検索結果を渡し、推論させる
    result = generate_answer(retrieved_contexts)

    # 回答を保存
    save_experiment_result(result)

    # Webから呼んだ場合は結果を返して出力？
    # return result



