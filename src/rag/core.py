"""
RAGの主要な処理（生成、検索、保存）を行うモジュール
"""

from typing import Any

from vector_store import generate_index, get_embeddings, init_collection

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


def rag_pipeline(query: str)->Any:
    # 質問クエリを埋め込みベクトル化
    query_embedded = get_embeddings(query)

    # 検索
    retrieved_contexts = retrieve_chunks(query_embedded)

    # モデルに検索結果を渡し、推論させる
    result = generate_answer(retrieved_contexts)

    # 回答を保存
    save_experiment_result(result)

    # Webから呼んだ場合は結果を返して出力？
    # return result



