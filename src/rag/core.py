"""
RAGの主要な処理（生成、検索、保存）を行うモジュール
"""

from logging import DEBUG, getLogger
from pathlib import Path
from typing import Any
import re

from src.config import EMBEDDING_MODEL, INFERENCE_MODEL, GEMINI_API_KEY, TOP_K, INFERENCE_PROMPT_PATH
from src.rag.vector_store import get_embeddings

from google import genai
from google.genai.types import GenerateContentConfig
from jinja2 import Template

logger = getLogger(__name__)
logger.setLevel(DEBUG)

def retrieve_chunks(collection: Any, query_embeddings: Any, top_k: int) -> Any:
    """ベクトル検索を行う関数"""

    # TODO: 複数の query を一括で受付可能なので、一括で渡すように実装を変更する？
    #   * そうすると全体の設計の見直しが必要かも
    #   * そもそも不要かも
    # TODO: 引数の型がわからん
    #   単一の埋め込みなのかリストなのか
    retrieved_chunks = collection.query(
        query_embeddings=[query_embeddings],
        n_results=top_k
    )

    logger.debug(retrieved_chunks)

    retrieved_contexts = []

    # 各要素は2重リストになっている
    # 現在の実装だと 1 query() あたり1個のみembeddingを渡しているので range(0)
    for i in range(1):

        # rank は 1-index にする
        for j, id in enumerate(retrieved_chunks["ids"][i]):
            retrieved_contexts.append({
                "chunk_id": id,
                "rank": j+1,
                "score": retrieved_chunks["distances"][i][j], 
                "text": retrieved_chunks["documents"][i][j]
            })

    return retrieved_contexts

def build_inference_prompt(prompt_template: str, query_text: str, retrieved_contexts: Any)->Any:
    """テンプレートプロンプト、質問文、検索結果から、推論用プロンプトを組み立てる関数"""

    logger.debug(retrieved_contexts)

    documents_section = "\n".join(
        f"- [Doc {i+1}]\n{retrieved_context["text"]}" for i, retrieved_context in enumerate(retrieved_contexts)
    )

    template = Template(prompt_template)

    return template.render(
        query_text=query_text,
        documents_section=documents_section
    )

def parse_llm_response(raw_response: str):
    """モデルの推論回答をXMLタグで parse する関数"""

    thought_match = re.search(r"<thought>(.*?)</thought>", raw_response, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", raw_response, re.DOTALL)

    thought_process = thought_match.group(1).strip() if thought_match else ""
    generated_answer = answer_match.group(1).strip() if answer_match else raw_response

    return thought_process, generated_answer

def generate_answer(inference_prompt: str, inference_model: Any)->Any:
    """modelに回答を生成させる関数"""

    client = genai.Client(api_key=GEMINI_API_KEY)

    raw_response = client.models.generate_content(
        model=inference_model,
        contents=inference_prompt,
        config=GenerateContentConfig(
            max_output_tokens=2000,
            temperature=0.1
        )
    )

    logger.debug(
        f"""使用token: 
        prompt:  {raw_response.usage_metadata.prompt_token_count}
        response:{raw_response.usage_metadata.candidates_token_count}
        total:   {raw_response.usage_metadata.total_token_count}"""
    )

    thought_process, generated_answer = parse_llm_response(raw_response)

    return thought_process, generated_answer


def rag_core_pipeline(query_text: Any, win_host: Any, chroma_collection: Any)->Any:
    """RAGのコアの部分を処理する関数
    """

    # TODO: 
    #   * DB_DIR はどのファイルで宣言すべき？
    #   * DB_DIR は rag_core_pipeline()は引数として受け取るべき？

    # 質問クエリを埋め込みベクトル化
    logger.debug("質問クエリのベクタ化を開始")
    logger.debug(f"type : {type(query_text)}, value: {query_text}")
    query_embedded = get_embeddings(query_text, win_host, embedding_model=EMBEDDING_MODEL)
    logger.debug("完了: 質問クエリのベクタ化")

    # 検索
    # [["chunk1...", "chunk2...", "chunk3..."]] と2重リストで返ってくるので、[0]で内側のリストを取り出す
    logger.debug("chroma_db をベクトル検索して、上位 TOP_K 件のチャンクを取得")
    retrieved_contexts = retrieve_chunks(chroma_collection, query_embedded, top_k=TOP_K)
    logger.debug("完了: chroma_db をベクトル検索して、上位 TOP_K 件のチャンクを取得")
    logger.debug(f"TOP_K: {TOP_K}")

    # 順位は 1-index で確認したいので i+1 にする
    logger.debug(f"retrieved_contexts: {retrieved_contexts}")

    # 推論させるためのプロンプトを組み立てる
    inference_prompt_template = INFERENCE_PROMPT_PATH.read_text(encoding="utf-8")
    inference_prompt_rendered = build_inference_prompt(inference_prompt_template, query_text, retrieved_contexts)

    logger.debug("推論用プロンプト:")
    logger.debug(inference_prompt_rendered)

    # モデルに検索結果を渡し、推論させる
    thought_process, generated_answer = generate_answer(inference_prompt_rendered, inference_model=INFERENCE_MODEL)

    result = {
        "rendered_prompt": inference_prompt_rendered,
        "thought_process": thought_process,
        "generated_answer": generated_answer,
        "retrieved_contexts": retrieved_contexts
    }


    # Webから呼んだ場合は結果を返して出力？
    return result
