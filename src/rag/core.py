"""
RAGの主要な処理（生成、検索、保存）を行うモジュール
"""

from logging import DEBUG, getLogger
from pathlib import Path
from typing import Any
import re

from src.config import GEMINI_API_KEY
from src.rag.vector_store import get_embeddings

from google import genai
from google.genai.types import GenerateContentConfig
from jinja2 import Template

logger = getLogger(__name__)
logger.setLevel(DEBUG)

def retrieve_chunks(collection: Any, query_embeddings: Any, top_k: int) -> Any:
    """複数の query を一括で受け付け、ベクトル検索を行う関数"""

    retrieved_chunks = collection.query(
        query_embeddings=query_embeddings,
        n_results=top_k
    )

    logger.debug(retrieved_chunks)

    retrieved_contexts = []

    # 各要素は2重リストになっている
    for i, _ in enumerate(retrieved_chunks["ids"]):
        retrieved_contexts_per_query = []

        # rank は 1-index にする
        for j in range(top_k):
            retrieved_contexts_per_query.append({
                "chunk_id": retrieved_chunks["ids"][i][j],
                "rank": j+1,
                "score": retrieved_chunks["distances"][i][j], 
                "text": retrieved_chunks["documents"][i][j]
            })

        retrieved_contexts.append(retrieved_contexts_per_query)

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

def generate_answer(inference_prompt: str, inference_model: Any, temperature: Any)->Any:
    """modelに回答を生成させる関数"""

    client = genai.Client(api_key=GEMINI_API_KEY)

    raw_response = client.models.generate_content(
        model=inference_model,
        contents=inference_prompt,
        config=GenerateContentConfig(
            max_output_tokens=2000,
            temperature=temperature
        )
    )

    logger.debug(
        f"""使用token: 
        prompt:  {raw_response.usage_metadata.prompt_token_count}
        response:{raw_response.usage_metadata.candidates_token_count}
        total:   {raw_response.usage_metadata.total_token_count}"""
    )

    thought_process, generated_answer = parse_llm_response(raw_response.text)

    return thought_process, generated_answer


def rag_core_pipeline(query_texts: Any, win_host: Any, chroma_collection: Any, config: dict)->Any:
    """
    RAGのコアの部分を処理する関数
    """

    # 質問クエリを埋め込みベクトル化
    logger.debug("質問クエリのベクタ化を開始")
    logger.debug(f"type : {type(query_texts)}, value: {query_texts}")
    query_embeddeds = get_embeddings(query_texts, win_host, embedding_model=config["retrieval"]["embedding_model"])
    logger.debug("完了: 質問クエリのベクタ化")
    logger.debug(f"{type(query_embeddeds)}")

    # 検索
    logger.debug("chroma_db をベクトル検索して、上位 TOP_K 件のチャンクを取得")
    retrieved_contexts = retrieve_chunks(chroma_collection, query_embeddeds, top_k=config["retrieval"]["top_k"])
    logger.debug(f"完了: chroma_db をベクトル検索して、上位 TOP_K: {config["retrieval"]["top_k"]} 件のチャンクを取得")

    logger.debug(f"retrieved_contexts: {retrieved_contexts}")

    results = []

    for query_text, retrieved_contexts_per_query in zip(query_texts, retrieved_contexts):

        # 推論させるためのプロンプトを組み立てる
        inference_prompt_template = config["generation"]["prompt"]["system_prompt_template"]
        inference_prompt_rendered = build_inference_prompt(inference_prompt_template, query_text, retrieved_contexts_per_query)

        logger.debug("推論用プロンプト:")
        logger.debug(inference_prompt_rendered)

        # モデルに質問文と検索結果を埋め込んだプロンプトを渡し、推論させる
        thought_process, generated_answer = generate_answer(inference_prompt_rendered, config["generation"]["model_name"], config["generation"]["temperature"])

        result = {
            "rendered_prompt": inference_prompt_rendered,
            "thought_process": thought_process,
            "generated_answer": generated_answer,
            "retrieved_contexts": retrieved_contexts_per_query
        }
        results.append(result)

    return results
