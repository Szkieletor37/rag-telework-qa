from logging import getLogger, DEBUG
import subprocess
import streamlit as st
import chromadb
import requests

from src.config import setup_environment, DB_DIR, CLEANED_MD_PATH, INFERENCE_MODEL, EMBEDDING_MODEL, TOP_K, CHUNK_OVERLAP, CHUNK_SIZE
from src.rag.vector_store import init_chroma_client, get_or_create_chroma_collection, delete_chroma_collection, build_index
from src.rag.core import rag_core_pipeline
from src.rag.output import save_experiment_result

setup_environment()

logger = getLogger(__name__)
logger.setLevel(DEBUG)


def main():

    chroma_client = init_chroma_client(DB_DIR)

    chroma_collection = get_or_create_chroma_collection(chroma_client)

    if "collection" not in st.session_state:
        st.session_state.collection = get_or_create_chroma_collection(chroma_client)
    
    win_host = subprocess.getoutput("hostname.exe").strip() + ".local"

    st.set_page_config(page_title="Local LLM Chat")

    # サイドバーの設定
    st.sidebar.title("設定")

    if st.sidebar.button("インデックス作成"):
        build_index(win_host, md_path=CLEANED_MD_PATH)

        st.sidebar.success("インデックス作成完了")

    # タイトル
    st.title("Local LLM Chat")

    # 会話の履歴を保持
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 会話の履歴を表示
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    query_text = st.chat_input("メッセージを入力")

    if query_text:

        with st.chat_message("user"):
            st.write(query_text)

        result = rag_core_pipeline(query_text, win_host, st.session_state.collection)

        print(f"RAGの結果: {result}")
        with st.chat_message('assistant'):
            st.write(f"RAGの結果: {result["retrieved_contexts"]}")

        evaluation_dataset = [{
            "execution_mode": "streamlit",
            "question_id": None,
            "category": None,
            "question": query_text,
            "ground_truth": None,
            "ideal_chunks": [
                {
                "chunk_text": None, 
                },
                {
                "chunk_text": None,
                }
            ]
        }]
        params = {
            "embedding_model": EMBEDDING_MODEL,
            "inference_model": INFERENCE_MODEL,
            "chunking_strategy": "fixed_size",
            "chunking_params": {
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
            },
            "top_k": TOP_K
        }
        save_experiment_result(evaluation_dataset, [result], params)
 
        with st.chat_message("assistant"):
            st.write(result["response_text"])
        
        st.session_state.messages.append({"role": "assistant", "content": result["response_text"]})



if __name__ == "__main__":
    main()