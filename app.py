from logging import getLogger, DEBUG
import subprocess
import streamlit as st
import chromadb
import requests

from src.config import setup_environment, DB_DIR, CLEANED_MD_DIR, TARGET_DOCUMENT_NAME, get_default_config, RESULTS_TMP_DIR
from src.utils import get_win_host
from src.rag.vector_store import init_chroma_client, get_or_create_chroma_collection, delete_chroma_collection, build_index
from src.rag.core import rag_core_pipeline
from src.rag.output import save_experiment_result, generate_metadata

setup_environment()

logger = getLogger(__name__)
logger.setLevel(DEBUG)


def main():

    config = get_default_config()

    chroma_client = init_chroma_client(DB_DIR)

    chroma_collection = get_or_create_chroma_collection(chroma_client)

    if "collection" not in st.session_state:
        st.session_state.collection = chroma_collection
    
    win_host = get_win_host()

    # タイトル
    st.set_page_config(page_title="Local LLM Chat")

    # サイドバーの設定
    st.sidebar.title("設定")

    if st.sidebar.button("インデックス作成"):
        build_index(win_host, md_path=(CLEANED_MD_DIR / (TARGET_DOCUMENT_NAME + ".md")))

        st.sidebar.success("インデックス作成完了")


    # 会話の履歴を保持
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 会話の履歴をリセットするボタン
    if st.sidebar.button("会話をリセット"):
        st.session_state.messages = []
        st.rerun()

    # 会話の履歴を表示
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    user_query_text = st.chat_input("メッセージを入力")

    if user_query_text:

        with st.chat_message("user"):
            st.write(user_query_text)

        st.session_state.messages.append({"role": "user", "content": user_query_text})

        result = rag_core_pipeline(user_query_text, win_host, st.session_state.collection)

        logger.debug(f"RAGの結果: {result}")
        # 検索チャンクを画面に表示する？
        #with st.chat_message('assistant'):
        #    st.write(f"RAGの結果: {result["retrieved_contexts"]}")

        metadata = generate_metadata(execution_source="streamlit", description="test")

        evaluation_dataset = [{
            "question_id": None,
            "category": None,
            "question": user_query_text,
            "ground_truth": None,
            "ideal_source": [{
                "source": None,
                "text": None,
            }]
        }]

        # Streamlit appからの実行は .../results/tmp/ に保存する
        save_experiment_result(evaluation_dataset, [result], metadata, config, RESULTS_TMP_DIR)
 
        with st.chat_message("assistant"):
            st.write(result["generated_answer"])
        
        st.session_state.messages.append({"role": "assistant", "content": result})



if __name__ == "__main__":
    main()