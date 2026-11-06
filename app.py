import subprocess
import streamlit as st
import chromadb
import requests

from src.config import setup_environment, DB_DIR, CLEANED_MD_PATH
from src.rag.vector_store import init_chroma_client, get_or_create_chroma_collection, delete_chroma_collection build_index
from src.rag.core import rag_core_pipeline
from src.rag.output import save_experiment_result

setup_environment()

logger = getLogger(__name__)
logger.setLevel(DEBUG)


def main():

    chroma_client = init_chroma_client(DB_DIR)

    chroma_collection = get_or_create_chroma_collection(chroma_client)

    if "collection" not in st.session_state:
        st.session_state.collection = get_or_create_chroma_collection
    
    win_host = subprocess.getoutput("hostname.exe").strip() + ".local"

    st.set_page_config(page_title="Local LLM Chat")

    # サイドバーの設定
    st.sidebar.title("設定")

    """実用を想定した際、不要そうな設定なのでコメントアウト

    model = st.sidebar.text_input("モデル名", value="llama3.1:8b")
    temperature = st.sidebar.slider("temperature", 0.0, 2.0, 0.3, 0.1)
    system_prompt = st.sidebar.text_area(
        "System Prompt",
        "あなたは茶目っ気のあるボクっ娘です。日本語で回答してください。"
    )

    # Wordファイルのアップロード
    uploaded_files = st.sidebar.file_uploader(
        "Wordファイルをアップロード(.docxのみ)",
        type=["docx"],
        accept_multiple_files=True
    )
    """

    if st.sidebar.button("インデックス作成"):
        build_index(win_host, md_path=CLEANED_MD_PATH)

        st.sidebar.success("インデックス作成完了")

    # タイトル
    st.title("Local LLM Chat")

    # 会話の履歴を保持
    if "messages" not in st.session_state:
        st.session_state.messages = []

    """未実装のためコメントアウト
    # 会話の履歴をリセットするボタン
    if st.sidebar.button("会話をリセット"):
        st.session_state.messages = []

        # TODO//マルチターンをリセットする処理
        st.rerun()
    """

    # 会話の履歴を表示
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    query_text = st.chat_input("メッセージを入力")

    if query_text:

        with st.chat_message("user"):
            st.write(prompt)

        result = rag_core_pipeline(query_text, win_host, st.session_state.collection)

        print(f"RAGの結果: {result}")
        with st.chat_message('assistant'):
            st.write(f"RAGの結果: {result["retrieved_contexts"]}")

        """
        print(f"results[\"documents\"]: {results["documents"]}")
        print(bool(results["documents"]))

        context_text = "\n".join(results["documents"][0]) if results["documents"] else ""
        if context_text.strip():
            rag_prompt = f"""
            """
            以下は関連ドキュメントの抜粋です。
            {context_text}
            この情報を参考に以下の質問に答えてください。
            {prompt}
            """
            """
            final_user_prompt = rag_prompt
        else:
            final_user_prompt = prompt

        print(f"final_user_prompt: {final_user_prompt}")

        st.session_state.messages.append({"role": "user", "content": final_user_prompt})

        if system_prompt.strip():
            messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages
        else:
            messages = st.session_state.messages

        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True
        )
        """

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
        save_experiment_result(evaluation_dataset, results, params)
 
        with st.chat_message("assistant"):
            st.write(result["response_text"])
        
        st.session_state.messages.append({"role": "assistant", "content": result["response_text"]})



if __name__ == "__main__":
    main()