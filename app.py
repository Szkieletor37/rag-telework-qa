import subprocess
import streamlit as st
from openai import OpenAI
import chromadb
from docx import Document
import requests

DB_DIR = "./chroma_db"

def ollama_embed(text, win_host):
    r = requests.post(
        f"http://{win_host}:12000/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text}
    )
    data = r.json()
    print(f"data = {data}")
    return data["embedding"]

def load_word_document(file):
    """wordファイルを読み込む関数
    """
    return "\n".join(para.text for para in Document(file).paragraphs)

def split_text(text):
    """ チャンキングする関数
    """
    chunk_size = 200
    overlap = 50
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

def main():

    chroma_client = chromadb.PersistentClient(path=DB_DIR)

    if "collection" not in st.session_state:
        st.session_state.collection = chroma_client.get_or_create_collection(
            name="local_docs"
        )
    
    win_host = subprocess.getoutput("hostname.exe").strip() + ".local"

    # Ollamaからインストールしたモデルを使ったベクトル化関数

    # Wordファイルを読み込む関数

    # チャンキング関数
    
    st.set_page_config(page_title="Local LLM Chat")

    # サイドバーの設定
    st.sidebar.title("設定")
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

    if st.sidebar.button("インデックス作成"):
        for file in uploaded_files:
            text = load_word_document(file)
            chunks = split_text(text)
            for i, chunk in enumerate(chunks):
                embed = ollama_embed(chunk, win_host)
                st.session_state.collection.add(
                    documents=[chunk],
                    embeddings=[embed],
                    ids=[f"{file.name}_{i}"]
                )
        st.sidebar.success("インデックス作成完了")

    # タイトル
    st.title("Local LLM Chat")

    # 会話の履歴を保持
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 会話の履歴をリセットするボタン
    if st.sidebar.button("会話をリセット"):
        st.session_state.messages = []

        try:
            chroma_client.delete_collection(name="local_docs")
        except Exception:
            pass
        st.session_state.collection = chroma_client.get_or_create_collection(name="local_docs")
        st.rerun()

    # 会話の履歴を表示
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    prompt = st.chat_input("メッセージを入力")

    client=OpenAI(
        api_key="hogehoge",
        # ollamaサーバーを指定
        base_url=f"http://{win_host}:12000/v1"
    )

    if prompt:

        with st.chat_message("user"):
            st.write(prompt)

        query_embed = ollama_embed(prompt, win_host)
        results = st.session_state.collection.query(
            query_embeddings=[query_embed],
            n_results=2
        )

        print(f"RAGの結果: {results}")
        with st.chat_message('assistant'):
            st.write(f"RAGの結果: {results}")

        print(f"results[\"documents\"]: {results["documents"]}")
        print(bool(results["documents"]))

        context_text = "\n".join(results["documents"][0]) if results["documents"] else ""
        if context_text.strip():
            rag_prompt = f"""
            以下は関連ドキュメントの抜粋です。
            {context_text}
            この情報を参考に以下の質問に答えてください。
            {prompt}
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
        with st.chat_message("assistant"):
            placeholder = st.empty()
            stream_response = ""
            for chunk in stream:
                stream_response += chunk.choices[0].delta.content
                placeholder.write(stream_response)

        
        st.session_state.messages.append({"role": "assistant", "content": stream_response})



if __name__ == "__main__":
    main()