import os
from logging import DEBUG, getLogger
from pathlib import Path

import pymupdf4llm
from dotenv import load_dotenv
from google import genai

load_dotenv()

logger = getLogger(__name__)
logger.setLevel(DEBUG)

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_API_KEY_FREE = os.getenv('GEMINI_API_KEY_FREE')

def convert_pdf_to_md(raw_file_path: Path, interim_file_path: Path,
                      processed_file_path: Path) -> None:
    """PDF を markdown に変換し、整形する関数"""
    #md_file_name = os.path.splitext(os.path.basename(pdf_file_name))[0] + ".md"
    #pdf_file_path = "data/raw_pdf/" + pdf_file_name

    if not raw_file_path.exists():
        logger.error(f"変換元ファイルがないよ: {raw_file_path}")
        return

    if not interim_file_path.exists():
        logger.info(f"PDFからMarkdownへの変換を開始します: {raw_file_path}")
        interim_md = pymupdf4llm.to_markdown(raw_file_path)
        interim_file_path.write_text(interim_md, encoding="utf-8")
        logger.info(f"変換後の.mdファイルを保存しました。: {interim_file_path}")
    else:
        logger.info(f"既存のinterim.mdファイルを利用します: {interim_file_path}")
        interim_md = interim_file_path.read_text(encoding="utf-8")


    logger.info("MarkdownをGeminiで整形します...")
    if processed_file_path.exists():
        logger.info("既に存在する整形後の.mdファイルは、上書きされます...")

    processed_md = clean_markdown(interim_md)
    processed_file_path.write_text(processed_md, encoding="utf-8")
    
    logger.info(f"整形後の.mdファイルを作成しました: {processed_file_path}")


def clean_markdown(md_content):
    """変換した markdown を LLM で整形する関数"""
    client = genai.Client(api_key=GEMINI_API_KEY_FREE)

    prompt = f"""
# 役割
あなたはRAG（検索拡張生成）システムの前処理を担当するデータエンジニアです。
PDFから自動変換されたMarkdownテキストを、文章の意味や内容を一切変更せずに整形するタスクを行ってください。

# 目的
テキストの検索精度（Embedding）を高めるため、レイアウトの崩れや不要なノイズを除去し、綺麗で一貫性のあるMarkdown構造に整えます。

# 整形ルール（必読）
1. **原文の完全保持（最重要）**:
   - 本文の単語、文章、表現、数値、記号などを**絶対に追加・削除・言い換え（要約や修正）しないでください**。
   - 誤字脱字に見える部分があっても、原文のまま維持してください。

2. **ノイズおよびレイアウトの修復**:
   - 不自然な途中の改行（文の途中で改行されているもの）を結合し、1つの連続した文章にしてください。
   - ページ番号、ヘッダー、フッターなどの繰り返し表示されるノイズ要素を除去してください。
   - 連続する余分な空白行やスペースを整理してください。

3. **構造化・Markdown修復**:
   - 見出し（`#`, `##`, `###`）のレベルが崩れている場合は、適切な階層構造に修正してください。
   - 箇条書きや番号付きリストのインデントを整えてください。
   - 表（Table）がつぶれている場合は、正しいMarkdownの表形式（`|---|---|`）に修復してください。図表の崩れた特殊記号などは可読できる範囲で整理してください。

4. **禁止事項**:
   - 前置き、後書き、解説などの挨拶文は一切出力しないでください。
   - マークダウンのコードブロック囲み（```markdown 〜 ```）も出力しないでください。

# 入力データ
---
{md_content}
---

# 出力
整形後のMarkdownテキストのみを直接出力してください。
    """

    model = "gemini-3.6-flash"
    
    # token数のカウント
    total_tokens = client.models.count_tokens(
        model=model,
        contents=prompt
    )
    logger.info(f"total入力tokens: {total_tokens.total_tokens}")


    interaction = client.interactions.create(
        model=model,
        input=prompt,
    )
    logger.info(f"使用したtoken数: {interaction.usage}")

    return interaction.output_text
