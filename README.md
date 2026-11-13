# README.md

RAG の学習のために作成。
PDF を対象に query を検索して、回答する。

## 仕様
* 複数のファイルによるインデックス構築には現状非対応

## 準備
* `.env` で、GEMINI_API_KEY(_FREE) を定義する。
  * デフォルトでは Gemini を前処理・推論で使用するようになっている。
* `data/raw_pdf/` 配下に、検索対象の PDF を配置してください。
* Powershell で `ollama serve` しておく。
  * ```Powershell
    ollama serve
    ```

## 前処理
```sh
uv run run_preprocessing.py
```
* `pymupdf4llm` により Markdown に変換され、`data/raw_md/`配下に .md が保存されます。
* 続けて整形を行い、`data/cleaned_md/`配下に保存されます。
  * `src/config.py` の `CLEANING_MODEL` で指定したモデルで、整形を行います。

## インデックス構築
```sh
uv run build_index.py
```

## CLI実行
```sh
uv run cli_eval.py
```

## Web application 実行
```sh
uv run streamlit run app.py
```