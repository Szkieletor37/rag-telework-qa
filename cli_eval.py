"""
cliからYAMLファイルを読み込み、質問を送付して回答を受け取るモジュール
"""

import argparse
from logging import getLogger, DEBUG
from pathlib import Path
from typing import Any
import yaml

from src.config import setup_environment, DB_DIR, QUESTION_DIR, get_generation_and_retrieval_config, RESULTS_DIR, RESULTS_TMP_DIR
from src.utils import get_win_host
from src.rag.vector_store import init_chroma_client, get_chroma_collection
from src.rag.core import rag_core_pipeline
from src.rag.output import save_experiment_result, generate_metadata

setup_environment()

logger = getLogger(__name__)
logger.setLevel(DEBUG)

def make_parser() -> Any:
    """parserをargparseで作成する関数"""
    pass

def load_yaml(yaml_path: Path) -> Any:
    """yamlファイルを読み込み、構造体に変換する関数"""
    with yaml_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def cli_eval() -> Any:
    """CLIでYAMLファイルの質問を読み込み、1問ずつRAGに処理させて回答を保存する関数"""

    # 実験結果に関わるパラメータ（results/.../*.json に保存するもの）
    config = get_generation_and_retrieval_config()

    # 実行時のみ必要な option （*.json に保存しない）
    run_options = {
        "is_test": True,
    }
    #args = make_parser().parse_args(argv)

    logger.debug("cli_eval()の処理を開始します……")

    chroma_client = init_chroma_client(DB_DIR)

    chroma_collection = get_chroma_collection(chroma_client, config["retrieval"]["collection_name"])

    indexing_config = {
        "indexing_config": chroma_collection.metadata
    }

    #queries = load_yaml(args.yaml)
    evaluation_dataset = load_yaml(QUESTION_DIR / "simple.yaml")
    logger.debug(evaluation_dataset)

    # 埋め込みベクトル作成用の Ollama 起動のための情報を取得
    win_host = get_win_host()
    logger.debug(win_host)

    # 1問ずつ rag_core_pipeline() に渡す
    results = []
    for question_entry in evaluation_dataset:
        query_text = question_entry["question"]
        result = rag_core_pipeline(query_text, win_host, chroma_collection, config)
        results.append(result)

    metadata = generate_metadata(execution_source="cli")

    save_experiment_result(evaluation_dataset, results, metadata, (config | indexing_config), run_options)


    logger.debug("完了: cli_eval()のすべての処理が完了しました。")

    return

if __name__ == "__main__":
    cli_eval()
