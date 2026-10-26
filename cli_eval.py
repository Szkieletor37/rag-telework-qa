"""
cliからYAMLファイルを読み込み、質問を送付して回答を受け取るモジュール
"""

import argparse
import subprocess
import yaml

from logging import getLogger, DEBUG
from typing import Any
from pathlib import Path

from src.config import setup_environment
from src.rag.core import rag_core_pipeline

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
    
def evaluate_questions(queries: Any):
    """質問を１問ずつrag::rag_core_pipeline()に送付する関数"""

    # TODO: 
    #   現在、埋め込みベクトルの作成 -> 検索 -> 回答 を1問ごとにやっている
    #       この状態だと、Ollamaとの通信（埋め込み時）、APIとの通信（回答時）に時間がかかる恐れあり
    #   将来的には、複数の質問Ｎ問をまとめてバッチ処理的に回答させたい場合、N問すべての埋め込みベクトルを作成 -> N問すべてを検索し結果を取得 -> N問すべてに回答 とさせたほうが高速化できる
    #
    # とりあえず気にせず、まず動くものを実装する

    for query in queries:
        result = rag_core_pipeline(query)

def cli_eval() -> Any:
    """CLIでYAMLファイルの質問を読み込み、1問ずつRAGに処理させて回答を保存する関数"""
    args = make_parser().parse_args(argv)

    queries = load_yaml(args.yaml)

    # 埋め込みベクトル作成用の Ollama 起動のための情報を取得
    win_host = subprocess.getoutput("hostname.exe").strip() + ".local"
    
    evaluate_questions(win_host, queries)
