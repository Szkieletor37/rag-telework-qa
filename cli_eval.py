"""
cliからYAMLファイルを読み込み、質問を送付して回答を受け取るモジュール
"""

import argparse
import subprocess
import yaml

from typing import Any
from pathlib import Path

def make_parser() -> Any:
    """parserをargparseで作成する関数"""
    pass

def load_yaml(yaml_path: Path) -> Any:
    """yamlファイルを読み込み、構造体に変換する関数"""
    with yaml_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def evaluate_questions(queries: Any):
    """質問を１問ずつrag::rag_core_pipeline()に送付する関数"""

    pass

    # TODO: 埋め込みベクトルの作成、queries のすべてに対してまとめてやる -> 検索回答を1問ずつやらせる ほうがいいのでは
    #   別プロセスの Ollama との通信に時間がかかりそうな気がする
    for query in queries:
        result = rag_core_pipeline(query)

def cli_eval() -> Any:
    """CLIでYAMLファイルの質問を読み込み、1問ずつRAGに処理させて回答を保存する関数"""
    args = make_parser().parse_args(argv)

    queries = load_yaml(args.yaml)

    # 埋め込みベクトル作成用の Ollama 起動のための情報を取得
    win_host = subprocess.getoutput("hostname.exe").strip() + ".local"
    
    evaluate_questions(win_host, queries)
