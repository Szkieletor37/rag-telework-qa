"""
cliからYAMLファイルを読み込み、質問を送付して回答を受け取るモジュール
"""

import argparse
import subprocess

from typing import Any
from pathlib import Path

def make_parser() -> Any:
    """parserをargparseで作成する関数"""
    pass

def parse_yaml(yaml_path: Path) -> Any:
    """yamlファイルを構造体に格納する関数？"""
    # TODO: これいる？
    pass
    
def evaluate_questions(queries: Any):
    """質問を１問ずつrag::rag_core_pipeline()に送付する関数"""

    pass

    for query in queries:
        result = rag_core_pipeline(query)

def cli_eval() -> Any:
    """CLIでYAMLファイルの質問を読み込み、1問ずつRAGに処理させて回答を保存する関数"""
    args = make_parser().parse_args(argv)

    queries = parse_yaml(args.yaml)

    # 埋め込みベクトル作成用の Ollama 起動のための情報を取得
    win_host = subprocess.getoutput("hostname.exe").strip() + ".local"
    
    evaluate_questions(win_host, queries)
