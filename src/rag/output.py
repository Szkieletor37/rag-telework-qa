from pathlib import Path
from typing import Any

from src.config import RESULTS_DIR


def save_experiment_result(question_entry: Any, result: Any):
    """設定（モデル、パラメータ）、質問entry（質問、模範解答等）, 推論結果（検索Top-Kチャンク、回答）を、JSONファイルとして保存する"""

