from datetime import datetime
import json
from logging import getLogger, DEBUG
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from tqdm import tqdm

from src.config import RESULTS_DIR

logger = getLogger(__name__)
logger.setLevel(DEBUG)

def generate_metadata(execution_source: str, description="") -> dict:
    """実験結果のメタデータを生成する関数"""

    experiment_id = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("exp_%Y%m%d_%H%M%S")
    created_at = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d_%H:%M:%S+09:00")

    return {
        "experiment_id": experiment_id,
        "created_at": created_at,
        "execution_source": execution_source,
        "description": description
    }



def save_experiment_result(evaluation_dataset: Any, results: Any, metadata: Any, config: Any) -> None:
    """ 評価セット（質問、模範解答等）, 推論結果（検索Top-Kチャンク、回答）, パラメータを、JSONファイルとして保存する"""

    jst_timestamp = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d_%H%M%S")

    dir_path = RESULTS_DIR / jst_timestamp
    dir_path.mkdir(parents=True)

    logger.debug("実験結果のJSONファイル出力を開始します")
    for i, (question_entry, result) in enumerate(tqdm(zip(evaluation_dataset, results))):

        data = {
            "metadata": metadata,
            "config": config,
            "results": [question_entry | result]
        }

        # 出力ファイル名は 1-index にする
        file_path = dir_path / f"result_{question_entry["question_id"]}.json"

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2
            )
    logger.debug(f"完了: 実験結果を{RESULTS_DIR / jst_timestamp}/ 下に出力しました。")

    return



