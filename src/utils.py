from logging import DEBUG, getLogger
import subprocess
from typing import Any

import numpy as np

logger = getLogger(__name__)
logger.setLevel(DEBUG)

def get_win_host():
    # WSLから見たデフォルトゲートウェイ（Windows側ホスト）のIPを取得
    route_output = subprocess.getoutput("ip route show | grep default | awk '{print $3}'").strip()
    return route_output

def _check_l2_normalized(vec: list[float], atol: float = 1e-5) -> dict:
    """
    埋め込みベクトルへの変換の仕様を検証・理解するための関数（処理上は不要）
    """
    # vec を Numpy 配列に変換
    v = np.asarray(vec, dtype=np.float64).ravel()

    l2_norm = float(np.linalg.norm(v, axis=-1))

    is_normalized = bool(np.isclose(l2_norm, 1.0, atol=atol, rtol=0.0))

    result = {
        "dim": int(v.size),
        "l2_norm": l2_norm,
        "is_normalized": is_normalized,
    }
    logger.debug(f"埋め込みベクトルの中身: {vec[:5]=}")
    logger.debug(f"{result=}")


