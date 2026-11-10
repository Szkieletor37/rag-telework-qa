import subprocess

def get_win_host():
    # 自宅用（自分のPC）
    #win_host = subprocess.getoutput("hostname.exe").strip() + ".local"

    # 事業所用（貸与PC)
    # WSLから見たデフォルトゲートウェイ（Windows側ホスト）のIPを取得
    route_output = subprocess.getoutput("ip route show | grep default | awk '{print $3}'").strip()
    return route_output