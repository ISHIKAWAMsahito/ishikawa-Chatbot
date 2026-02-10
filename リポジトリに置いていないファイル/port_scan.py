import socket

def check_port(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    # connect_ex は 0 なら成功（オープン）、それ以外はエラー（クローズ）
    result = sock.connect_ex((ip, port))
    if result == 0:
        print(f"[ OPEN ] Port {port} is welcoming!")
    else:
        print(f"[CLOSED] Port {port} is shut.")
    sock.close()

if __name__ == "__main__":
    print("--- Testing My Local Ports ---")
    # ss で見つけた 53番（DNS）をチェック
    check_port("127.0.0.53", 53)
    # 恐らく閉まっている 80番（Web）をチェック
    check_port("127.0.0.1", 80)