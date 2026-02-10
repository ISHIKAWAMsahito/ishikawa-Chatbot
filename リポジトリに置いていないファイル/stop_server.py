import os
import platform
import subprocess

def kill_process_by_port(port):
    current_os = platform.system()
    print(f"Detected OS: {current_os}")

    try:
        if current_os == "Windows":
            # Windows版: netstat で PID を探す
            cmd = f'netstat -ano | findstr :{port} | findstr LISTENING'
            output = subprocess.check_output(cmd, shell=True).decode()
            if output:
                # 行の最後にある数字が PID
                pid = output.strip().split()[-1]
                print(f"Found Windows process {pid}. Killing it...")
                os.system(f"taskkill /F /PID {pid}")
        
        else:
            # Linux (WSL) 版
            cmd = f"lsof -ti:{port}"
            pid = subprocess.check_output(cmd, shell=True).decode().strip()
            if pid:
                print(f"Found Linux process {pid}. Killing it...")
                os.system(f"kill -9 {pid}")

    except Exception as e:
        print(f"Port {port} is already free or Error occurred: {e}")

if __name__ == "__main__":
    kill_process_by_port(8080)