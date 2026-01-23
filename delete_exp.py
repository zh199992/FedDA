# A/delete_exp.py
import sys
import os
import json
import shutil
import subprocess
from pathlib import Path

# === 配置区 ===
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # 项目主目录 A
NNI_WORKING_DIR = ROOT_DIR  # nnictl 命令执行的目录（必须是运行 create 的目录）
NNICTL = os.path.join(os.path.dirname(sys.executable), "nnictl")  # 关键！
def run_cmd(cmd, cwd=None, input_str=None, timeout=15):
    """
    运行命令，可选自动输入（如 "y\n"）
    """
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd or ROOT_DIR,
            input=input_str,
            text=True,
            capture_output=True,
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)

def delete_experiment_by_id(exp_id):
    print(f"开始清理实验: {exp_id}")
    cwd = NNI_WORKING_DIR

    # === 第一步：nnictl stop <id> ===
    print("1 正在停止实验...")
    run_cmd(f"{sys.executable} -m nni stop {exp_id}", timeout=10)

    # === 第二步：nnictl experiment delete <id> ===
    print("2正在删除 NNI 实验元数据...")
    ret, out, err = run_cmd(
        cmd=f"{sys.executable} -m nni experiment delete {exp_id}",
        input_str="y\n",   # 👈 自动输入 y
        timeout=15
    )
    if ret == 0:
        print("NNI 实验已删除")
    else:
        print(f"删除失败: {err}")

    # === 第三步：删除本地/远程的 graph 和 config 目录 ===
    index_file = Path(ROOT_DIR) / "logs" / ".experiment_index" / f"{exp_id}.json"
    if not index_file.exists():
        print("[Info] 无本地索引文件，跳过自定义文件删除。")
        return

    try:
        with open(index_file) as f:
            meta = json.load(f)
        graph_path = meta["graph_path"]
        config_path = meta["config_path"]
    except Exception as e:
        print(f"[Error] 读取索引失败: {e}")
        return

    paths = [p for p in [graph_path, config_path] if os.path.exists(p)]
    if not paths:
        print("[Info] 对应的结果文件目录已不存在。")
    else:
        print("3将删除以下自定义结果目录:")
        for p in paths:
            print(f"  - {p}")
        if input("确认删除这些文件？(y/N): ").lower() == 'y':
            for p in paths:
                shutil.rmtree(p)
                print(f" 已删除: {p}")
        else:
            print(" 文件删除已取消。")

    # 清理索引文件
    index_file.unlink(missing_ok=True)
    print(" 全部清理完成！")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python delete_exp.py <experiment_id>")
        sys.exit(1)
    exp_id = sys.argv[1]
    delete_experiment_by_id(exp_id)