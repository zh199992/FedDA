import os
import json
import shutil
import subprocess
import sys


def delete_experiment(exp_id):
    """
    删除指定实验ID的所有相关文件和文件夹

    步骤：
    1. 停止实验：nnictl stop exp_id
    2. 删除实验：nnictl experiment delete exp_id + 确认输入y
    3. 删除索引文件中记录的所有config_path和graph_path文件夹
    4. 更新索引文件

    Args:
        exp_id: 要删除的实验ID
    """

    # 1. 停止实验
    print(f"正在停止实验 {exp_id}...")
    stop_command = f"nnictl stop {exp_id}"
    stop_result = subprocess.run(stop_command, shell=True, capture_output=True, text=True)

    if stop_result.returncode == 0:
        print(f"✓ 实验 {exp_id} 已停止")
    else:
        print(f"⚠ 停止实验时出现警告: {stop_result.stderr}")

    # 2. 删除实验（自动输入y确认）
    print(f"正在删除实验 {exp_id}...")
    delete_command = f"echo y | nnictl experiment delete {exp_id}"
    delete_result = subprocess.run(delete_command, shell=True, capture_output=True, text=True)

    if delete_result.returncode == 0:
        print(f"✓ 实验 {exp_id} 已从NNI中删除")
    else:
        print(f"⚠ 删除实验时出现警告: {delete_result.stderr}")

    # 3. 删除索引文件中记录的文件夹
    root_dir = os.path.dirname(os.path.abspath(__file__))  # 假设根目录是当前脚本的父目录的父目录
    index_dir = os.path.join(root_dir, "logs", ".experiment_index")
    index_file = os.path.join(index_dir, f"{exp_id}.json")
    print('__file__:'+__file__)
    print('index_file:'+index_file)
    deleted_folders = []

    if os.path.exists(index_file):
        try:
            # 读取索引文件
            with open(index_file, 'r') as f:
                index_data = json.load(f)

            # 删除config_paths中记录的所有文件夹
            if "config_paths" in index_data:
                for config_path in index_data["config_paths"]:
                    if os.path.exists(config_path):
                        shutil.rmtree(config_path)
                        deleted_folders.append(config_path)
                        print(f"✓ 已删除配置文件夹: {config_path}")
                    else:
                        print(f"⚠ 配置文件夹不存在: {config_path}")

            # 删除graph_paths中记录的所有文件夹
            if "graph_paths" in index_data:
                for graph_path in index_data["graph_paths"]:
                    if os.path.exists(graph_path):
                        shutil.rmtree(graph_path)
                        deleted_folders.append(graph_path)
                        print(f"✓ 已删除图形文件夹: {graph_path}")
                    else:
                        print(f"⚠ 图形文件夹不存在: {graph_path}")

            # 删除索引文件本身
            os.remove(index_file)
            print(f"✓ 已删除索引文件: {index_file}")

        except Exception as e:
            print(f"❌ 处理索引文件时出错: {e}")
            return False
    else:
        print(f"⚠ 索引文件不存在: {index_file}")

    # 4. 额外清理：检查是否有其他相关的文件夹
    logs_dir = os.path.join(root_dir, "logs")
    if os.path.exists(logs_dir):
        # 查找可能存在的其他相关文件夹
        for root, dirs, files in os.walk(logs_dir):
            for dir_name in dirs:
                if exp_id in dir_name:
                    dir_path = os.path.join(root, dir_name)
                    try:
                        shutil.rmtree(dir_path)
                        deleted_folders.append(dir_path)
                        print(f"✓ 已删除相关文件夹: {dir_path}")
                    except Exception as e:
                        print(f"⚠ 删除文件夹失败 {dir_path}: {e}")

    print(f"\n🎉 清理完成!")
    print(f"已删除 {len(deleted_folders)} 个文件夹")
    for folder in deleted_folders:
        print(f"  - {folder}")

    return True


def main():
    if len(sys.argv) != 2:
        print("使用方法: python delete_experiment.py <experiment_id>")
        print("示例: python delete_experiment.py exp123")
        sys.exit(1)

    exp_id = sys.argv[1]

    print(f"即将删除实验 {exp_id} 的所有相关文件")
    print("此操作将执行以下步骤:")
    print("1. nnictl stop", exp_id)
    print("2. nnictl experiment delete", exp_id)
    print("3. 删除所有相关的配置和图形文件夹")
    print("4. 删除索引文件")

    confirm = input("确认删除? (输入 'yes' 继续): ")
    if confirm.lower() != 'yes':
        print("操作已取消")
        sys.exit(0)

    try:
        success = delete_experiment(exp_id)
        if success:
            print(f"\n✅ 实验 {exp_id} 删除完成")
        else:
            print(f"\n❌ 实验 {exp_id} 删除过程中出现错误")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ 删除过程中出现异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()