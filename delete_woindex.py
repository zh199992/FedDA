import os
import shutil
import subprocess
import sys


def find_and_delete_experiment_folders(root_dir, exp_id):
    """
    遍历logs文件夹，查找并删除所有包含exp_id的文件夹

    Args:
        root_dir: 根目录路径
        exp_id: 实验ID

    Returns:
        list: 被删除的文件夹路径列表
    """
    logs_dir = os.path.join(root_dir, "logs")
    deleted_folders = []

    if not os.path.exists(logs_dir):
        print(f"⚠ logs目录不存在: {logs_dir}")
        return deleted_folders

    print(f"正在扫描logs目录: {logs_dir}")

    # 遍历logs目录下的所有子目录
    for root, dirs, files in os.walk(logs_dir):
        for dir_name in dirs:
            if exp_id in dir_name:
                dir_path = os.path.join(root, dir_name)
                try:
                    # 检查是否在config或graph路径中，或者是实验ID命名的文件夹
                    if any(x in dir_path.lower() for x in ['config', 'graph']) or dir_name == exp_id:
                        shutil.rmtree(dir_path)
                        deleted_folders.append(dir_path)
                        print(f"✓ 已删除文件夹: {dir_path}")
                except Exception as e:
                    print(f"⚠ 删除文件夹失败 {dir_path}: {e}")

    return deleted_folders


def delete_experiment_without_index(exp_id):
    """
    删除指定实验ID的所有相关文件（不依赖索引文件）

    步骤：
    1. 停止实验：nnictl stop exp_id
    2. 删除实验：nnictl experiment delete exp_id + 确认输入y
    3. 遍历logs文件夹删除所有包含exp_id的config和graph文件夹

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

    # 3. 遍历删除相关文件夹
    # 根目录是当前脚本的父目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)

    print(f"开始查找并删除与实验 {exp_id} 相关的文件夹...")
    deleted_folders = find_and_delete_experiment_folders(root_dir, exp_id)

    # 4. 额外删除索引文件（如果存在）
    index_dir = os.path.join(root_dir, "logs", ".experiment_index")
    index_file = os.path.join(index_dir, f"{exp_id}.json")
    if os.path.exists(index_file):
        try:
            os.remove(index_file)
            print(f"✓ 已删除索引文件: {index_file}")
        except Exception as e:
            print(f"⚠ 删除索引文件失败: {e}")

    print(f"\n🎉 清理完成!")
    print(f"已删除 {len(deleted_folders)} 个文件夹")
    if deleted_folders:
        print("删除的文件夹列表:")
        for folder in deleted_folders:
            print(f"  - {folder}")
    else:
        print("未找到与实验ID相关的文件夹")

    return True


def search_experiment_folders(exp_id):
    """
    仅搜索不删除，用于预览将要删除的文件夹
    """
    script_dir = os.path.abspath(__file__)
    root_dir = os.path.dirname(script_dir)
    logs_dir = os.path.join(root_dir, "logs")

    found_folders = []

    if not os.path.exists(logs_dir):
        print(f"⚠ logs目录不存在: {logs_dir}")
        return found_folders

    print(f"扫描logs目录: {logs_dir}")
    print(f"查找包含 '{exp_id}' 的文件夹...")

    for root, dirs, files in os.walk(logs_dir):
        for dir_name in dirs:
            if exp_id in dir_name:
                dir_path = os.path.join(root, dir_name)
                # 检查是否在config或graph路径中，或者是实验ID命名的文件夹
                if any(x in dir_path.lower() for x in ['config', 'graph']) or dir_name == exp_id:
                    found_folders.append(dir_path)

    return found_folders


def main():
    if len(sys.argv) < 2:
        print("使用方法: python delete_experiment.py <experiment_id> [--dry-run]")
        print("示例: python delete_experiment.py exp123")
        print("示例（预览模式）: python delete_experiment.py exp123 --dry-run")
        sys.exit(1)

    exp_id = sys.argv[1]
    dry_run = len(sys.argv) > 2 and sys.argv[2] == '--dry-run'

    if dry_run:
        # 预览模式：只显示将要删除的文件夹，不实际删除
        print(f"🔍 预览模式：查找与实验 {exp_id} 相关的文件夹")
        found_folders = search_experiment_folders(exp_id)

        if found_folders:
            print(f"\n找到 {len(found_folders)} 个相关文件夹:")
            for folder in found_folders:
                print(f"  - {folder}")
            print(f"\n如果要删除这些文件夹，请运行: python delete_experiment.py {exp_id}")
        else:
            print(f"\n未找到与实验 {exp_id} 相关的文件夹")
    else:
        # 实际删除模式
        print(f"即将删除实验 {exp_id} 的所有相关文件")
        print("此操作将执行以下步骤:")
        print("1. nnictl stop", exp_id)
        print("2. nnictl experiment delete", exp_id)
        print("3. 遍历logs文件夹删除所有包含实验ID的config和graph文件夹")

        confirm = input("确认删除? (输入 'yes' 继续): ")
        if confirm.lower() != 'yes':
            print("操作已取消")
            sys.exit(0)

        try:
            success = delete_experiment_without_index(exp_id)
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