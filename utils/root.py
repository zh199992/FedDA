import os

def find_project_root(target_dirname='FedDA'):
    curr_path = os.path.abspath(__file__)
    while True:
        curr_path = os.path.dirname(curr_path)
        if os.path.basename(curr_path) == target_dirname:
            return curr_path
        if curr_path == '/':  # 根目录了还没找到
            raise FileNotFoundError(f"Target directory '{target_dirname}' not found.")



