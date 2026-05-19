# from tbparse import SummaryReader
# import pandas as pd
#
# log_dir = '/data/4T/zh_file/project/FedDA/logs/graph/FedAvg/kchf54yu/FedAvg/vMsiX-2026-05-06T16-27-41/' # 比如 logs/
# reader = SummaryReader(log_dir)
#
# # 1. 提取所有标量（Loss, RMSE, Accuracy等）
# df_scalars = reader.scalars
# print(df_scalars.head())
#
# # 2. 如果你记录了超参数 (HParams)
# df_hparams = reader.hparams
# print(df_hparams)

import sqlite3
import pandas as pd
import json
import os


def extract_nni_experiment_data(exp_id, nni_base='/home/zh/nni-experiments/'):
    """
    自动定位工程根目录，并将分析结果保存到 logs/nni_analysis
    """
    # 1. 自动获取工程根目录 (project_root)
    # os.path.abspath(__file__) 获取当前运行脚本的绝对路径
    # os.path.dirname(...) 获取其父目录
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(current_script_path)

    # 如果你的脚本是在子文件夹里（比如在 /utils/ 下），则需要向上跳一级：
    # project_root = os.path.dirname(os.path.dirname(current_script_path))

    # 2. 构建路径
    db_path = os.path.join(nni_base, exp_id, 'db/nni.sqlite')
    output_dir = os.path.join(project_root, 'logs', 'nni_analysis')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(db_path):
        print(f"❌ 未找到数据库: {db_path}")
        return None

    # 3. 读取并解析数据库 (逻辑保持不变)
    conn = sqlite3.connect(db_path)
    df_raw = pd.read_sql_query("SELECT trialJobId, event, data FROM TrialJobEvent", conn)
    conn.close()

    results = {}
    for _, row in df_raw.iterrows():
        tid = row['trialJobId']
        if row['data'] is None: continue
        if tid not in results: results[tid] = {'trial_id': tid}
        try:
            data = json.loads(row['data'])
            if isinstance(data, dict):
                if 'parameters' in data:
                    results[tid].update(data['parameters'])
                if row['event'] in ['FINAL_METRIC', 'SUCCEEDED']:
                    metric = data.get('data', data.get('value', data))
                    if isinstance(metric, dict) and 'default' in metric:
                        metric = metric['default']
                    results[tid]['final_metric'] = metric
        except:
            continue

    # 4. 导出
    if results:
        final_df = pd.DataFrame(list(results.values())).dropna(thresh=2)
        file_base = os.path.join(output_dir, f'analysis_{exp_id}')
        final_df.to_csv(f'{file_base}.csv', index=False)
        print(f"✅ 自动定位工程目录: {project_root}")
        print(f"✅ 结果已保存至: {output_dir}/analysis_{exp_id}.csv")
        return final_df
    return None


# 现在只需输入实验 ID 即可
# extract_nni_auto_path('kchf54yu')

# extract_nni_experiment_data('kchf54yu')

import os
import pandas as pd
from tbparse import SummaryReader
from pathlib import Path


def extract_tb_curves_for_ai(exp_id):
    """
    根据 exp_id 批量提取 TensorBoard 数据，降采样后保存至 logs/nni_analysis/<exp_id>/
    """
    # 1. 自动定位工程根目录 (假设此脚本在工程内的某处)
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(current_script_path)

    # 如果脚本在子目录(如 utils/)，请取消下面这行的注释向上跳一级
    # project_root = os.path.dirname(project_root)

    # 2. 构建目标输出目录
    output_dir = os.path.join(project_root, 'logs', 'nni_analysis', exp_id)
    os.makedirs(output_dir, exist_ok=True)

    # 3. 动态探测 TensorBoard 真实路径
    # 针对路径: logs/graph/{aim}/{exp_id}/{algorithm}/{trial_id}-{TIMESTAMP}
    graph_base = Path(project_root) / 'logs' / 'graph'

    # 使用 rglob 忽略层级差异，直接定位包含 exp_id 的目录
    exp_dirs = list(graph_base.rglob(exp_id))
    if not exp_dirs:
        print(f"❌ 错误：在 {graph_base} 及其子目录下未找到实验 {exp_id}。")
        return None

    exp_dir = exp_dirs[0]
    print(f"🔍 定位到实验日志库: {exp_dir}")

    # 4. 遍历所有 trial 文件夹并解析
    # 匹配 {algorithm}/{trial_id}-{TIMESTAMP} 层级
    trial_dirs = list(exp_dir.glob('*/*'))

    all_curves = []

    for t_dir in trial_dirs:
        if not t_dir.is_dir():
            continue

        # 提取 trial_id：根据你的逻辑 nni.get_trial_id() + '-' + args.TIMESTAMP
        # NNI 的 trial_id 内部不含 '-', 所以用 split 截取第一部分极其精准
        folder_name = t_dir.name
        trial_id = folder_name.split('-')[0]

        print(f"⏳ 正在深度解析 Trial: {trial_id} ...")
        try:
            # 禁用 pivot 提升读取大文件的速度
            reader = SummaryReader(str(t_dir), pivot=False)
            df = reader.scalars
            if df.empty:
                continue

            df['trial_id'] = trial_id

            # --- 核心：AI 视角的动态降采样 ---
            # 保证 AI 能够掌握全局趋势而不被冗余波动“撑爆”上下文
            sampled_dfs = []
            for tag, group in df.groupby('tag'):
                # 动态计算采样间隔：无论跑了1万步还是25万步，每条曲线只抽 ~100 个点
                interval = max(1, len(group) // 100)

                df_sampled = group.iloc[::interval].copy()
                df_last = group.tail(1).copy()  # 强制保留绝对的最后一步，用于定标

                sampled_dfs.extend([df_sampled, df_last])

            # 合并当前 trial 并去除可能因采样导致的重复最后一步
            df_trial_final = pd.concat(sampled_dfs).drop_duplicates(subset=['step', 'tag'])
            all_curves.append(df_trial_final)

        except Exception as e:
            print(f"⚠️ 解析 {trial_id} 发生异常: {e}")

    # 5. 汇总并写出分析级 CSV
    if all_curves:
        final_df = pd.concat(all_curves, ignore_index=True)

        output_file = os.path.join(output_dir, f'tb_curves_ai_ready_{exp_id}.csv')
        final_df.to_csv(output_file, index=False)
        print(f"\n✅ 提取与动态降采样完成！")
        print(f"📁 结果就绪: {output_file}")
        print(f"📊 数据被完美提炼，总行数优化至 {len(final_df)} 行。")
        return output_file
    else:
        print("⚠️ 未提取到有效的曲线数据，请检查日志文件夹内是否有 tfevents 文件。")
        return None

# 执行示例
# extract_tb_curves_for_ai('kchf54yu')

def extract_tb_curves_full(exp_id):
    """
    根据 exp_id 批量提取 TensorBoard 数据（保留 100% 全量原始数据，不进行任何压缩）
    并保存至 logs/nni_analysis/<exp_id>/
    """
    # 1. 自动定位工程根目录
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(current_script_path)

    # 如果脚本在子目录(如 utils/)，请取消下面这行的注释向上跳一级
    # project_root = os.path.dirname(project_root)

    # 2. 构建目标输出目录
    output_dir = os.path.join(project_root, 'logs', 'nni_analysis', exp_id)
    os.makedirs(output_dir, exist_ok=True)

    # 3. 动态探测 TensorBoard 真实路径
    graph_base = Path(project_root) / 'logs' / 'graph'

    exp_dirs = list(graph_base.rglob(exp_id))
    if not exp_dirs:
        print(f"❌ 错误：在 {graph_base} 及其子目录下未找到实验 {exp_id}。")
        return None

    exp_dir = exp_dirs[0]
    print(f"🔍 定位到实验日志库: {exp_dir}")

    # 4. 遍历所有 trial 文件夹并解析
    trial_dirs = list(exp_dir.glob('*/*'))

    all_curves = []

    for t_dir in trial_dirs:
        if not t_dir.is_dir():
            continue

        folder_name = t_dir.name
        trial_id = folder_name.split('-')[0]

        print(f"⏳ 正在深度解析 Trial: {trial_id} (全量抽取模式) ...")
        try:
            reader = SummaryReader(str(t_dir), pivot=False)
            df = reader.scalars
            if df.empty:
                continue

            df['trial_id'] = trial_id

            # --- 直接追加全量 DataFrame，去除了之前的所有降采样逻辑 ---
            all_curves.append(df)

        except Exception as e:
            print(f"⚠️ 解析 {trial_id} 发生异常: {e}")

    # 5. 汇总并写出全量 CSV
    if all_curves:
        final_df = pd.concat(all_curves, ignore_index=True)

        output_file = os.path.join(output_dir, f'tb_curves_full_{exp_id}.csv')
        final_df.to_csv(output_file, index=False)
        print(f"\n✅ 全量数据提取完成！")
        print(f"📁 结果就绪: {output_file}")
        print(f"📊 保留了所有的原始波动，当前文件总行数: {len(final_df)} 行。")
        return output_file
    else:
        print("⚠️ 未提取到有效的曲线数据，请检查日志文件夹内是否有 tfevents 文件。")
        return None

# 执行示例
extract_tb_curves_full('kchf54yu')