# -*- coding: utf-8 -*-
"""
BW_0_extract_single.py — BW 试件 top-N 节点提取 (按 σ₁ 排序)

输入: ./toe_raw_csv_BW/*.csv
输出: ./toe_sdv_tables_BW/sigma1_top{TOP_N}/*_topN.csv

注: 本脚本对两侧 CSV 均处理; KL 侧过滤在 BW_1 中完成.
"""
import os
import pandas as pd
import numpy as np

from config_single import TOP_N, RAW_STRESS_COLS

RAW_DATA_DIR_BW = './toe_raw_csv_BW'
DATA_DIR_BW     = f'./toe_sdv_tables_BW/sigma1_top{TOP_N}'

drop_cols = ['eps_p_a', 'eqplas', 'DeltaWp']

os.makedirs(DATA_DIR_BW, exist_ok=True)
all_files = sorted([f for f in os.listdir(RAW_DATA_DIR_BW) if f.lower().endswith('.csv')])
print(f'BW raw csv 源: {RAW_DATA_DIR_BW}')
print(f'找到 {len(all_files)} 个 CSV, 取 top{TOP_N} (按 σ₁)\n')

total = 0
for fn in all_files:
    fp = os.path.join(RAW_DATA_DIR_BW, fn)
    bn = os.path.splitext(fn)[0]
    try:
        df = pd.read_csv(fp)
    except Exception as e:
        print(f'  ⚠️ 读取失败 {fn}: {e}')
        continue

    df.columns = [str(c).strip() for c in df.columns]
    df = df.drop(columns=[c for c in df.columns if c in ('X.1', 'Y.1', 'Z.1')], errors='ignore')
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    if any(c not in df.columns for c in RAW_STRESS_COLS):
        missing = [c for c in RAW_STRESS_COLS if c not in df.columns]
        print(f'  ⚠️ 跳过 {fn}: 缺列 {missing}')
        continue

    if df.empty:
        continue

    raw = np.nan_to_num(df[RAW_STRESS_COLS].values)
    S11, S22, S33 = raw[:, 0], raw[:, 1], raw[:, 2]
    S12, S13, S23 = raw[:, 3], raw[:, 4], raw[:, 5]
    tensors = np.zeros((len(raw), 3, 3))
    tensors[:, 0, 0] = S11; tensors[:, 1, 1] = S22; tensors[:, 2, 2] = S33
    tensors[:, 0, 1] = tensors[:, 1, 0] = S12
    tensors[:, 0, 2] = tensors[:, 2, 0] = S13
    tensors[:, 1, 2] = tensors[:, 2, 1] = S23
    df['sigma1'] = np.linalg.eigvalsh(tensors)[:, 2]

    n_sel = min(TOP_N, len(df))
    top = df.nlargest(n_sel, 'sigma1').reset_index(drop=True)
    top.insert(0, 'Rank', np.arange(1, len(top) + 1))
    result = top.sort_values(by='X').reset_index(drop=True)

    out = os.path.join(DATA_DIR_BW, f'{bn}_top{n_sel}.csv')
    result.to_csv(out, index=False, encoding='utf-8-sig')
    total += 1
    s1_max = result['sigma1'].max()
    s22_max = result['S22_peak'].max() if 'S22_peak' in result.columns else np.nan
    if total <= 5 or total % 5 == 0:
        print(f'  {fn} -> {n_sel} 点 (σ₁_max={s1_max:.1f}, S22_max={s22_max:.1f})')

print(f'\n完成! {total} 个文件 -> {DATA_DIR_BW}')
