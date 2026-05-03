# -*- coding: utf-8 -*-
"""
脚本 0: 从原始 CSV 提取 TOP N 节点 (按 σ₁ 排序).
σ₁ 由 6 分量应力张量的最大特征值得到.
"""
import os, re
import pandas as pd
import numpy as np
from config_single import RAW_DATA_DIR, DATA_DIR, TOP_N, RAW_STRESS_COLS

dj_table_path = './DJ.xlsx'
drop_cols = ['eps_p_a', 'eqplas', 'DeltaWp']

dj_ranges = {}
if os.path.exists(dj_table_path):
    df_dj = pd.read_excel(dj_table_path)
    for _, row in df_dj.iterrows():
        dj_ranges[row['ID']] = {
            0: (row['toe0_s'], row['toe0_e']),
            1: (row['toe1_s'], row['toe1_e']),
        }
    print(f'加载 DJ.xlsx: {len(dj_ranges)} 个试件')

os.makedirs(DATA_DIR, exist_ok=True)
all_files = sorted([f for f in os.listdir(RAW_DATA_DIR) if f.lower().endswith('.csv')])
print(f'找到 {len(all_files)} 个 CSV, 取 top{TOP_N} (按 σ₁)\n')

total = 0
for fn in all_files:
    fp = os.path.join(RAW_DATA_DIR, fn)
    bn = os.path.splitext(fn)[0]
    try:
        df = pd.read_csv(fp)
    except Exception:
        continue

    df.columns = [str(c).strip() for c in df.columns]
    df = df.drop(columns=[c for c in df.columns if c in ('X.1', 'Y.1', 'Z.1')], errors='ignore')
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    if any(c not in df.columns for c in RAW_STRESS_COLS):
        continue

    # DJ 试件按 toe 段裁剪
    m = re.match(r'^(DJ\d+-\d+)_toe(\d+)_raw', bn)
    if m:
        sid, tid = m.group(1), int(m.group(2))
        if sid in dj_ranges and tid in dj_ranges[sid]:
            ys, ye = dj_ranges[sid][tid]
            df = df[(df['Y'] >= ys) & (df['Y'] <= ye)].copy()

    if df.empty:
        continue

    # 计算 σ₁ (最大主应力)
    raw = np.nan_to_num(df[RAW_STRESS_COLS].values)
    S11, S22, S33 = raw[:, 0], raw[:, 1], raw[:, 2]
    S12, S13, S23 = raw[:, 3], raw[:, 4], raw[:, 5]
    tensors = np.zeros((len(raw), 3, 3))
    tensors[:, 0, 0] = S11; tensors[:, 1, 1] = S22; tensors[:, 2, 2] = S33
    tensors[:, 0, 1] = tensors[:, 1, 0] = S12
    tensors[:, 0, 2] = tensors[:, 2, 0] = S13
    tensors[:, 1, 2] = tensors[:, 2, 1] = S23
    df['sigma1'] = np.linalg.eigvalsh(tensors)[:, 2]

    # 取 TOP N
    n_sel = min(TOP_N, len(df))
    top = df.nlargest(n_sel, 'sigma1').reset_index(drop=True)
    top.insert(0, 'Rank', np.arange(1, len(top) + 1))
    result = top.sort_values(by='X').reset_index(drop=True)

    out = os.path.join(DATA_DIR, f'{bn}_top{n_sel}.csv')
    result.to_csv(out, index=False, encoding='utf-8-sig')
    total += 1
    if total <= 3 or total % 20 == 0:
        s1_max = result['sigma1'].max()
        s22_max = result['S22_peak'].max()
        print(f'  {fn} -> {n_sel}点 (σ₁_max={s1_max:.1f}, S22_max={s22_max:.1f})')

print(f'\n完成! {total} 个文件 -> {DATA_DIR}')
