# -*- coding: utf-8 -*-
"""
BW_2_preprocess_single.py — BW 图预处理 (复用原 DJ/TX/UL 标准化参数)

★ 核心原则: 必须用原 ./standardize_params.pkl 里的 stress_eps / edge_mean/std /
  hs_mean/std 做变换, 不能用 BW 自己算的 — 否则输入分布与训练时不一致.

依赖:
  ./standardize_params.pkl   由原 2_preprocess_single.py 跑完后产生
  ./single_level_graphs_BW   由 BW_1_build_single.py 产生

输出:
  ./bw_data_log10.pkl
"""
import os, pickle, glob
import numpy as np
import torch

from config_single import (
    EDGE_DIM, FEAT_DIM, DERIVED_FEATURE_NAMES,
    HOTSPOT_DIM, HOTSPOT_KEYS,
    STRESS_PREPROCESS, USE_LOG10,
)

GRAPH_DIR_BW    = './single_level_graphs_BW'
BW_AUG_PKL_PATH = './bw_data_log10.pkl' if USE_LOG10 else './bw_data.pkl'
PARAM_PATH      = './standardize_params.pkl'

if not os.path.exists(PARAM_PATH):
    raise FileNotFoundError(f'未找到原训练标准化参数 {PARAM_PATH}\n'
                            f'→ 请先运行 `2_preprocess_single.py` 生成该文件')

with open(PARAM_PATH, 'rb') as f:
    meta = pickle.load(f)

stress_mean = meta.get('stress_mean')
stress_std  = meta.get('stress_std')
stress_eps  = meta.get('stress_eps')
ea_mean     = meta['edge_mean']
ea_std      = meta['edge_std']
hs_mean     = meta['hs_mean']
hs_std      = meta['hs_std']

print(f'加载原训练标准化参数: {PARAM_PATH}')
print(f'  stress_preprocess = {meta.get("stress_preprocess")}')
if STRESS_PREPROCESS == 'log10':
    print(f'  stress_eps        = {stress_eps}')
print(f'  edge_std          = {ea_std}')
print(f'  hs_mean           = {hs_mean}')
print(f'  hs_std            = {hs_std}\n')

files = sorted(glob.glob(os.path.join(GRAPH_DIR_BW, '*.pkl')))
if len(files) == 0:
    raise RuntimeError(f'{GRAPH_DIR_BW} 为空, 请先运行 BW_1_build_single.py')
specimens = [pickle.load(open(fp, 'rb')) for fp in files]
print(f'加载 {len(specimens)} 个 BW 图 (来自 {GRAPH_DIR_BW})')


def to_tensors(sp):
    feat_raw = sp['features']
    if not np.all(np.isfinite(feat_raw)):
        n_bad = int(np.sum(~np.isfinite(feat_raw)))
        print(f"  ⚠️ [BW] {sp['ID']}: 节点应力有 {n_bad}/{feat_raw.size} 个 nan/inf, 用 1 MPa 替换")
        feat_raw = np.where(np.isfinite(feat_raw), feat_raw, 1.0)

    if STRESS_PREPROCESS == 'log10':
        stress_safe = np.where(feat_raw > 0, feat_raw, stress_eps)
        stress = np.log10(np.clip(stress_safe, stress_eps, None))
    elif STRESS_PREPROCESS == 'standardize':
        stress = (feat_raw - stress_mean) / stress_std
    else:
        stress = feat_raw

    if not np.all(np.isfinite(stress)):
        stress = np.nan_to_num(stress, nan=0.0, posinf=0.0, neginf=0.0)

    x = torch.tensor(stress, dtype=torch.float)

    edges = sp['edge_index']
    if len(edges) > 0:
        ea_raw = sp['edge_attr']
        if not np.all(np.isfinite(ea_raw)):
            ea_raw = np.nan_to_num(ea_raw, nan=0.0, posinf=0.0, neginf=0.0)

        src = list(edges[:, 0]) + list(edges[:, 1])
        dst = list(edges[:, 1]) + list(edges[:, 0])
        ei = torch.tensor([src, dst], dtype=torch.long)

        ea_fwd = (ea_raw - ea_mean) / ea_std
        ea_rev = (-ea_raw - ea_mean) / ea_std
        edge_attr = torch.tensor(np.vstack([ea_fwd, ea_rev]), dtype=torch.float)
    else:
        ei = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, EDGE_DIM), dtype=torch.float)

    hs_raw = np.array([sp['hotspot_descriptors'][k] for k in HOTSPOT_KEYS])
    hotspot_desc = torch.tensor((hs_raw - hs_mean) / hs_std, dtype=torch.float)

    return {
        'x':             x,
        'edge_index':    ei,
        'edge_attr':     edge_attr,
        'n_nodes':       sp['n_nodes'],
        'y':             torch.tensor([sp['log_life']], dtype=torch.float),
        'life':          sp['life'],
        'ID':            sp['ID'],             # 'BW20-1'
        'kl_side':       sp['kl_side'],
        'joint_type':    sp['joint_type'],     # 'BW'
        'corrosion_hours': sp['corrosion_hours'],
        'censored':      sp.get('censored', False),
        'hotspot_desc':  hotspot_desc,
    }


all_data = [to_tensors(sp) for sp in specimens]
print(f'总计: {len(all_data)} 个 BW 图\n')

# 检查变换后分布
print(f'样例输入 (全部 BW 聚合):')
print(f'  节点特征 x:')
for i, name in enumerate(DERIVED_FEATURE_NAMES):
    col_all = np.concatenate([d['x'][:, i].numpy() for d in all_data])
    print(f'    {name}: [{col_all.min():.2f}, {col_all.max():.2f}], mean={col_all.mean():.2f}')

hs_cat = np.stack([d['hotspot_desc'].numpy() for d in all_data])
print(f'  hotspot_desc (标准化后):')
for i, k in enumerate(HOTSPOT_KEYS):
    print(f'    {k}: [{hs_cat[:, i].min():.2f}, {hs_cat[:, i].max():.2f}], '
          f'mean={hs_cat[:, i].mean():.2f}')

with open(BW_AUG_PKL_PATH, 'wb') as f:
    pickle.dump(all_data, f)
print(f'\n保存: {BW_AUG_PKL_PATH}')
print(f'下一步: python BW_3_train_v11.py')
