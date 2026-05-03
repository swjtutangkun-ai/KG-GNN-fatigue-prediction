# -*- coding: utf-8 -*-
"""
脚本 2: 节点应力变换 + 双向边构造 (反向边差值取负).

★ V11 修改: to_tensors 增加 'coords' 字段, baseline 对比需要坐标信息.
"""
import os, pickle, glob
import numpy as np
import torch
from config_single import (
    GRAPH_DIR, AUG_PKL_PATH, EDGE_DIM, FEAT_DIM,
    DERIVED_FEATURE_NAMES, HOTSPOT_DIM, HOTSPOT_KEYS,
    STRESS_PREPROCESS, USE_LOG10,
)

files = sorted(glob.glob(os.path.join(GRAPH_DIR, '*.pkl')))
specimens = [pickle.load(open(fp, 'rb')) for fp in files]
n_failed = sum(1 for sp in specimens if not sp.get('censored', False))
n_censored = sum(1 for sp in specimens if sp.get('censored', False))
print(f'加载 {len(specimens)} 个试件 (失效 {n_failed}, 删失 {n_censored})')
print(f'目标 {"log10(N)" if USE_LOG10 else "log(1+N)"}, 应力变换 {STRESS_PREPROCESS}')

# ==================== 节点特征 ==================== #
all_stress = np.vstack([sp['features'] for sp in specimens])
n_bad = int(np.sum(~np.isfinite(all_stress)))
if n_bad > 0:
    print(f'\n⚠️ 应力中发现 {n_bad}/{all_stress.size} 个 nan/inf')

print(f'\n应力 [{FEAT_DIM} 维] 原始范围 (排除 nan):')
for i, name in enumerate(DERIVED_FEATURE_NAMES):
    s = all_stress[:, i][np.isfinite(all_stress[:, i])]
    if len(s) > 0:
        print(f'  {name}: [{s.min():.1f}, {s.max():.1f}] MPa, mean={s.mean():.1f}')

if STRESS_PREPROCESS == 'log10':
    stress_eps = 1.0
    stress_mean, stress_std = None, None
    print(f'  → log10 变换, eps={stress_eps} MPa')
elif STRESS_PREPROCESS == 'standardize':
    stress_clean = np.where(np.isfinite(all_stress), all_stress, np.nan)
    stress_mean = np.nanmean(stress_clean, axis=0)
    stress_std = np.nanstd(stress_clean, axis=0) + 1e-8
    stress_eps = None
    print(f'  → 标准化')
else:
    stress_mean, stress_std, stress_eps = None, None, None

# ==================== 坐标标准化参数 ==================== #
all_coords = np.vstack([sp['coords'] for sp in specimens])
coords_mean = all_coords.mean(axis=0)
coords_std = all_coords.std(axis=0) + 1e-8
print(f'\n坐标 [3 维] [x, y, z]:')
for i, name in enumerate(['x', 'y', 'z']):
    print(f'  {name}: mean={coords_mean[i]:.4f}, std={coords_std[i]:.4f}')

# ==================== 边特征 (有向差值, 双向标准化) ==================== #
# 由对称性 mean = 0, 只算 std
all_diffs_both = []
for sp in specimens:
    if len(sp['edge_attr']) > 0:
        all_diffs_both.append(sp['edge_attr'])
        all_diffs_both.append(-sp['edge_attr'])

if all_diffs_both:
    all_diffs_both = np.vstack(all_diffs_both)
    ea_std = all_diffs_both.std(axis=0) + 1e-8
    ea_mean = np.zeros(EDGE_DIM)
    print(f'\n边特征 [{EDGE_DIM} 维] [dx, dy, dz] 双向集合 std: {ea_std}')
else:
    ea_std = np.ones(EDGE_DIM)
    ea_mean = np.zeros(EDGE_DIM)

# ==================== 热点描述符标准化 ==================== #
raw_hotspot = np.array([[sp['hotspot_descriptors'][k] for k in HOTSPOT_KEYS]
                        for sp in specimens])
hs_mean = raw_hotspot.mean(axis=0)
hs_std = raw_hotspot.std(axis=0) + 1e-8

print(f'\n热点描述符 [{HOTSPOT_DIM} 维]:')
for i, k in enumerate(HOTSPOT_KEYS):
    print(f'  {k}: mean={hs_mean[i]:.4f}, std={hs_std[i]:.4f}')


def to_tensors(sp):
    feat_raw = sp['features']

    if not np.all(np.isfinite(feat_raw)):
        n_bad = int(np.sum(~np.isfinite(feat_raw)))
        cen = '删失' if sp.get('censored', False) else '失效'
        print(f"  ⚠️ [{cen}] {sp['ID']}: 节点应力有 {n_bad}/{feat_raw.size} 个 nan/inf, 用 1 MPa 替换")
        feat_raw = np.where(np.isfinite(feat_raw), feat_raw, 1.0)

    if STRESS_PREPROCESS == 'log10':
        stress_safe = np.where(feat_raw > 0, feat_raw, stress_eps)
        stress = np.log10(np.clip(stress_safe, stress_eps, None))
    elif STRESS_PREPROCESS == 'standardize':
        stress = (feat_raw - stress_mean) / stress_std
    else:
        stress = feat_raw

    if not np.all(np.isfinite(stress)):
        print(f"  ⚠️ {sp['ID']}: 应力变换后仍有 nan/inf")
        stress = np.nan_to_num(stress, nan=0.0, posinf=0.0, neginf=0.0)

    x = torch.tensor(stress, dtype=torch.float)

    # ★ 坐标标准化 (baseline 对比需要)
    coords_raw = sp['coords']
    coords_normed = (coords_raw - coords_mean) / coords_std
    coords = torch.tensor(coords_normed, dtype=torch.float32)

    edges = sp['edge_index']
    if len(edges) > 0:
        ea_raw = sp['edge_attr']
        if not np.all(np.isfinite(ea_raw)):
            print(f"  ⚠️ {sp['ID']}: edge_attr 有 nan/inf, 替换为 0")
            ea_raw = np.nan_to_num(ea_raw, nan=0.0, posinf=0.0, neginf=0.0)

        src = list(edges[:, 0]) + list(edges[:, 1])
        dst = list(edges[:, 1]) + list(edges[:, 0])
        ei = torch.tensor([src, dst], dtype=torch.long)

        # 反向边的差值取负
        ea_fwd = (ea_raw - ea_mean) / ea_std
        ea_rev = (-ea_raw - ea_mean) / ea_std
        edge_attr = torch.tensor(np.vstack([ea_fwd, ea_rev]), dtype=torch.float)
    else:
        ei = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, EDGE_DIM), dtype=torch.float)

    hs_raw = np.array([sp['hotspot_descriptors'][k] for k in HOTSPOT_KEYS])
    hotspot_desc = torch.tensor((hs_raw - hs_mean) / hs_std, dtype=torch.float)

    return {
        'x': x, 'edge_index': ei, 'edge_attr': edge_attr,
        'coords': coords,  # ★ 新增: 标准化坐标 (N, 3), baseline 对比用
        'n_nodes': sp['n_nodes'],
        'y': torch.tensor([sp['log_life']], dtype=torch.float),
        'life': sp['life'], 'ID': sp['ID'],
        'censored': sp.get('censored', False),
        'hotspot_desc': hotspot_desc,
    }


all_data = [to_tensors(sp) for sp in specimens]
print(f'\n总计: {len(all_data)} 个试件')

# 检查最终输入特征
print(f'\n样例输入 (前 20 个 specimen 聚合):')
print(f'  节点特征 x:')
for i, name in enumerate(DERIVED_FEATURE_NAMES):
    col_all = np.concatenate([d['x'][:, i].numpy() for d in all_data[:20]])
    print(f'    {name}: [{col_all.min():.2f}, {col_all.max():.2f}], mean={col_all.mean():.2f}')
print(f'  坐标 coords:')
for i, name in enumerate(['x', 'y', 'z']):
    col_all = np.concatenate([d['coords'][:, i].numpy() for d in all_data[:20]])
    print(f'    {name}: [{col_all.min():.2f}, {col_all.max():.2f}], mean={col_all.mean():.2f}')
print(f'  边特征 edge_attr:')
for i, name in enumerate(['dx', 'dy', 'dz']):
    col_all = np.concatenate([d['edge_attr'][:, i].numpy() for d in all_data[:20]
                              if d['edge_attr'].numel() > 0])
    print(f'    {name}: [{col_all.min():.2f}, {col_all.max():.2f}], mean={col_all.mean():.2f}')

meta = {
    'stress_preprocess': STRESS_PREPROCESS,
    'stress_mean': stress_mean, 'stress_std': stress_std, 'stress_eps': stress_eps,
    'feat_names': DERIVED_FEATURE_NAMES,
    'edge_mean': ea_mean, 'edge_std': ea_std,
    'coords_mean': coords_mean, 'coords_std': coords_std,  # ★ 新增
    'hs_mean': hs_mean, 'hs_std': hs_std,
    'hotspot_keys': HOTSPOT_KEYS,
    'use_log10': USE_LOG10,
}
with open('./standardize_params.pkl', 'wb') as f:
    pickle.dump(meta, f)
print('标准化参数: ./standardize_params.pkl')

with open(AUG_PKL_PATH, 'wb') as f:
    pickle.dump(all_data, f)
print(f'保存: {AUG_PKL_PATH}')
