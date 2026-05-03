# -*- coding: utf-8 -*-
"""
BW_1_build_single.py — BW 试件构图 (★ 仅为实验观察的开裂侧 KL 构图)

★ KL_N_BW.xlsx 中 KL ∈ {0, 1} 标注了实验观察到的开裂焊趾侧. 每根试件仅在
  该侧的焊趾区取 top-N 节点构图, 整个 BW 数据集共 14 个图
  (与 DJ/TX/UL 训练集中每根试件 1 个图的处理保持一致).

输出:
  ./single_level_graphs_BW/{sid}.pkl   (ID 不带 _toe 后缀, 直接 'BW20-1')

节点 3 维 [σ₁, τ_max, σ_vm], 边 3 维 [dx, dy, dz] 有向差值; 与 DJ/TX/UL 训练一致.
"""
import os, glob, pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.cluster import DBSCAN

from config_single import (
    TOP_N, RAW_STRESS_COLS, COORD_COLS,
    COL_ID, COL_TOE, COL_LIFE, parse_id,
    K_NEIGHBORS, USE_LOG10,
)

LIFE_TABLE_PATH_BW = './KL_N_BW.xlsx'
DATA_DIR_BW        = f'./toe_sdv_tables_BW/sigma1_top{TOP_N}'
GRAPH_DIR_BW       = './single_level_graphs_BW'

os.makedirs(GRAPH_DIR_BW, exist_ok=True)

# 清理旧 pkl
for fp in glob.glob(os.path.join(GRAPH_DIR_BW, '*.pkl')):
    os.remove(fp)


def life_to_y(life):
    return float(np.log10(life)) if USE_LOG10 else float(np.log1p(life))


def compute_derived_features(raw_stress):
    N = len(raw_stress)
    S11, S22, S33 = raw_stress[:, 0], raw_stress[:, 1], raw_stress[:, 2]
    S12, S13, S23 = raw_stress[:, 3], raw_stress[:, 4], raw_stress[:, 5]
    tensors = np.zeros((N, 3, 3))
    tensors[:, 0, 0] = S11; tensors[:, 1, 1] = S22; tensors[:, 2, 2] = S33
    tensors[:, 0, 1] = tensors[:, 1, 0] = S12
    tensors[:, 0, 2] = tensors[:, 2, 0] = S13
    tensors[:, 1, 2] = tensors[:, 2, 1] = S23
    eigvals = np.linalg.eigvalsh(tensors)
    sigma1 = eigvals[:, 2]; sigma3 = eigvals[:, 0]
    tau_max = (sigma1 - sigma3) / 2.0
    sigma_vm = np.sqrt(0.5 * ((S11 - S22) ** 2 + (S22 - S33) ** 2 + (S33 - S11) ** 2
                              + 6 * (S12 ** 2 + S13 ** 2 + S23 ** 2)))
    return np.column_stack([sigma1, tau_max, sigma_vm])


def compute_edge_features(coords, edges):
    if len(edges) == 0:
        return np.zeros((0, 3))
    src, dst = edges[:, 0], edges[:, 1]
    return coords[dst] - coords[src]


def compute_hotspot_descriptors(features, coords):
    sigma1 = features[:, 0]
    N = len(sigma1)
    s_max = sigma1.max()
    s_mean = sigma1.mean() + 1e-8
    F_concentration = float(s_max / s_mean)
    threshold = 0.75 * s_max
    high_mask = sigma1 >= threshold
    n_high = int(high_mask.sum())
    F_volume = float(n_high / N)

    if n_high < 5:
        return {'F_concentration': F_concentration, 'F_volume': F_volume,
                'F_gradient': 0.0, 'F_radius': 0.0,
                'F_n_hotspots': 1.0, 'F_separation': 0.0}

    high_coords = coords[high_mask]
    k_nn = min(5, n_high - 1)
    nn = NearestNeighbors(n_neighbors=k_nn); nn.fit(high_coords)
    dists, _ = nn.kneighbors(high_coords)
    eps = float(np.median(dists[:, -1]) * 1.5)

    db = DBSCAN(eps=eps, min_samples=3).fit(high_coords)
    labels = db.labels_
    unique_labels = set(labels) - {-1}
    n_clusters = len(unique_labels)
    F_n_hotspots = float(max(n_clusters, 1))

    if n_clusters == 0:
        centroid = high_coords.mean(axis=0)
        F_radius = float(np.sqrt(np.mean(np.sum((high_coords - centroid) ** 2, axis=1))))
        F_gradient = float((s_max - s_mean) / (F_radius + 1e-8))
        return {'F_concentration': F_concentration, 'F_volume': F_volume,
                'F_gradient': F_gradient, 'F_radius': F_radius,
                'F_n_hotspots': 1.0, 'F_separation': 0.0}

    cluster_sizes = {l: int(np.sum(labels == l)) for l in unique_labels}
    main_label = max(cluster_sizes, key=cluster_sizes.get)
    main_coords = high_coords[labels == main_label]
    centroid = main_coords.mean(axis=0)
    F_radius = float(np.sqrt(np.mean(np.sum((main_coords - centroid) ** 2, axis=1))))
    F_gradient = float((s_max - s_mean) / (F_radius + 1e-8))

    if n_clusters >= 2:
        centroids = np.array([high_coords[labels == l].mean(axis=0) for l in unique_labels])
        n_c = len(centroids); total_dist = 0.0; n_pairs = 0
        for i in range(n_c):
            for j in range(i + 1, n_c):
                total_dist += np.sqrt(np.sum((centroids[i] - centroids[j]) ** 2))
                n_pairs += 1
        avg_dist = total_dist / max(n_pairs, 1)
        F_separation = float(avg_dist / (F_radius + 1e-8))
    else:
        F_separation = 0.0

    return {'F_concentration': F_concentration, 'F_volume': F_volume,
            'F_gradient': F_gradient, 'F_radius': F_radius,
            'F_n_hotspots': F_n_hotspots, 'F_separation': F_separation}


def load_and_build_bw(sid, kl, life, jt, ct):
    """只构 KL 侧: csv 路径指向 {sid}_toe{kl}_raw_top{TOP_N}.csv."""
    csv_path = os.path.join(DATA_DIR_BW, f'{sid}_toe{kl}_raw_top{TOP_N}.csv')
    if not os.path.exists(csv_path):
        return None, f'csv 不存在: {os.path.basename(csv_path)}'

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    if any(c not in df.columns for c in RAW_STRESS_COLS + COORD_COLS):
        return None, 'csv 缺应力或坐标列'

    raw_stress = np.nan_to_num(df[RAW_STRESS_COLS].values)
    coords = df[COORD_COLS].values
    n = len(raw_stress)
    features = compute_derived_features(raw_stress)

    k_actual = min(K_NEIGHBORS, n - 1)
    if k_actual < 1:
        return None, f'节点数 {n} 过少'

    adj = kneighbors_graph(coords, n_neighbors=k_actual,
                           mode='connectivity', include_self=False)
    edges = np.array(adj.nonzero()).T
    edge_attr = compute_edge_features(coords, edges)
    hotspot_desc = compute_hotspot_descriptors(features, coords)

    return {
        'ID': sid,                   # 'BW20-1'  (无 _toe 后缀, 与 DJ/TX/UL 一致)
        'kl_side': int(kl),          # 记录哪一侧, 仅元数据
        'life': life,
        'log_life': life_to_y(life),
        'joint_type': jt,
        'corrosion_hours': ct,
        'censored': False,
        'n_nodes': n,
        'features': features,
        'coords': coords,
        'edge_index': edges,
        'edge_attr': edge_attr,
        'hotspot_descriptors': hotspot_desc,
    }, None


# ============================================================ #
# 主程序
# ============================================================ #
if not os.path.exists(LIFE_TABLE_PATH_BW):
    raise FileNotFoundError(f'未找到 BW 寿命表: {LIFE_TABLE_PATH_BW}')

df_bw = pd.read_excel(LIFE_TABLE_PATH_BW)
print(f'BW 寿命表: {len(df_bw)} 行 ({LIFE_TABLE_PATH_BW})')
print(f'  KL 标注: {df_bw[COL_TOE].notnull().sum()}/{len(df_bw)} 根')
print(f'  KL 分布: toe0={(df_bw[COL_TOE]==0).sum()}, toe1={(df_bw[COL_TOE]==1).sum()}')
print(f'★ 仅为开裂侧 KL 构图, 每根试件 1 个图 (与 DJ/TX/UL 训练集一致)\n')

success = 0
missing = []

for _, row in df_bw.iterrows():
    sid, life, kl = row[COL_ID], row[COL_LIFE], row[COL_TOE]
    if pd.isna(life) or life <= 0:
        print(f'  ⏭️ 跳过 {sid}: N 无效 (life={life})')
        continue
    if pd.isna(kl):
        print(f'  ⏭️ 跳过 {sid}: KL 未标注')
        continue

    kl = int(kl); jt, ct = parse_id(sid)
    sp, err = load_and_build_bw(sid, kl, life=float(life), jt=jt, ct=ct)
    if sp is None:
        missing.append(f'{sid} ({err})')
        continue

    with open(os.path.join(GRAPH_DIR_BW, f'{sid}.pkl'), 'wb') as f:
        pickle.dump(sp, f)
    success += 1
    print(f'  [BW] {sid} (KL=toe{kl}): {sp["n_nodes"]} 节点, '
          f'σ₁_max={sp["features"][:, 0].max():.0f}MPa, true_N={int(life)}')

if missing:
    print(f'\n--- 缺失/失败 ({len(missing)}) ---')
    for m in missing:
        print(f'  [缺失] {m}')

print(f'\n完成! 生成 {success} 个 BW 图 -> {GRAPH_DIR_BW}')
