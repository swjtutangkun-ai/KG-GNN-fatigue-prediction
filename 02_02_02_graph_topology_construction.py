# -*- coding: utf-8 -*-
"""
脚本 1: 构建单图.
  节点 3 维 [σ₁, τ_max, σ_vm]
  边   3 维 [dx, dy, dz] 有向差值
"""
import os, glob, pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.cluster import DBSCAN

from config_single import (
    LIFE_TABLE_PATH, DATA_DIR, GRAPH_DIR, TOP_N,
    RAW_STRESS_COLS, COORD_COLS,
    COL_ID, COL_TOE, COL_LIFE, parse_id,
    K_NEIGHBORS, USE_LOG10,
)

os.makedirs(GRAPH_DIR, exist_ok=True)

# 清理旧 pkl 避免上次运行的占位数据残留
for fp in glob.glob(os.path.join(GRAPH_DIR, '*.pkl')):
    os.remove(fp)


def life_to_y(life):
    return float(np.log10(life)) if USE_LOG10 else float(np.log1p(life))


def is_valid_life(life):
    if life is None:
        return False
    try:
        if pd.isna(life):
            return False
    except (TypeError, ValueError):
        return False
    try:
        v = float(life)
    except (TypeError, ValueError):
        return False
    return np.isfinite(v) and v > 0


def compute_derived_features(raw_stress):
    """3 维节点特征 [σ₁, τ_max, σ_vm]."""
    N = len(raw_stress)
    S11, S22, S33 = raw_stress[:, 0], raw_stress[:, 1], raw_stress[:, 2]
    S12, S13, S23 = raw_stress[:, 3], raw_stress[:, 4], raw_stress[:, 5]

    tensors = np.zeros((N, 3, 3))
    tensors[:, 0, 0] = S11; tensors[:, 1, 1] = S22; tensors[:, 2, 2] = S33
    tensors[:, 0, 1] = tensors[:, 1, 0] = S12
    tensors[:, 0, 2] = tensors[:, 2, 0] = S13
    tensors[:, 1, 2] = tensors[:, 2, 1] = S23
    eigvals = np.linalg.eigvalsh(tensors)
    sigma1 = eigvals[:, 2]
    sigma3 = eigvals[:, 0]
    tau_max = (sigma1 - sigma3) / 2.0

    sigma_vm = np.sqrt(0.5 * ((S11 - S22) ** 2 + (S22 - S33) ** 2 + (S33 - S11) ** 2
                              + 6 * (S12 ** 2 + S13 ** 2 + S23 ** 2)))

    return np.column_stack([sigma1, tau_max, sigma_vm])


def compute_edge_features(coords, edges):
    """边特征 3 维: 有向差值 coords[dst] - coords[src]."""
    if len(edges) == 0:
        return np.zeros((0, 3))
    src, dst = edges[:, 0], edges[:, 1]
    return coords[dst] - coords[src]


def compute_hotspot_descriptors(features, coords):
    """6 维热点场描述符. features 第 0 列是 σ₁."""
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
        return {
            'F_concentration': F_concentration, 'F_volume': F_volume,
            'F_gradient': 0.0, 'F_radius': 0.0,
            'F_n_hotspots': 1.0, 'F_separation': 0.0,
        }

    high_coords = coords[high_mask]
    k_nn = min(5, n_high - 1)
    nn = NearestNeighbors(n_neighbors=k_nn)
    nn.fit(high_coords)
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
        return {
            'F_concentration': F_concentration, 'F_volume': F_volume,
            'F_gradient': F_gradient, 'F_radius': F_radius,
            'F_n_hotspots': 1.0, 'F_separation': 0.0,
        }

    cluster_sizes = {l: int(np.sum(labels == l)) for l in unique_labels}
    main_label = max(cluster_sizes, key=cluster_sizes.get)
    main_coords = high_coords[labels == main_label]
    centroid = main_coords.mean(axis=0)
    F_radius = float(np.sqrt(np.mean(np.sum((main_coords - centroid) ** 2, axis=1))))
    F_gradient = float((s_max - s_mean) / (F_radius + 1e-8))

    if n_clusters >= 2:
        centroids = np.array([high_coords[labels == l].mean(axis=0) for l in unique_labels])
        n_c = len(centroids)
        total_dist = 0.0
        n_pairs = 0
        for i in range(n_c):
            for j in range(i + 1, n_c):
                total_dist += np.sqrt(np.sum((centroids[i] - centroids[j]) ** 2))
                n_pairs += 1
        avg_dist = total_dist / max(n_pairs, 1)
        F_separation = float(avg_dist / (F_radius + 1e-8))
    else:
        F_separation = 0.0

    return {
        'F_concentration': F_concentration, 'F_volume': F_volume,
        'F_gradient': F_gradient, 'F_radius': F_radius,
        'F_n_hotspots': F_n_hotspots, 'F_separation': F_separation,
    }


def load_and_build(sid, toe, life, censored, jt, ct):
    csv_path = os.path.join(DATA_DIR, f'{sid}_toe{toe}_raw_top{TOP_N}.csv')
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    if any(c not in df.columns for c in RAW_STRESS_COLS + COORD_COLS):
        return None

    raw_stress = np.nan_to_num(df[RAW_STRESS_COLS].values)
    coords = df[COORD_COLS].values
    n = len(raw_stress)

    features = compute_derived_features(raw_stress)            # (N, 3)

    k_actual = min(K_NEIGHBORS, n - 1)
    if k_actual < 1:
        return None

    adj = kneighbors_graph(coords, n_neighbors=k_actual,
                           mode='connectivity', include_self=False)
    edges = np.array(adj.nonzero()).T
    edge_attr = compute_edge_features(coords, edges)           # (E, 3)
    hotspot_desc = compute_hotspot_descriptors(features, coords)

    return {
        'ID': sid, 'toe': toe, 'life': life,
        'log_life': life_to_y(life),
        'joint_type': jt, 'corrosion_time': ct,
        'censored': censored,
        'n_nodes': n,
        'features': features,
        'coords': coords,
        'edge_index': edges,
        'edge_attr': edge_attr,
        'hotspot_descriptors': hotspot_desc,
    }


# ==================== 主程序 ==================== #
df_all = pd.read_excel(LIFE_TABLE_PATH)
df_life = df_all[df_all[COL_TOE].notnull()].copy()
df_life[COL_TOE] = df_life[COL_TOE].astype(int)
df_runout = df_all[df_all[COL_TOE].isnull()].copy()
print(f'主表: {len(df_life)} 失效, {len(df_runout)} 未开裂, TOP{TOP_N}, KNN k={K_NEIGHBORS}')
print(f'节点 3 维 [σ₁,τ_max,σ_vm], 边 3 维 [dx,dy,dz] 有向, 目标 {"log10(N)" if USE_LOG10 else "log(1+N)"}\n')

failed_toes = {(row[COL_ID], row[COL_TOE]): row[COL_LIFE] for _, row in df_life.iterrows()}

success_failed = 0
n_skipped_failed = 0
for _, row in df_life.iterrows():
    sid, toe, life = row[COL_ID], row[COL_TOE], row[COL_LIFE]
    if not is_valid_life(life):
        print(f"  ⏭️ [跳过失效] {sid}_toe{toe}: N 列无效 (life={life})")
        n_skipped_failed += 1
        continue
    jt, ct = parse_id(sid)
    specimen = load_and_build(sid, toe, life, censored=False, jt=jt, ct=ct)
    if specimen is None:
        continue

    with open(os.path.join(GRAPH_DIR, f'{sid}_toe{toe}.pkl'), 'wb') as f:
        pickle.dump(specimen, f)
    success_failed += 1
    if success_failed <= 3:
        print(f'  [失效] {sid}_toe{toe}: {specimen["n_nodes"]} 节点, '
              f'feat={specimen["features"].shape}, '
              f'σ₁_max={specimen["features"][:, 0].max():.0f}MPa, life={life}')

print(f'\n--- 扫描 DJ/TX/UL 未失效焊趾 ---')
success_censored = 0
missing_censored = []
for sid in df_life[COL_ID].unique():
    jt, ct = parse_id(sid)
    if jt not in ('TX', 'DJ', 'UL'):
        continue

    failed_for_id = {toe for (s, toe) in failed_toes if s == sid}
    for other_toe in [0, 1]:
        if other_toe in failed_for_id:
            continue
        life_lower = max(failed_toes[(sid, t)] for t in failed_for_id)
        specimen = load_and_build(sid, other_toe, life_lower, censored=True, jt=jt, ct=ct)
        if specimen is None:
            missing_censored.append(f'{sid}_toe{other_toe}')
            continue

        with open(os.path.join(GRAPH_DIR, f'{sid}_toe{other_toe}_censored.pkl'), 'wb') as f:
            pickle.dump(specimen, f)
        success_censored += 1
        if success_censored <= 5:
            print(f'  [删失] {sid}_toe{other_toe}: {specimen["n_nodes"]} 节点, life>={life_lower}')

print(f'\n--- 扫描未开裂试件 ---')
success_runout = 0
n_skipped_runout = 0
for _, row in df_runout.iterrows():
    sid, life = row[COL_ID], row[COL_LIFE]
    if not is_valid_life(life):
        print(f"  ⏭️ [跳过未开裂] {sid}: N 列无效 (life={life})")
        n_skipped_runout += 1
        continue
    jt, ct = parse_id(sid)
    for toe in [0, 1]:
        specimen = load_and_build(sid, toe, life, censored=True, jt=jt, ct=ct)
        if specimen is None:
            missing_censored.append(f'{sid}_toe{toe} (未开裂)')
            continue

        with open(os.path.join(GRAPH_DIR, f'{sid}_toe{toe}_censored.pkl'), 'wb') as f:
            pickle.dump(specimen, f)
        success_censored += 1
        success_runout += 1
        print(f'  [未开裂] {sid}_toe{toe}: {specimen["n_nodes"]} 节点, life>={life}')

if missing_censored:
    print(f'\n--- 未找到 CSV 的删失焊趾 ({len(missing_censored)}) ---')
    for m in missing_censored:
        print(f'  [缺失] {m}')

print(f'\n完成! 失效: {success_failed}, 删失: {success_censored} '
      f'(其中未开裂: {success_runout}), 缺失: {len(missing_censored)}')
if n_skipped_failed + n_skipped_runout > 0:
    print(f'⏭️ 跳过: 失效={n_skipped_failed}, 未开裂={n_skipped_runout}')
