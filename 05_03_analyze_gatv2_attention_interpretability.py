# -*- coding: utf-8 -*-
"""
analyze_attention_deep.py v3 — 注意力驱动因子分析 (6 个核心物理量)

6 个特征 (3 类):
  A) 应力幅值 (3):    sigma1, tau_max, sigma_vm  (模型输入)
  B) 空间特征 (2):    dist_to_hotspot, grad_sigma1
  C) 多轴状态 (1):    triaxiality

用法:
  python analyze_attention_deep.py --result_dir ./v11_final --out_dir ./v11_analysis
"""
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from plot_style import (
    apply_style, savefig_multi,
    SINGLE_COL, ONE_HALF_COL, DOUBLE_COL,
    JT_COLORS, MACARON_LAVENDER, MACARON_APRICOT, MACARON_GREEN,
    MACARON_ROSE, MACARON_BROWN, IIW_REF_COLOR,
)
apply_style()
JT_ORDER = ['DJ', 'TX', 'UL']
CT_ORDER = [0, 20, 40, 60]

# ★ 6 个核心物理量
SELECTED_FEATURES = [
    'sigma1', 'tau_max', 'sigma_vm',
    'dist_to_hotspot',
    'grad_sigma1',
    'triaxiality',
]

FEAT_DISPLAY = {
    'tau_max':         r'$\tau_{\max}$',
    'sigma_vm':        r'$\sigma_{\mathrm{vm}}$',
    'sigma1':          r'$\sigma_1$',
    'dist_to_hotspot': r'$d_{\mathrm{hotspot}}$',
    'grad_sigma1':     r'$|\nabla\sigma_1|$',
    'triaxiality':     r'$\eta$  (triaxiality)',
}


def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)
def save_table(df, path):
    df.to_csv(path, index=False, encoding='utf-8-sig', float_format='%.6g')
    print(f'  ✓ {path.name}  ({len(df)} rows × {len(df.columns)} cols)')


def _get_config_defaults():
    d = {'data_dir': './toe_sdv_tables/sigma1_top500', 'top_n': 500}
    try:
        from config_single import DATA_DIR, TOP_N
        d['data_dir'], d['top_n'] = DATA_DIR, TOP_N
    except ImportError:
        pass
    return d


def load_topN_coords(data_dir, top_n):
    pattern = re.compile(rf'^(.+)_toe(\d+)_raw_top{top_n}\.csv$')
    coords_by_id = {}
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f'⚠️  data_dir not found: {data_dir}'); return coords_by_id
    for fp in sorted(data_dir.glob(f'*_top{top_n}.csv')):
        m = pattern.match(fp.name)
        if not m: continue
        sid = m.group(1)
        if sid in coords_by_id: continue
        df = pd.read_csv(fp)
        df.columns = [c.strip() for c in df.columns]
        if not {'X', 'Y', 'Z'}.issubset(df.columns): continue
        sc = ['S11_peak', 'S22_peak', 'S33_peak', 'S12_peak', 'S13_peak', 'S23_peak']
        if not all(c in df.columns for c in sc): continue
        # 完整应力张量特征值分解
        raw = np.nan_to_num(df[sc].values)
        tens = np.zeros((len(raw), 3, 3))
        tens[:, 0, 0] = raw[:, 0]; tens[:, 1, 1] = raw[:, 1]; tens[:, 2, 2] = raw[:, 2]
        tens[:, 0, 1] = tens[:, 1, 0] = raw[:, 3]
        tens[:, 0, 2] = tens[:, 2, 0] = raw[:, 4]
        tens[:, 1, 2] = tens[:, 2, 1] = raw[:, 5]
        eigvals = np.linalg.eigvalsh(tens)  # 升序: σ₃, σ₂, σ₁
        df['sigma3'] = eigvals[:, 0]
        df['sigma2'] = eigvals[:, 1]
        df['sigma1'] = eigvals[:, 2]
        df['tau_max_raw'] = (df['sigma1'] - df['sigma3']) / 2.0
        df['sigma_vm_raw'] = np.sqrt(0.5 * (
            (raw[:, 0] - raw[:, 1]) ** 2 + (raw[:, 1] - raw[:, 2]) ** 2 +
            (raw[:, 2] - raw[:, 0]) ** 2 + 6 * (raw[:, 3] ** 2 + raw[:, 4] ** 2 + raw[:, 5] ** 2)))
        df = df.reset_index(drop=True)
        df['node_idx'] = df.index
        coords_by_id[sid] = df
    print(f'  loaded {len(coords_by_id)} specimens from {data_dir}')
    return coords_by_id


# ================================================================ #
# 计算 30 个候选特征
# ================================================================ #
def compute_all_features(cdf, k_neighbors=8):
    """返回 DataFrame (N 行, 每列一个特征)."""
    N = len(cdf)
    xyz = cdf[['X', 'Y', 'Z']].values
    s1 = cdf['sigma1'].values
    s2 = cdf['sigma2'].values
    s3 = cdf['sigma3'].values
    tm = cdf['tau_max_raw'].values
    vm = cdf['sigma_vm_raw'].values

    f = {}

    # ========== A) 应力幅值 (5) ========== #
    f['sigma1'] = s1
    f['sigma2'] = s2
    f['sigma3'] = s3
    f['tau_max'] = tm
    f['sigma_vm'] = vm

    # ========== B) 应力不变量与多轴状态 (8) ========== #
    I1 = s1 + s2 + s3
    sigma_h = I1 / 3.0
    safe_vm = np.clip(vm, 1e-8, None)

    f['I1'] = I1
    f['sigma_h'] = sigma_h
    f['triaxiality'] = sigma_h / safe_vm           # η = σ_h / σ_vm
    # Lode 角参数 θ̄ ∈ [-1, 1]:  θ̄ = 1 − (6θ/π)
    #   其中 cos(3θ) = (3√3/2) · J₃/J₂^(3/2)
    J2 = (1.0 / 6.0) * ((s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2)
    s_dev1 = s1 - sigma_h; s_dev2 = s2 - sigma_h; s_dev3 = s3 - sigma_h
    J3 = s_dev1 * s_dev2 * s_dev3
    safe_J2 = np.clip(J2, 1e-16, None)
    cos3theta = np.clip((3 * np.sqrt(3) / 2) * J3 / (safe_J2 ** 1.5), -1, 1)
    theta = np.arccos(cos3theta) / 3.0
    f['lode_angle'] = 1.0 - 6.0 * theta / np.pi   # θ̄

    safe_s1 = np.where(np.abs(s1) > 1e-8, s1, 1e-8)
    f['biaxiality'] = s2 / safe_s1                  # σ₂/σ₁
    f['tau_over_sigma1'] = tm / np.abs(safe_s1)
    f['vm_over_sigma1'] = vm / np.abs(safe_s1)
    f['swt_like'] = np.abs(s1) * vm                 # 类 SWT 参数

    # ========== C) 空间梯度 (6) ========== #
    k = min(k_neighbors, N - 1)
    if k < 2:
        for gn in ['grad_sigma1', 'grad_tau_max', 'grad_sigma_vm',
                    'laplacian_sigma1', 'grad_direction_z', 'grad_triaxiality']:
            f[gn] = np.zeros(N)
    else:
        tree = cKDTree(xyz)
        dists, indices = tree.query(xyz, k=k + 1)
        nbr_d = dists[:, 1:]
        nbr_i = indices[:, 1:]
        safe_d = np.clip(nbr_d, 1e-6, None)

        # C1–C3: 一阶梯度 |∇f| = mean(|f_i - f_j| / d_ij)
        for vals, name in [(s1, 'grad_sigma1'), (tm, 'grad_tau_max'), (vm, 'grad_sigma_vm')]:
            diff = np.abs(vals[nbr_i] - vals[:, None])
            f[name] = np.mean(diff / safe_d, axis=1)

        # C4: 二阶梯度 (Laplacian 近似) = mean(f_j - f_i) / mean(d_ij²)
        #     正值 = 局部凹 (应力集中 "底部"), 负值 = 局部凸 (应力集中 "尖峰")
        nbr_s1 = s1[nbr_i]
        lap = np.mean(nbr_s1 - s1[:, None], axis=1) / np.mean(safe_d ** 2, axis=1)
        f['laplacian_sigma1'] = lap

        # C5: 梯度方向的 Z 分量占比 (沿厚度方向集中程度)
        #     对每个邻居, 计算 (s1_j - s1_i) * (z_j - z_i) / (d_ij * |Δs1|)
        dxyz = xyz[nbr_i] - xyz[:, None, :]          # (N, k, 3)
        ds1 = nbr_s1 - s1[:, None]                    # (N, k)
        abs_ds1 = np.abs(ds1) + 1e-8
        z_component = np.abs(dxyz[:, :, 2]) / safe_d  # |Δz/d| ∈ [0,1]
        # 加权平均: 梯度越大的邻居权重越高
        weights = abs_ds1 / abs_ds1.sum(axis=1, keepdims=True)
        f['grad_direction_z'] = np.sum(weights * z_component, axis=1)

        # C6: 三轴度梯度 |∇η|
        eta = f['triaxiality']
        diff_eta = np.abs(eta[nbr_i] - eta[:, None])
        f['grad_triaxiality'] = np.mean(diff_eta / safe_d, axis=1)

    # ========== D) 空间位置与距离 (5) ========== #
    # D1: 到 σ₁ 加权质心
    w = np.clip(s1, 0, None); w_sum = w.sum() + 1e-8
    centroid = (xyz * w[:, None]).sum(axis=0) / w_sum
    f['dist_to_s1_centroid'] = np.sqrt(np.sum((xyz - centroid) ** 2, axis=1))

    # D2: 到 DBSCAN 热点簇中心
    thr = np.percentile(s1, 95)
    hmask = s1 >= thr
    hotspot_center = xyz[np.argmax(s1)]  # fallback
    if hmask.sum() >= 5:
        hxyz = xyz[hmask]
        nn = NearestNeighbors(n_neighbors=min(5, hmask.sum() - 1))
        nn.fit(hxyz)
        d_nn, _ = nn.kneighbors(hxyz)
        eps = float(np.median(d_nn[:, -1]) * 1.5)
        db = DBSCAN(eps=eps, min_samples=3).fit(hxyz)
        labels = db.labels_
        uniq = set(labels) - {-1}
        if len(uniq) > 0:
            biggest = max(uniq, key=lambda l: np.sum(labels == l))
            hotspot_center = hxyz[labels == biggest].mean(axis=0)
    f['dist_to_hotspot'] = np.sqrt(np.sum((xyz - hotspot_center) ** 2, axis=1))

    # D3: 到最大 σ₁ 节点
    f['dist_to_max_sigma1'] = np.sqrt(np.sum((xyz - xyz[np.argmax(s1)]) ** 2, axis=1))

    # D4: Z 坐标 (厚度方向深度)
    f['Z_coord'] = xyz[:, 2]

    # D5: 局部点密度 = 1 / 平均邻居距离
    if k >= 2:
        f['neighbor_density'] = 1.0 / (np.mean(nbr_d, axis=1) + 1e-8)
    else:
        f['neighbor_density'] = np.ones(N)

    # ========== E) 局部统计 (4) ========== #
    if k >= 2:
        nbr_s1_vals = s1[nbr_i]
        f['local_sigma1_std'] = np.std(nbr_s1_vals, axis=1)
        nbr_mean = np.mean(nbr_s1_vals, axis=1)
        nbr_std = np.std(nbr_s1_vals, axis=1) + 1e-8
        f['node_anomaly'] = (s1 - nbr_mean) / nbr_std
        f['nbr_sigma1_max'] = np.max(nbr_s1_vals, axis=1)
        f['nbr_sigma1_range'] = np.ptp(nbr_s1_vals, axis=1)
    else:
        for n in ['local_sigma1_std', 'node_anomaly', 'nbr_sigma1_max', 'nbr_sigma1_range']:
            f[n] = np.zeros(N)

    # ========== F) 交叉/组合 (2) ========== #
    f['sigma1_times_grad'] = np.abs(s1) * f.get('grad_sigma1', np.zeros(N))
    dist_hp = f['dist_to_hotspot']
    f['stress_intensity_proxy'] = np.abs(s1) / np.sqrt(np.clip(dist_hp, 0.1, None))

    return pd.DataFrame(f)


# ================================================================ #
# 主分析
# ================================================================ #
def run_deep_analysis(df_attn_mean, coords_by_id, df_ens, out_tbl, out_prev):
    print('\n[深度分析] 30 候选物理量 × 114 试件...')

    all_rho_rows = []
    all_scatter_rows = []
    n_done = 0

    for sid, attn_sub in df_attn_mean.groupby('ID'):
        coords = coords_by_id.get(sid)
        if coords is None: continue
        if len(attn_sub) < 10: continue

        feat_df = compute_all_features(coords)
        # ★ 只保留筛选后的 12 个特征
        feat_df = feat_df[[c for c in SELECTED_FEATURES if c in feat_df.columns]]
        feat_names = list(feat_df.columns)

        attn_vals = attn_sub.sort_values('node_idx')['attention'].values
        if len(attn_vals) != len(feat_df):
            feat_df['node_idx'] = feat_df.index
            merged = attn_sub[['node_idx', 'attention']].merge(feat_df, on='node_idx', how='inner')
            if len(merged) < 10: continue
            attn_vals = merged['attention'].values
            feat_df = merged[feat_names]

        ens_row = df_ens[df_ens['ID'] == sid]
        jt = ens_row['joint_type'].values[0] if len(ens_row) > 0 else 'UNK'
        ct = int(ens_row['corrosion_hours'].values[0]) if len(ens_row) > 0 else -1
        mre = float(ens_row['rel_error_ensemble'].values[0]) if len(ens_row) > 0 else np.nan

        for fname in feat_names:
            fv = feat_df[fname].values
            if np.std(fv) < 1e-12:
                rho, p = 0.0, 1.0
            else:
                rho, p = spearmanr(fv, attn_vals)
            all_rho_rows.append({
                'ID': sid, 'joint_type': jt, 'corrosion_hours': ct,
                'MRE': mre, 'feature': fname, 'rho': rho, 'p_value': p,
            })

        for ni in range(len(attn_vals)):
            row = {'ID': sid, 'joint_type': jt, 'corrosion_hours': ct,
                   'node_idx': ni, 'attention': float(attn_vals[ni])}
            for fn in feat_names:
                row[fn] = float(feat_df[fn].values[ni])
            all_scatter_rows.append(row)

        n_done += 1
        if n_done <= 3 or n_done % 20 == 0:
            print(f'    [{n_done}] {sid}: {len(feat_names)} features')

    print(f'  完成 {n_done} 个试件, {len(feat_names)} 特征')

    # ========== 导出 ========== #
    df_rho = pd.DataFrame(all_rho_rows)
    save_table(df_rho, out_tbl / 'deep_feature_rho_per_specimen.csv')

    # 排名
    feat_names = sorted(df_rho['feature'].unique())
    rows_s = []
    for fn in feat_names:
        sub = df_rho[df_rho['feature'] == fn]
        ar = sub['rho'].abs()
        rows_s.append({
            'feature': fn,
            'abs_rho_median': ar.median(), 'abs_rho_mean': ar.mean(),
            'abs_rho_q75': ar.quantile(0.75),
            'rho_median': sub['rho'].median(), 'rho_mean': sub['rho'].mean(),
            'rho_std': sub['rho'].std(),
            'pct_significant': (sub['p_value'] < 0.05).mean(),
            'n': len(sub),
        })
    df_sum = pd.DataFrame(rows_s).sort_values('abs_rho_median', ascending=False).reset_index(drop=True)
    df_sum.insert(0, 'rank', range(1, len(df_sum) + 1))
    save_table(df_sum, out_tbl / 'deep_feature_rho_summary.csv')

    # 分组
    grp_rows = []
    for fn in feat_names:
        for jt in JT_ORDER:
            for ct in CT_ORDER:
                sub = df_rho[(df_rho['feature'] == fn) &
                             (df_rho['joint_type'] == jt) &
                             (df_rho['corrosion_hours'] == ct)]
                if len(sub) == 0: continue
                grp_rows.append({
                    'feature': fn, 'joint_type': jt, 'corrosion_hours': ct,
                    'n': len(sub), 'rho_median': sub['rho'].median(),
                    'abs_rho_median': sub['rho'].abs().median(),
                })
    save_table(pd.DataFrame(grp_rows), out_tbl / 'deep_feature_rho_by_group.csv')

    # 最佳特征散点
    best = df_sum.iloc[0]['feature']
    print(f'\n  🏆 最佳: {best}  (|ρ| median = {df_sum.iloc[0]["abs_rho_median"]:.3f})')
    df_sc = pd.DataFrame(all_scatter_rows)
    cols_out = ['ID', 'joint_type', 'corrosion_hours', 'node_idx', 'attention']
    # top-5 特征都导出
    for i in range(min(5, len(df_sum))):
        fn = df_sum.iloc[i]['feature']
        if fn in df_sc.columns:
            cols_out.append(fn)
    save_table(df_sc[cols_out], out_tbl / 'deep_top5_feature_scatter.csv')

    # ================================================================ #
    # 可视化
    # ================================================================ #
    n_feat = len(df_sum)

    # ---- 图 1: 特征排名 ---- #
    # 按特征类别分色
    FEAT_CATEGORY = {
        'sigma1': 'A', 'tau_max': 'A', 'sigma_vm': 'A',
        'dist_to_hotspot': 'B',
        'grad_sigma1': 'B',
        'triaxiality': 'C',
    }
    CAT_COLORS = {'A': MACARON_BROWN, 'B': MACARON_GREEN, 'C': MACARON_APRICOT}
    CAT_LABELS = {'A': 'Stress amplitude', 'B': 'Spatial', 'C': 'Multiaxial'}

    FEAT_DISPLAY = {
        'tau_max':        r'$\tau_{\max}$',
        'sigma_vm':       r'$\sigma_{\mathrm{vm}}$',
        'sigma1':         r'$\sigma_1$',
        'dist_to_hotspot': r'$d_{\mathrm{hotspot}}$',
        'grad_sigma1':    r'$|\nabla\sigma_1|$',
        'triaxiality':    r'$\eta$  (triaxiality)',
    }

    fig, ax = plt.subplots(figsize=(ONE_HALF_COL, max(SINGLE_COL * 0.8, n_feat * 0.32)),
                           constrained_layout=True)
    feat_names = df_sum['feature'].values
    display_names = [FEAT_DISPLAY.get(f, f) for f in feat_names]
    colors = [CAT_COLORS.get(FEAT_CATEGORY.get(f, 'A'), MACARON_LAVENDER)
              for f in feat_names]
    y_pos = np.arange(n_feat)
    bars = ax.barh(y_pos, df_sum['abs_rho_median'].values,
                   color=colors, edgecolor='#555', lw=0.4, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel(r'Median |$\rho$| (Spearman)')
    ax.set_title('Attention–feature correlation', fontsize=9)
    for i, (bar, row_i) in enumerate(zip(bars, df_sum.itertuples())):
        sign = '+' if row_i.rho_median > 0 else '−'
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f'{row_i.abs_rho_median:.3f} ({sign})', va='center', fontsize=5.5)
    # 分类标注
    ax.axhline(-0.5, color='#ccc', lw=0.5)
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=CAT_COLORS[k], label=CAT_LABELS[k])
               for k in ['A', 'B', 'C']]
    ax.legend(handles=patches, fontsize=5.5, loc='lower right')
    savefig_multi(fig, out_prev / 'fig_deep_rho_ranking')
    plt.close(fig)

    # ---- 图 2: 热力图 Top-12 × JT ---- #
    n_show = min(12, n_feat)
    top_f = df_sum['feature'].values[:n_show]
    hm = np.zeros((n_show, len(JT_ORDER)))
    for i, fn in enumerate(top_f):
        for j, jt in enumerate(JT_ORDER):
            sub = df_rho[(df_rho['feature'] == fn) & (df_rho['joint_type'] == jt)]
            hm[i, j] = sub['rho'].median() if len(sub) > 0 else 0
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 0.9, SINGLE_COL * 1.5), constrained_layout=True)
    im = ax.imshow(hm, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(JT_ORDER))); ax.set_xticklabels(JT_ORDER, fontsize=8)
    ax.set_yticks(range(n_show)); ax.set_yticklabels(top_f, fontsize=6.5)
    ax.set_xlabel('Joint type')
    ax.set_title(r'Median $\rho$ by joint type', fontsize=9)
    for i in range(n_show):
        for j in range(len(JT_ORDER)):
            ax.text(j, i, f'{hm[i, j]:.2f}', ha='center', va='center',
                    fontsize=6, color='white' if abs(hm[i, j]) > 0.3 else '#333')
    fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02).set_label(r'$\rho$', fontsize=8)
    savefig_multi(fig, out_prev / 'fig_deep_rho_heatmap')
    plt.close(fig)

    # ---- 图 3: 热力图 Top-12 × corrosion ---- #
    hm_ct = np.zeros((n_show, len(CT_ORDER)))
    for i, fn in enumerate(top_f):
        for j, ct in enumerate(CT_ORDER):
            sub = df_rho[(df_rho['feature'] == fn) & (df_rho['corrosion_hours'] == ct)]
            hm_ct[i, j] = sub['rho'].median() if len(sub) > 0 else 0
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 1.5), constrained_layout=True)
    im = ax.imshow(hm_ct, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(CT_ORDER))); ax.set_xticklabels([f'{c}h' for c in CT_ORDER], fontsize=8)
    ax.set_yticks(range(n_show)); ax.set_yticklabels(top_f, fontsize=6.5)
    ax.set_xlabel('Corrosion hours')
    ax.set_title(r'Median $\rho$ by corrosion level', fontsize=9)
    for i in range(n_show):
        for j in range(len(CT_ORDER)):
            ax.text(j, i, f'{hm_ct[i, j]:.2f}', ha='center', va='center',
                    fontsize=6, color='white' if abs(hm_ct[i, j]) > 0.3 else '#333')
    fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02).set_label(r'$\rho$', fontsize=8)
    savefig_multi(fig, out_prev / 'fig_deep_rho_heatmap_corrosion')
    plt.close(fig)

    # ---- 图 4: Top-1 特征散点 + 直方图 ---- #
    sub_rho = df_rho[df_rho['feature'] == best]
    med_abs = sub_rho['rho'].abs().median()
    idx = (sub_rho['rho'].abs() - med_abs).abs().idxmin()
    rep_id, rep_rho = sub_rho.loc[idx, 'ID'], sub_rho.loc[idx, 'rho']
    rep_data = df_sc[df_sc['ID'] == rep_id]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(ONE_HALF_COL, SINGLE_COL * 0.9),
                                   constrained_layout=True)
    if best in rep_data.columns:
        ax1.scatter(rep_data[best], rep_data['attention'],
                    s=10, color=MACARON_LAVENDER, edgecolor='white', lw=0.3, alpha=0.8)
    ax1.set_xlabel(best.replace('_', ' '), fontsize=8)
    ax1.set_ylabel('Attention weight')
    ax1.set_title(f'{rep_id}  ($\\rho$ = {rep_rho:.2f})', fontsize=9)

    vals = sub_rho['rho'].values
    ax2.hist(vals, bins=25, color=MACARON_LAVENDER, edgecolor='#555', lw=0.5, alpha=0.85)
    ax2.axvline(np.median(vals), color=MACARON_BROWN, ls='--', lw=0.9,
                label=f'Median = {np.median(vals):.2f}')
    ax2.axvline(0, color='#888', ls=':', lw=0.6)
    ax2.set_xlabel(r'Spearman $\rho$')
    ax2.set_ylabel('Number of specimens')
    ax2.legend(fontsize=7)
    savefig_multi(fig, out_prev / 'fig_deep_best_scatter')
    plt.close(fig)

    # ---- 图 5: Top-4 特征 × 3 接头类型 ρ 分布 (4×3 面板) ---- #
    top4 = df_sum['feature'].values[:4]
    fig, axes = plt.subplots(4, 3, figsize=(DOUBLE_COL, DOUBLE_COL * 0.8),
                             constrained_layout=True, sharex=True, sharey=True)
    for i, fn in enumerate(top4):
        for j, jt in enumerate(JT_ORDER):
            ax = axes[i, j]
            sub = df_rho[(df_rho['feature'] == fn) & (df_rho['joint_type'] == jt)]
            if len(sub) > 0:
                ax.hist(sub['rho'].values, bins=15, color=JT_COLORS[jt],
                        edgecolor='#555', lw=0.4, alpha=0.85)
                med = sub['rho'].median()
                ax.axvline(med, color=MACARON_BROWN, ls='--', lw=0.7)
                ax.text(0.95, 0.92, f'med={med:.2f}', transform=ax.transAxes,
                        ha='right', va='top', fontsize=6)
            ax.axvline(0, color='#aaa', ls=':', lw=0.5)
            if i == 0: ax.set_title(jt, fontsize=9)
            if j == 0: ax.set_ylabel(fn.replace('_', '\n'), fontsize=6.5)
            if i == 3: ax.set_xlabel(r'$\rho$', fontsize=8)
    savefig_multi(fig, out_prev / 'fig_deep_top4_panel')
    plt.close(fig)

    # ---- 控制台排名 ---- #
    print('\n' + '=' * 72)
    print(f'  注意力驱动因子排名 (30 物理量, 按 |ρ| 中位数)')
    print('=' * 72)
    for _, r in df_sum.iterrows():
        sign = '+' if r['rho_median'] > 0 else '−'
        print(f"  #{int(r['rank']):2d}  {r['feature']:28s}  "
              f"|ρ|={r['abs_rho_median']:.3f}  "
              f"ρ={r['rho_median']:+.3f}  "
              f"sig={r['pct_significant']*100:.0f}%")
    print('=' * 72)


def main():
    cfg = _get_config_defaults()
    ap = argparse.ArgumentParser()
    ap.add_argument('--result_dir', default='./v11_final')
    ap.add_argument('--data_dir', default=cfg['data_dir'])
    ap.add_argument('--top_n', type=int, default=cfg['top_n'])
    ap.add_argument('--out_dir', default='./v11_analysis')
    args = ap.parse_args()

    result_dir, out_dir = Path(args.result_dir), Path(args.out_dir)
    out_tbl, out_prev = out_dir / 'tables', out_dir / 'previews'
    ensure_dir(out_tbl); ensure_dir(out_prev)

    df_attn = pd.read_csv(result_dir / 'attention_export.csv')
    df_ens = pd.read_csv(result_dir / 'specimen_ensemble.csv')
    print(f'Reading: {result_dir.resolve()}')
    print(f'  {len(df_attn)} attention rows, {df_attn["ID"].nunique()} specimens')

    df_am = df_attn.groupby(['ID', 'joint_type', 'corrosion_hours', 'node_idx']).agg(
        attention=('attention', 'mean'), sigma1=('sigma1', 'mean'),
        tau_max=('tau_max', 'mean'), sigma_vm=('sigma_vm', 'mean')).reset_index()

    coords = load_topN_coords(args.data_dir, args.top_n)
    if not coords:
        print('⚠️ 无坐标数据, 无法计算空间特征。用 --data_dir 指定 top500 CSV 目录。')
        return

    run_deep_analysis(df_am, coords, df_ens, out_tbl, out_prev)
    print(f'\nDone → {out_dir.resolve()}')


if __name__ == '__main__':
    main()