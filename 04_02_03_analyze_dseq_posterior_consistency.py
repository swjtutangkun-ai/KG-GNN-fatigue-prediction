# -*- coding: utf-8 -*-
"""
analyze_dSeq_posterior.py — ΔSeq 后验物理一致性分析 (top-5% 版)

对照量: FEA 应力场 σ₁ / σ_vm / τ_max 的 top-5% 均值 (500 节点取前 25 个平均).
选 top-5%: 聚焦焊趾热点区, 避开单点峰值的数值敏感性, 也不被整场平均稀释.

输出:
  master_table.csv        — 每试件 ΔSeq + 3 个 top-5% 统计量 + 分组信息
  correlation_table.csv   — 总体 + DJ/TX/UL 分组的 Spearman + log-log Pearson
  origin_data_panel_{1,2,3}.csv — 3 个面板各自的原始散点数据 (Origin 直接导入)
  scatter_3panel.png/pdf  — 1×3 预览图 (最终画图以 Origin 为准)

用法:
  python analyze_dSeq_posterior.py
  python analyze_dSeq_posterior.py --result_dir ./v11_final \
                                    --graph_dir ./single_level_graphs \
                                    --out_dir ./dSeq_posterior
"""
import os
import argparse
import glob
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt


# ============================================================ #
# 3 个候选量
# ============================================================ #
# (列名, 组件索引, 显示标签, 短名)
# features[:, 0]=sigma1, [:, 1]=tau_max, [:, 2]=sigma_vm
STATS_SPEC = [
    ('sigma1_top5pct',   0, r'$\sigma_{1,\mathrm{top5\%}}$',       'sigma1'),
    ('sigma_vm_top5pct', 2, r'$\sigma_{vm,\mathrm{top5\%}}$',      'sigma_vm'),
    ('tau_max_top5pct',  1, r'$\tau_{\max,\mathrm{top5\%}}$',      'tau_max'),
]

JT_COLORS = {
    'DJ': '#5C8AAE',
    'TX': '#D4A574',
    'UL': '#B06C88',
}


# ============================================================ #
# 工具
# ============================================================ #
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def top5pct_mean(arr):
    """top-5% 均值: 500 节点取前 25 个平均."""
    n = len(arr)
    k = max(1, int(round(n * 0.05)))
    return float(np.sort(arr)[-k:].mean())


def extract_stats_from_pkl(graph_dir):
    """扫描 single_level_graphs/*.pkl, 对每个失效 ID 提取 3 个 top-5% 统计量.

    多 toe 文件 (DJ20-1_toe0.pkl + DJ20-1_toe1.pkl) 跨 toe 取 max,
    与 specimen_ensemble.csv 的 ID 聚合口径一致.
    """
    per_id = {}
    files = sorted(glob.glob(os.path.join(graph_dir, '*.pkl')))
    n_censored = n_failed = 0

    for fp in files:
        if 'censored' in os.path.basename(fp):
            n_censored += 1
            continue
        with open(fp, 'rb') as f:
            sp = pickle.load(f)
        if sp.get('censored', False):
            continue

        sid = sp['ID']
        feats = sp['features']   # (N, 3), 原始 MPa
        stats = {name: top5pct_mean(feats[:, col_idx])
                 for name, col_idx, _, _ in STATS_SPEC}

        if sid not in per_id:
            per_id[sid] = stats
        else:
            for k in stats:
                per_id[sid][k] = max(per_id[sid][k], stats[k])
        n_failed += 1

    print(f'  ✓ 扫描 {len(files)} 个 pkl: 失效 {n_failed}, 跳过 censored {n_censored}')
    print(f'  ✓ 得到 {len(per_id)} 个唯一失效 ID 的 top-5% 统计量')
    return per_id


def build_master_table(ensemble_csv, peak_map):
    df_ens = pd.read_csv(ensemble_csv)
    rows = []
    missing = []
    for _, r in df_ens.iterrows():
        sid = r['ID']
        row = {
            'ID': sid,
            'joint_type': r['joint_type'],
            'corrosion_hours': r.get('corrosion_hours', np.nan),
            'true_log10N': r['true_log10N'],
            'true_N': r['true_N'],
            'dSeq': r['DS_MPa_mean'],
        }
        if sid in peak_map:
            row.update(peak_map[sid])
        else:
            for name, *_ in STATS_SPEC:
                row[name] = np.nan
            missing.append(sid)
        rows.append(row)

    df = pd.DataFrame(rows)
    if missing:
        print(f'  ⚠️  {len(missing)} 个 ID 在 pkl 中缺失, 前 5: {missing[:5]}')
    return df


# ============================================================ #
# 相关分析
# ============================================================ #
def safe_spearman(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan, int(mask.sum())
    rho, pval = spearmanr(x[mask], y[mask])
    return float(rho), float(pval), int(mask.sum())


def safe_pearson_loglog(x, y):
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if mask.sum() < 3:
        return np.nan, np.nan, int(mask.sum())
    r, pval = pearsonr(np.log10(x[mask]), np.log10(y[mask]))
    return float(r), float(pval), int(mask.sum())


def correlation_table(df, group='overall'):
    rows = []
    dSeq = df['dSeq'].values
    for name, _, _, comp in STATS_SPEC:
        stat = df[name].values
        rho, p_s, n_s = safe_spearman(dSeq, stat)
        r, p_p, _ = safe_pearson_loglog(dSeq, stat)
        rows.append({
            'group': group, 'stat': name, 'component': comp,
            'spearman_rho': rho, 'spearman_p': p_s,
            'pearson_loglog_r': r, 'pearson_loglog_p': p_p,
            'n': n_s,
        })
    return pd.DataFrame(rows)


# ============================================================ #
# Origin 原始数据导出
# ============================================================ #
def export_origin_data(df_clean, out_dir):
    """每个子图一个 CSV, 按接头类型分列, 方便 Origin 直接分组画散点.

    列格式:
        DJ_x, DJ_y, TX_x, TX_y, UL_x, UL_y
        (其中 x = FEA 统计量, y = dSeq)
    """
    for panel_idx, (name, _, label, _) in enumerate(STATS_SPEC, start=1):
        cols = {}
        max_len = 0
        for jt in ['DJ', 'TX', 'UL']:
            sub = df_clean[df_clean['joint_type'] == jt]
            x = sub[name].values
            y = sub['dSeq'].values
            # 过滤 NaN
            mask = np.isfinite(x) & np.isfinite(y)
            cols[f'{jt}_x'] = x[mask]
            cols[f'{jt}_y'] = y[mask]
            max_len = max(max_len, mask.sum())

        # 补齐等长 (Origin 导入更顺畅)
        for k in cols:
            arr = cols[k]
            if len(arr) < max_len:
                pad = np.full(max_len - len(arr), np.nan)
                cols[k] = np.concatenate([arr, pad])

        df_out = pd.DataFrame(cols)
        fp = os.path.join(out_dir, f'origin_data_panel_{panel_idx}_{name}.csv')
        df_out.to_csv(fp, index=False, float_format='%.6g')
        print(f'  ✓ {os.path.basename(fp)}  ({max_len} rows, {len(cols)} cols)')


# ============================================================ #
# 预览图 (最终以 Origin 为准)
# ============================================================ #
def plot_preview(df, out_path, dpi=200):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    dSeq = df['dSeq'].values

    for idx, (name, _, label, _) in enumerate(STATS_SPEC):
        ax = axes[idx]
        stat = df[name].values

        for jt, c in JT_COLORS.items():
            mask = (df['joint_type'] == jt).values & np.isfinite(stat) & np.isfinite(dSeq)
            if mask.sum() > 0:
                ax.scatter(stat[mask], dSeq[mask], s=36, color=c,
                           edgecolor='#444', lw=0.5, alpha=0.8, label=jt)

        rho, _, n = safe_spearman(dSeq, stat)
        r_ll, _, _ = safe_pearson_loglog(dSeq, stat)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(f'{label} [MPa]', fontsize=11)
        ax.set_ylabel(r'$\Delta S_{\mathrm{eq}}$ [MPa]', fontsize=11)
        ax.set_title(rf'$\rho_s$={rho:.3f},  $r_{{\log\log}}$={r_ll:.3f}  (n={n})',
                     fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(True, which='both', ls=':', lw=0.3, alpha=0.5)
        if idx == 0:
            ax.legend(fontsize=9, loc='upper left', framealpha=0.85)

    fig.suptitle(
        r'$\Delta S_{\mathrm{eq}}$ vs FEA stress-field top-5% statistics (preview)',
        fontsize=12,
    )
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    fig.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {out_path}')


# ============================================================ #
# 主入口
# ============================================================ #
def main():
    ap = argparse.ArgumentParser(description='ΔSeq 后验物理一致性分析 (top-5% 版)')
    ap.add_argument('--result_dir', default='./v11_final')
    ap.add_argument('--graph_dir',  default='./single_level_graphs')
    ap.add_argument('--out_dir',    default='./dSeq_posterior')
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    print('=' * 60)
    print('  ΔSeq 后验物理一致性分析 (top-5%)')
    print('=' * 60)

    # ---- 1. ΔSeq ----
    ens_csv = os.path.join(args.result_dir, 'specimen_ensemble.csv')
    if not os.path.exists(ens_csv):
        raise FileNotFoundError(f'找不到 {ens_csv}')
    print(f'\n[1] 读取 ΔSeq: {ens_csv}')

    # ---- 2. FEA top-5% 统计量 ----
    print(f'\n[2] 扫描 {args.graph_dir}/*.pkl')
    peak_map = extract_stats_from_pkl(args.graph_dir)

    # ---- 3. 主表 ----
    print(f'\n[3] 合并主表')
    df = build_master_table(ens_csv, peak_map)
    df_clean = df.dropna(subset=['dSeq', 'sigma1_top5pct']).copy()
    print(f'  有效行数: {len(df_clean)}')
    df_clean.to_csv(os.path.join(args.out_dir, 'master_table.csv'),
                    index=False, float_format='%.6g')
    print(f'  ✓ master_table.csv')

    # ---- 4. 相关系数 ----
    print(f'\n[4] 相关系数')
    df_overall = correlation_table(df_clean, group='overall')
    print(f'\n  [总体]  n = {len(df_clean)}')
    print(f'  {"statistic":<22} {"rho":>8} {"p-value":>10} {"log-log r":>10}')
    print('-' * 56)
    for _, r in df_overall.iterrows():
        print(f'  {r["stat"]:<22} {r["spearman_rho"]:+8.3f} '
              f'{r["spearman_p"]:10.2e} {r["pearson_loglog_r"]:+10.3f}')

    group_dfs = [df_overall]
    for jt in ['DJ', 'TX', 'UL']:
        sub = df_clean[df_clean['joint_type'] == jt]
        if len(sub) >= 5:
            tbl = correlation_table(sub, group=jt)
            group_dfs.append(tbl)
            print(f'\n  [{jt}]  n = {len(sub)}')
            for _, r in tbl.iterrows():
                print(f'    {r["stat"]:<22} {r["spearman_rho"]:+8.3f} '
                      f'(log-log r={r["pearson_loglog_r"]:+.3f})')

    pd.concat(group_dfs, ignore_index=True).to_csv(
        os.path.join(args.out_dir, 'correlation_table.csv'),
        index=False, float_format='%.4g')
    print(f'\n  ✓ correlation_table.csv')

    # ---- 5. Origin 原始数据 ----
    print(f'\n[5] 导出 Origin 原始数据')
    export_origin_data(df_clean, args.out_dir)

    # ---- 6. 预览图 ----
    print(f'\n[6] 预览图')
    plot_preview(df_clean, os.path.join(args.out_dir, 'scatter_3panel.png'))

    # ---- 结尾 ----
    print(f'\n{"=" * 60}')
    print(f'  输出目录: {args.out_dir}/')
    print(f'    master_table.csv                — 每试件 ΔSeq + 3 个统计量')
    print(f'    correlation_table.csv           — 相关系数 (总体 + 分组)')
    print(f'    origin_data_panel_1_*.csv       — σ₁ top-5%  散点原始数据')
    print(f'    origin_data_panel_2_*.csv       — σ_vm top-5% 散点原始数据')
    print(f'    origin_data_panel_3_*.csv       — τ_max top-5% 散点原始数据')
    print(f'    scatter_3panel.png/pdf          — 预览图 (Origin 出版以此为参考)')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
