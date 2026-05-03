# -*- coding: utf-8 -*-
"""
analyze_baseline_comparison.py — §4.3 基准方法全面对比分析

读取:
  ./v11_final/           — KG-GNN 结果
  ./baseline_4_3/        — A 组 (3000D 展平) 6 个 baseline
  ./baseline_4_3b/       — B 组 (24D 手工特征) 4 个 baseline

输出:
  tables/   — CSV 汇总表
  previews/ — 对比图 (PDF + PNG)

用法:
  python analyze_baseline_comparison.py [--v11_dir ./v11_final]
                                         [--a_dir ./baseline_4_3]
                                         [--b_dir ./baseline_4_3b]
                                         [--out_dir ./v11_analysis]
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from plot_style import (
    apply_style, savefig_multi,
    SINGLE_COL, ONE_HALF_COL, DOUBLE_COL,
    JT_COLORS, MACARON_LAVENDER, MACARON_APRICOT, MACARON_GREEN,
    MACARON_ROSE, MACARON_BROWN, IIW_REF_COLOR,
)
apply_style()

JT_ORDER = ['DJ', 'TX', 'UL']
JT_LABEL = {'DJ': 'BWJ', 'TX': 'TWJ', 'UL': 'DURJ'}
CT_ORDER = [0, 20, 40, 60]


def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)
def save_table(df, path):
    df.to_csv(path, index=False, encoding='utf-8-sig', float_format='%.4g')
    print(f'  ✓ {path.name}  ({len(df)} rows × {len(df.columns)} cols)')


# ============================================================ #
# 数据加载
# ============================================================ #
MODEL_ORDER = [
    ('KG-GNN',  'KG-GNN (proposed)', None, None),
    # A 组
    ('B-1',     'RF (3000D)',         'a', 'B-1'),
    ('B-2',     'XGBoost (3000D)',    'a', 'B-2'),
    ('B-3',     'SVR (PCA→20D)',      'a', 'B-3'),
    ('B-4',     'GPR (PCA→20D)',      'a', 'B-4'),
    ('B-5',     'MLP (3000D)',        'a', 'B-5'),
    ('B-6',     'PointNet (500×6)',   'a', 'B-6'),
    # B 组
    ('B-1b',    'RF (24D)',           'b', 'B-1b'),
    ('B-2b',    'XGBoost (24D)',      'b', 'B-2b'),
    ('B-3b',    'SVR (24D)',          'b', 'B-3b'),
    ('B-4b',    'GPR (24D)',          'b', 'B-4b'),
]


def load_all(v11_dir, a_dir, b_dir):
    """返回 {model_key: {'meta': dict, 'fold': df, 'spec': df, 'ens': df}}."""
    data = {}

    # KG-GNN (V11)
    v11 = Path(v11_dir)
    if (v11 / 'meta.json').exists():
        with open(v11 / 'meta.json', encoding='utf-8') as f:
            meta = json.load(f)
        fold_df = pd.read_csv(v11 / 'results.csv')
        spec_df = pd.read_csv(v11 / 'specimen_preds.csv')
        ens_df = pd.read_csv(v11 / 'specimen_ensemble.csv')
        # 只取 seed 11/22/33 与 baseline 对齐 (3 seeds)
        fold_3s = fold_df[fold_df['seed'].isin([11, 22, 33])].copy()
        spec_3s = spec_df[spec_df['seed'].isin([11, 22, 33])].copy()
        # ★ 从 3 seeds 的 specimen_preds 重新计算集成, 确保与 baseline 公平
        ens_3s = spec_3s.groupby('ID').agg(
            true_N=('true_N', 'first'), true_log10N=('true_log10N', 'first'),
            pred_log10N_mean=('pred_log10N', 'mean'), pred_log10N_std=('pred_log10N', 'std'),
            joint_type=('joint_type', 'first'), corrosion_hours=('corrosion_hours', 'first'),
            n_evals=('pred_log10N', 'count'),
        ).reset_index()
        ens_3s['pred_N_ensemble'] = np.power(10.0, ens_3s['pred_log10N_mean'])
        ens_3s['rel_error_ensemble'] = np.abs(
            ens_3s['pred_N_ensemble'] - ens_3s['true_N']) / (ens_3s['true_N'] + 1e-8)
        data['KG-GNN'] = {
            'meta': meta, 'fold': fold_df, 'fold_3s': fold_3s,
            'spec': spec_df, 'spec_3s': spec_3s,
            'ens': ens_3s, 'ens_5s': ens_df}
        ens_mre_3s = ens_3s['rel_error_ensemble'].mean()
        ens_mre_5s = ens_df['rel_error_ensemble'].mean()
        print(f'  ✓ KG-GNN: {len(fold_df)} folds (5 seeds), {len(fold_3s)} folds (3 seeds)')
        print(f'    ensemble MRE: 3-seed={ens_mre_3s*100:.1f}%, 5-seed={ens_mre_5s*100:.1f}%')

    # A 组
    a_path = Path(a_dir)
    for key, label, grp, subdir in MODEL_ORDER:
        if grp != 'a': continue
        d = a_path / subdir
        if not (d / 'results.csv').exists(): continue
        fold_df = pd.read_csv(d / 'results.csv')
        spec_df = pd.read_csv(d / 'specimen_preds.csv') if (d / 'specimen_preds.csv').exists() else None
        ens_df = pd.read_csv(d / 'specimen_ensemble.csv') if (d / 'specimen_ensemble.csv').exists() else None
        meta = json.load(open(d / 'meta.json', encoding='utf-8')) if (d / 'meta.json').exists() else {}
        data[key] = {'meta': meta, 'fold': fold_df, 'spec': spec_df, 'ens': ens_df}
        print(f'  ✓ {key}: {len(fold_df)} folds')

    # B 组
    b_path = Path(b_dir)
    for key, label, grp, subdir in MODEL_ORDER:
        if grp != 'b': continue
        d = b_path / subdir
        if not (d / 'results.csv').exists(): continue
        fold_df = pd.read_csv(d / 'results.csv')
        spec_df = pd.read_csv(d / 'specimen_preds.csv') if (d / 'specimen_preds.csv').exists() else None
        ens_df = pd.read_csv(d / 'specimen_ensemble.csv') if (d / 'specimen_ensemble.csv').exists() else None
        meta = json.load(open(d / 'meta.json', encoding='utf-8')) if (d / 'meta.json').exists() else {}
        data[key] = {'meta': meta, 'fold': fold_df, 'spec': spec_df, 'ens': ens_df}
        print(f'  ✓ {key}: {len(fold_df)} folds')

    return data


# ============================================================ #
# 表 1: 全模型汇总对比表
# ============================================================ #
def build_summary_table(data):
    rows = []
    for key, label, grp, subdir in MODEL_ORDER:
        if key not in data: continue
        d = data[key]
        fold = d.get('fold_3s', d['fold']) if key == 'KG-GNN' else d['fold']
        ens = d.get('ens')

        row = {
            'Model': key,
            'Label': label,
            'Group': 'Proposed' if key == 'KG-GNN' else ('A' if grp == 'a' else 'B'),
            'fold_MRE_mean': fold['MRE'].mean(),
            'fold_MRE_std': fold['MRE'].std(),
            'fold_R2_logN_mean': fold['R2_logN'].mean(),
            'fold_R2_N_mean': fold['R2_N'].mean(),
            'fold_P2X_mean': fold['P2X_cov'].mean(),
            'fold_RMSE_N_mean': fold['RMSE_N'].mean() if 'RMSE_N' in fold.columns else np.nan,
        }

        if ens is not None and 'rel_error_ensemble' in ens.columns:
            row['ens_MRE'] = ens['rel_error_ensemble'].mean()
            t_log = ens['true_log10N'].values
            p_log = ens['pred_log10N_mean'].values
            row['ens_R2_logN'] = 1 - np.sum((p_log - t_log)**2) / np.sum((t_log - t_log.mean())**2)
            t_N = ens['true_N'].values
            p_N = ens['pred_N_ensemble'].values
            ratio = p_N / (t_N + 1e-8)
            row['ens_P2X'] = np.mean((ratio >= 0.5) & (ratio <= 2.0))
        else:
            row['ens_MRE'] = row['fold_MRE_mean']
            row['ens_R2_logN'] = row['fold_R2_logN_mean']
            row['ens_P2X'] = row['fold_P2X_mean']

        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================ #
# 表 2: 分接头类型 MRE 对比
# ============================================================ #
def build_per_jt_table(data):
    rows = []
    for key, label, grp, subdir in MODEL_ORDER:
        if key not in data: continue
        d = data[key]
        ens = d.get('ens')
        if ens is None: continue
        jt_col = 'joint_type'
        if jt_col not in ens.columns: continue
        for jt in JT_ORDER:
            sub = ens[ens[jt_col] == jt]
            if len(sub) == 0: continue
            rows.append({
                'Model': key, 'Label': label,
                'joint_type': jt, 'joint_label': JT_LABEL[jt],
                'n_specimens': len(sub),
                'MRE_mean': sub['rel_error_ensemble'].mean(),
                'MRE_std': sub['rel_error_ensemble'].std(),
                'MRE_median': sub['rel_error_ensemble'].median(),
            })
    return pd.DataFrame(rows)


# ============================================================ #
# 表 3: 分腐蚀工况 MRE 对比
# ============================================================ #
def build_per_ct_table(data):
    rows = []
    for key, label, grp, subdir in MODEL_ORDER:
        if key not in data: continue
        d = data[key]
        ens = d.get('ens')
        if ens is None: continue
        ct_col = 'corrosion_hours'
        if ct_col not in ens.columns: continue
        for ct in CT_ORDER:
            sub = ens[ens[ct_col] == ct]
            if len(sub) == 0: continue
            rows.append({
                'Model': key, 'Label': label,
                'corrosion_hours': ct,
                'n_specimens': len(sub),
                'MRE_mean': sub['rel_error_ensemble'].mean(),
                'MRE_std': sub['rel_error_ensemble'].std(),
            })
    return pd.DataFrame(rows)


# ============================================================ #
# 图 1: 集成 MRE 柱状图 (A+B 全模型)
# ============================================================ #
def plot_mre_bar(df_sum, out_path):
    df = df_sum.sort_values('ens_MRE').reset_index(drop=True)
    n = len(df)

    fig, ax = plt.subplots(figsize=(ONE_HALF_COL, SINGLE_COL * 1.2), constrained_layout=True)

    colors = []
    for _, r in df.iterrows():
        if r['Group'] == 'Proposed':
            colors.append(MACARON_BROWN)
        elif r['Group'] == 'A':
            colors.append(MACARON_LAVENDER)
        else:
            colors.append(MACARON_APRICOT)

    y_pos = np.arange(n)
    bars = ax.barh(y_pos, df['ens_MRE'].values * 100, color=colors,
                   edgecolor='#555', lw=0.5, height=0.65)

    # 误差棒 (fold std)
    for i, (_, r) in enumerate(df.iterrows()):
        ax.errorbar(r['ens_MRE'] * 100, i, xerr=r['fold_MRE_std'] * 100,
                     fmt='none', color='#555', capsize=2, capthick=0.6, lw=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['Label'].values, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Ensemble MRE (%)')
    ax.set_title('Baseline comparison', fontsize=9)

    for i, bar in enumerate(bars):
        val = df.iloc[i]['ens_MRE'] * 100
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', fontsize=6.5)

    # 图例
    patches = [
        mpatches.Patch(color=MACARON_BROWN, label='Proposed (KG-GNN)'),
        mpatches.Patch(color=MACARON_LAVENDER, label='Group A (3000D raw)'),
        mpatches.Patch(color=MACARON_APRICOT, label='Group B (24D handcrafted)'),
    ]
    ax.legend(handles=patches, fontsize=6, loc='lower right')

    savefig_multi(fig, out_path)
    plt.close(fig)


# ============================================================ #
# 图 2: Fold-level MRE 箱线图
# ============================================================ #
def plot_mre_boxplot(data, df_sum, out_path):
    # 按 ens_MRE 排序, 排除崩溃模型
    df = df_sum[df_sum['ens_MRE'] < 0.5].sort_values('ens_MRE').reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(ONE_HALF_COL, SINGLE_COL * 1.0), constrained_layout=True)

    box_data = []
    labels = []
    colors_face = []
    for _, r in df.iterrows():
        key = r['Model']
        d = data[key]
        fold = d.get('fold_3s', d['fold']) if key == 'KG-GNN' else d['fold']
        box_data.append(fold['MRE'].values * 100)
        labels.append(r['Label'])
        if r['Group'] == 'Proposed':
            colors_face.append(MACARON_BROWN)
        elif r['Group'] == 'A':
            colors_face.append(MACARON_LAVENDER)
        else:
            colors_face.append(MACARON_APRICOT)

    bp = ax.boxplot(box_data, vert=False, patch_artist=True, widths=0.6,
                    medianprops=dict(color='#333', lw=1.2),
                    flierprops=dict(marker='o', markersize=3, alpha=0.5))
    for patch, c in zip(bp['boxes'], colors_face):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
        patch.set_edgecolor('#555')

    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Fold MRE (%)')
    ax.set_title('Fold-level MRE distribution', fontsize=9)

    savefig_multi(fig, out_path)
    plt.close(fig)


# ============================================================ #
# 图 3: 多指标雷达图 (R², P2X, 1-MRE)
# ============================================================ #
def plot_multi_metric_bar(df_sum, out_path):
    df = df_sum[df_sum['ens_MRE'] < 0.5].sort_values('ens_MRE').reset_index(drop=True)
    metrics = ['ens_MRE', 'ens_R2_logN', 'ens_P2X']
    metric_labels = ['MRE (%)', r'$R^2_{\log N}$', r'$P_{2\times}$']

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, SINGLE_COL * 0.9),
                             constrained_layout=True, sharey=True)

    n = len(df)
    y_pos = np.arange(n)

    for ax, met, mlab in zip(axes, metrics, metric_labels):
        vals = df[met].values.copy()
        stds = df['fold_MRE_std'].values * 100 if met == 'ens_MRE' else None
        if met == 'ens_MRE':
            vals = vals * 100  # 转百分比

        colors = []
        for _, r in df.iterrows():
            if r['Group'] == 'Proposed':
                colors.append(MACARON_BROWN)
            elif r['Group'] == 'A':
                colors.append(MACARON_LAVENDER)
            else:
                colors.append(MACARON_APRICOT)

        bars = ax.barh(y_pos, vals, color=colors, edgecolor='#555', lw=0.5, height=0.6)

        # MRE 面板加误差棒
        if stds is not None:
            for i in range(n):
                ax.errorbar(vals[i], i, xerr=stds[i],
                            fmt='none', color='#555', capsize=2, capthick=0.6, lw=0.6)

        ax.set_yticks(y_pos)
        if ax == axes[0]:
            ax.set_yticklabels(df['Label'].values, fontsize=6.5)
        ax.invert_yaxis()
        ax.set_xlabel(mlab, fontsize=8)

        for i, v in enumerate(vals):
            if met == 'ens_MRE':
                # MRE: 标注 mean±std
                fmt = f'{v:.1f}±{stds[i]:.1f}'
            else:
                fmt = f'{v:.3f}'
            offset = stds[i] + 0.5 if met == 'ens_MRE' else 0.003
            ax.text(v + offset, i, fmt, va='center', fontsize=5.5)

    axes[0].set_title('MRE ↓', fontsize=8)
    axes[1].set_title(r'$R^2$ ↑', fontsize=8)
    axes[2].set_title(r'$P_{2\times}$ ↑', fontsize=8)

    savefig_multi(fig, out_path)
    plt.close(fig)


# ============================================================ #
# 图 4: 分接头类型 MRE 对比 (grouped bar)
# ============================================================ #
def plot_per_jt_comparison(df_jt, out_path):
    # 只选主要模型
    show_models = ['KG-GNN', 'B-1', 'B-1b', 'B-2b', 'B-6']
    df = df_jt[df_jt['Model'].isin(show_models)].copy()
    if len(df) == 0: return

    models = [m for m in show_models if m in df['Model'].unique()]
    n_models = len(models)
    n_jt = len(JT_ORDER)

    fig, ax = plt.subplots(figsize=(ONE_HALF_COL, SINGLE_COL * 0.9), constrained_layout=True)

    model_colors = {
        'KG-GNN': MACARON_BROWN,
        'B-1': MACARON_LAVENDER,
        'B-1b': MACARON_APRICOT,
        'B-2b': MACARON_GREEN,
        'B-6': MACARON_ROSE,
    }

    width = 0.15
    x = np.arange(n_jt)

    for i, m in enumerate(models):
        vals, errs = [], []
        for jt in JT_ORDER:
            sub = df[(df['Model'] == m) & (df['joint_type'] == jt)]
            if len(sub) > 0:
                vals.append(sub['MRE_mean'].values[0] * 100)
                errs.append(sub['MRE_std'].values[0] * 100)
            else:
                vals.append(0); errs.append(0)

        label_map = {k: l for k, l, _, _ in MODEL_ORDER}
        ax.bar(x + i * width, vals, width * 0.9, label=label_map.get(m, m),
               color=model_colors.get(m, '#999'), edgecolor='#555', lw=0.4,
               yerr=errs, capsize=2, error_kw={'lw': 0.6})

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([JT_LABEL[jt] for jt in JT_ORDER], fontsize=9)
    ax.set_ylabel('MRE (%)')
    ax.set_title('MRE by joint type', fontsize=9)
    ax.legend(fontsize=6, loc='upper right')

    savefig_multi(fig, out_path)
    plt.close(fig)


# ============================================================ #
# 图 5: KG-GNN vs 最佳 baseline 散点对比
# ============================================================ #
def plot_scatter_comparison(data, best_key, out_path):
    gnn_ens = data['KG-GNN']['ens']
    bl_ens = data[best_key]['ens']
    if gnn_ens is None or bl_ens is None: return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL * 0.85, SINGLE_COL * 1.1),
                                   constrained_layout=True)

    for ax, ens, title in [(ax1, gnn_ens, 'KG-GNN'),
                            (ax2, bl_ens, {k: l for k, l, _, _ in MODEL_ORDER}.get(best_key, best_key))]:
        t_log = ens['true_log10N'].values
        p_log = ens['pred_log10N_mean'].values
        r2 = 1 - np.sum((p_log - t_log)**2) / np.sum((t_log - t_log.mean())**2)
        t_N = ens['true_N'].values
        p_N = ens['pred_N_ensemble'].values
        mre = np.mean(np.abs(p_N - t_N) / (t_N + 1e-8))

        jt_col = 'joint_type'
        for jt in JT_ORDER:
            mask = ens[jt_col] == jt
            ax.scatter(t_log[mask], p_log[mask], s=25, alpha=0.8,
                       color=JT_COLORS[jt], edgecolor='white', lw=0.3,
                       label=JT_LABEL[jt], zorder=3)

        lims = [min(t_log.min(), p_log.min()) - 0.1,
                max(t_log.max(), p_log.max()) + 0.1]
        ax.plot(lims, lims, 'k-', lw=0.8, zorder=1)
        ax.plot(lims, [l + np.log10(2) for l in lims], '--', color='#aaa', lw=0.5)
        ax.plot(lims, [l - np.log10(2) for l in lims], '--', color='#aaa', lw=0.5)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel(r'True $\log_{10}(N)$', fontsize=8)
        ax.set_ylabel(r'Predicted $\log_{10}(N)$', fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=6, loc='upper left')

        ax.text(0.95, 0.05,
                f'$R^2$ = {r2:.4f}\nMRE = {mre*100:.2f}%',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=7, bbox=dict(facecolor='white', alpha=0.8, edgecolor='#ccc'))

    savefig_multi(fig, out_path)
    plt.close(fig)


# ============================================================ #
# 图 6: 逐试件相对误差对比 (KG-GNN vs best baseline)
# ============================================================ #
def plot_per_specimen_error(data, best_key, out_path):
    gnn_ens = data['KG-GNN']['ens']
    bl_ens = data[best_key]['ens']
    if gnn_ens is None or bl_ens is None: return

    merged = gnn_ens[['ID', 'joint_type', 'true_N', 'rel_error_ensemble']].merge(
        bl_ens[['ID', 'rel_error_ensemble']], on='ID', suffixes=('_gnn', '_bl'))
    merged = merged.sort_values('true_N').reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, SINGLE_COL * 0.7), constrained_layout=True)

    x = np.arange(len(merged))
    bl_label = {k: l for k, l, _, _ in MODEL_ORDER}.get(best_key, best_key)

    ax.bar(x - 0.2, merged['rel_error_ensemble_gnn'].values * 100, width=0.4,
           color=MACARON_BROWN, edgecolor='#555', lw=0.3, alpha=0.8, label='KG-GNN')
    ax.bar(x + 0.2, merged['rel_error_ensemble_bl'].values * 100, width=0.4,
           color=MACARON_LAVENDER, edgecolor='#555', lw=0.3, alpha=0.8, label=bl_label)

    ax.set_xlabel('Specimen (sorted by true $N$)', fontsize=8)
    ax.set_ylabel('Relative error (%)', fontsize=8)
    ax.set_title('Per-specimen error comparison', fontsize=9)
    ax.legend(fontsize=7)
    ax.set_xlim(-1, len(merged))

    # 标注 joint type 区间
    for jt in JT_ORDER:
        mask = merged['joint_type'] == jt
        if mask.sum() > 0:
            idx = np.where(mask)[0]
            mid = idx[len(idx) // 2]
            ax.text(mid, ax.get_ylim()[1] * 0.92, JT_LABEL[jt],
                    ha='center', fontsize=6.5, color=JT_COLORS[jt], fontweight='bold')

    savefig_multi(fig, out_path)
    plt.close(fig)


# ============================================================ #
# 图 7: A 组 vs B 组 配对对比 (RF, XGB, SVR, GPR)
# ============================================================ #
def plot_ab_comparison(df_sum, out_path):
    pairs = [('B-1', 'B-1b', 'RF'), ('B-2', 'B-2b', 'XGBoost'),
             ('B-3', 'B-3b', 'SVR'), ('B-4', 'B-4b', 'GPR')]

    fig, ax = plt.subplots(figsize=(ONE_HALF_COL, SINGLE_COL * 0.8), constrained_layout=True)

    x = np.arange(len(pairs))
    width = 0.3

    vals_a, vals_b, stds_a, stds_b = [], [], [], []
    labels = []
    for ka, kb, name in pairs:
        ra = df_sum[df_sum['Model'] == ka]
        rb = df_sum[df_sum['Model'] == kb]
        vals_a.append(ra['ens_MRE'].values[0] * 100 if len(ra) > 0 else 0)
        vals_b.append(rb['ens_MRE'].values[0] * 100 if len(rb) > 0 else 0)
        stds_a.append(ra['fold_MRE_std'].values[0] * 100 if len(ra) > 0 else 0)
        stds_b.append(rb['fold_MRE_std'].values[0] * 100 if len(rb) > 0 else 0)
        labels.append(name)

    ax.bar(x - width / 2, vals_a, width, color=MACARON_LAVENDER, edgecolor='#555',
           lw=0.5, label='Group A (3000D)', yerr=stds_a, capsize=3, error_kw={'lw': 0.6})
    ax.bar(x + width / 2, vals_b, width, color=MACARON_APRICOT, edgecolor='#555',
           lw=0.5, label='Group B (24D)', yerr=stds_b, capsize=3, error_kw={'lw': 0.6})

    # KG-GNN 参考线
    gnn_row = df_sum[df_sum['Model'] == 'KG-GNN']
    if len(gnn_row) > 0:
        gnn_mre = gnn_row['ens_MRE'].values[0] * 100
        ax.axhline(gnn_mre, color=MACARON_BROWN, ls='--', lw=1.0,
                   label=f'KG-GNN ({gnn_mre:.1f}%)')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Ensemble MRE (%)')
    ax.set_title('Group A vs Group B', fontsize=9)
    ax.legend(fontsize=6.5)

    savefig_multi(fig, out_path)
    plt.close(fig)


# ============================================================ #
# 主入口
# ============================================================ #
def main():
    ap = argparse.ArgumentParser(description='§4.3 基准方法全面对比分析')
    ap.add_argument('--v11_dir', default='./v11_final')
    ap.add_argument('--a_dir', default='./baseline_4_3')
    ap.add_argument('--b_dir', default='./baseline_4_3b')
    ap.add_argument('--out_dir', default='./v11_analysis')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_tbl = out_dir / 'tables'
    out_prev = out_dir / 'previews'
    ensure_dir(out_tbl); ensure_dir(out_prev)

    print('=' * 60)
    print('  §4.3 基准方法全面对比分析')
    print('=' * 60)

    # 加载
    print('\n[1] 加载数据...')
    data = load_all(args.v11_dir, args.a_dir, args.b_dir)
    if 'KG-GNN' not in data:
        print('⚠️ 未找到 KG-GNN 结果, 退出'); return

    # 表 1: 汇总
    print('\n[2] 汇总表...')
    df_sum = build_summary_table(data)
    save_table(df_sum, out_tbl / 'baseline_comparison_summary.csv')

    # 控制台打印
    print(f'\n  {"Model":<25} {"MRE±std":>14} {"R²":>8} {"P2X":>6} {"ens_MRE":>8}')
    print('-' * 68)
    for _, r in df_sum.sort_values('ens_MRE').iterrows():
        print(f'  {r["Label"]:<25} '
              f'{r["fold_MRE_mean"]*100:.1f}±{r["fold_MRE_std"]*100:.1f}% '
              f'{r["ens_R2_logN"]:8.4f} {r["ens_P2X"]:6.3f} '
              f'{r["ens_MRE"]*100:7.1f}%')

    # 表 2: 分接头类型
    print('\n[3] 分接头类型...')
    df_jt = build_per_jt_table(data)
    if len(df_jt) > 0:
        save_table(df_jt, out_tbl / 'baseline_per_joint_type.csv')

    # 表 3: 分腐蚀工况
    print('\n[4] 分腐蚀工况...')
    df_ct = build_per_ct_table(data)
    if len(df_ct) > 0:
        save_table(df_ct, out_tbl / 'baseline_per_corrosion.csv')

    # 找最佳 baseline
    bl_sum = df_sum[df_sum['Model'] != 'KG-GNN'].copy()
    if len(bl_sum) > 0:
        best_key = bl_sum.loc[bl_sum['ens_MRE'].idxmin(), 'Model']
        best_label = bl_sum.loc[bl_sum['ens_MRE'].idxmin(), 'Label']
        print(f'\n  🏆 最佳 baseline: {best_key} ({best_label}), '
              f'ens MRE = {bl_sum["ens_MRE"].min()*100:.1f}%')
    else:
        best_key = None

    # 图 1: MRE 柱状图
    print('\n[5] 图: MRE 柱状图...')
    plot_mre_bar(df_sum, out_prev / 'fig_4_3a_mre_bar')

    # 图 2: 箱线图
    print('[6] 图: MRE 箱线图...')
    plot_mre_boxplot(data, df_sum, out_prev / 'fig_4_3b_mre_boxplot')

    # 图 3: 多指标
    print('[7] 图: 多指标对比...')
    plot_multi_metric_bar(df_sum, out_prev / 'fig_4_3c_multi_metric')

    # 图 4: 分接头类型
    if len(df_jt) > 0:
        print('[8] 图: 分接头类型 MRE...')
        plot_per_jt_comparison(df_jt, out_prev / 'fig_4_3d_per_jt')

    # 图 5: 散点对比
    if best_key and best_key in data:
        print('[9] 图: 散点对比 (KG-GNN vs best baseline)...')
        plot_scatter_comparison(data, best_key, out_prev / 'fig_4_3e_scatter_compare')

    # 图 6: 逐试件误差
    if best_key and best_key in data:
        print('[10] 图: 逐试件误差对比...')
        plot_per_specimen_error(data, best_key, out_prev / 'fig_4_3f_per_specimen')

    # 图 7: A vs B 配对对比
    print('[11] 图: A 组 vs B 组...')
    plot_ab_comparison(df_sum, out_prev / 'fig_4_3g_ab_compare')

    print(f'\n{"="*60}')
    print(f'  完成! 输出: {out_dir}/')
    print(f'    tables/:   baseline_comparison_summary.csv')
    print(f'               baseline_per_joint_type.csv')
    print(f'               baseline_per_corrosion.csv')
    print(f'    previews/: fig_4_3a ~ fig_4_3g (7 张图)')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()