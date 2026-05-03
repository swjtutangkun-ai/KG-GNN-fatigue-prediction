# -*- coding: utf-8 -*-
"""
analyze_ablation.py — §5.1 消融实验分析

读取:
  ./v11_final/      — P-A1 基线 (seed 11/22/33 的 30 折)
  ./ablation_5_1/   — P-A2, P-B, P-C, G-1~G-4, I-1~I-4

输出:
  tables/   — 消融汇总表
  previews/ — 消融对比图

用法:
  python analyze_ablation.py [--v11_dir ./v11_final]
                              [--abl_dir ./ablation_5_1]
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
    MACARON_LAVENDER, MACARON_APRICOT, MACARON_GREEN,
    MACARON_ROSE, MACARON_BROWN,
)
apply_style()


def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)
def save_table(df, path):
    df.to_csv(path, index=False, encoding='utf-8-sig', float_format='%.4g')
    print(f'  ✓ {path.name}  ({len(df)} rows × {len(df.columns)} cols)')


# ============================================================ #
# 变体定义
# ============================================================ #
# (key, label, group, description)
VARIANTS = [
    ('P-A1', 'P-A1 (baseline)',       'P', 'Hard-coded Basquin + IIW prior'),
    ('P-A2', 'P-A2 (random init)',    'P', 'Random init A/m, no IIW prior'),
    ('P-B',  'P-B (soft constraint)', 'P', 'MLP + λ·Basquin penalty'),
    ('P-C',  'P-C (no physics)',      'P', 'Pure MLP, no Basquin'),
    ('G-1',  'G-1 (no GNN)',          'G', 'Hotspot descriptors only'),
    ('G-2',  'G-2 (GCN)',             'G', 'GCN backbone, no attention'),
    ('G-3',  'G-3 (+edge feat)',      'G', 'Add edge features (reverse)'),
    ('G-4',  'G-4 (mean pool)',       'G', 'Mean pooling, no attn aggr'),
    ('I-1',  'I-1 (no hotspot)',      'I', 'Remove hotspot descriptors'),
    ('I-2',  'I-2 (no augment)',      'I', 'No data augmentation'),
    ('I-3',  'I-3 (uniform aug)',     'I', 'Uniform 3× augmentation'),
    ('I-4',  'I-4 (σ₁ only)',        'I', 'Node features: σ₁ only'),
]

GROUP_COLORS = {
    'P': MACARON_ROSE,
    'G': MACARON_LAVENDER,
    'I': MACARON_GREEN,
}
GROUP_LABELS = {
    'P': '§5.1.1 Physics injection',
    'G': '§5.1.2 Graph structure',
    'I': '§5.1.3 Input & augmentation',
}
BASELINE_COLOR = MACARON_BROWN


# ============================================================ #
# 数据加载
# ============================================================ #
def load_all(v11_dir, abl_dir):
    data = {}
    v11 = Path(v11_dir)

    # P-A1 基线: 从 v11_final 取 seed 11/22/33
    if (v11 / 'results.csv').exists():
        fold_df = pd.read_csv(v11 / 'results.csv')
        spec_df = pd.read_csv(v11 / 'specimen_preds.csv')
        fold_3s = fold_df[fold_df['seed'].isin([11, 22, 33])].copy()
        spec_3s = spec_df[spec_df['seed'].isin([11, 22, 33])].copy()
        # 3-seed 集成
        ens = spec_3s.groupby('ID').agg(
            true_N=('true_N', 'first'), true_log10N=('true_log10N', 'first'),
            pred_log10N_mean=('pred_log10N', 'mean'),
            joint_type=('joint_type', 'first'),
            corrosion_hours=('corrosion_hours', 'first'),
        ).reset_index()
        ens['pred_N_ensemble'] = np.power(10.0, ens['pred_log10N_mean'])
        ens['rel_error_ensemble'] = np.abs(
            ens['pred_N_ensemble'] - ens['true_N']) / (ens['true_N'] + 1e-8)
        data['P-A1'] = {'fold': fold_3s, 'ens': ens}
        print(f'  ✓ P-A1 (baseline): {len(fold_3s)} folds (3 seeds)')

    # 消融变体
    abl = Path(abl_dir)
    for key, label, grp, desc in VARIANTS:
        if key == 'P-A1': continue
        d = abl / key
        if not (d / 'results.csv').exists():
            continue
        fold_df = pd.read_csv(d / 'results.csv')
        ens_df = pd.read_csv(d / 'specimen_ensemble.csv') if (d / 'specimen_ensemble.csv').exists() else None
        meta = {}
        if (d / 'meta.json').exists():
            with open(d / 'meta.json', encoding='utf-8') as f:
                meta = json.load(f)
        data[key] = {'fold': fold_df, 'ens': ens_df, 'meta': meta}
        print(f'  ✓ {key}: {len(fold_df)} folds')

    return data


# ============================================================ #
# 汇总表
# ============================================================ #
def build_summary(data):
    rows = []
    for key, label, grp, desc in VARIANTS:
        if key not in data: continue
        d = data[key]
        fold = d['fold']
        ens = d.get('ens')

        row = {
            'Variant': key, 'Label': label, 'Group': grp, 'Description': desc,
            'fold_MRE_mean': fold['MRE'].mean(),
            'fold_MRE_std': fold['MRE'].std(),
            'fold_R2_logN': fold['R2_logN'].mean(),
            'fold_P2X': fold['P2X_cov'].mean(),
        }

        # A / m
        if 'A' in fold.columns and fold['A'].notna().any():
            row['A_mean'] = fold['A'].mean()
            row['A_std'] = fold['A'].std()
        else:
            row['A_mean'] = np.nan; row['A_std'] = np.nan
        if 'm' in fold.columns and fold['m'].notna().any():
            row['m_mean'] = fold['m'].mean()
            row['m_std'] = fold['m'].std()
        else:
            row['m_mean'] = np.nan; row['m_std'] = np.nan

        # n_params
        if 'n_params' in fold.columns:
            row['n_params'] = int(fold['n_params'].iloc[0])
        else:
            row['n_params'] = np.nan

        # 集成
        if ens is not None and 'rel_error_ensemble' in ens.columns:
            row['ens_MRE'] = ens['rel_error_ensemble'].mean()
            t_log = ens['true_log10N'].values
            p_log = ens['pred_log10N_mean'].values
            row['ens_R2_logN'] = 1 - np.sum((p_log - t_log)**2) / np.sum((t_log - t_log.mean())**2)
        else:
            row['ens_MRE'] = row['fold_MRE_mean']
            row['ens_R2_logN'] = row['fold_R2_logN']

        # 相对基线变化
        if 'P-A1' in data:
            bl_mre = data['P-A1']['fold']['MRE'].mean()
            row['delta_MRE_pct'] = (row['fold_MRE_mean'] - bl_mre) / bl_mre * 100
        else:
            row['delta_MRE_pct'] = np.nan

        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================ #
# 图: 消融 MRE 排名柱状图 (P/G/I 分色)
# ============================================================ #
def plot_ablation_bar(df_sum, out_path):
    # 按组排序: P → G → I, 组内按 ens_MRE 排序
    group_order = ['P', 'G', 'I']
    sorted_rows = []
    for g in group_order:
        sub = df_sum[df_sum['Group'] == g].sort_values('ens_MRE')
        sorted_rows.append(sub)
    df = pd.concat(sorted_rows).reset_index(drop=True)

    n = len(df)
    fig, ax = plt.subplots(figsize=(ONE_HALF_COL, max(SINGLE_COL * 1.3, n * 0.28)),
                           constrained_layout=True)

    y_pos = np.arange(n)
    colors = []
    for _, r in df.iterrows():
        if r['Variant'] == 'P-A1':
            colors.append(BASELINE_COLOR)
        else:
            colors.append(GROUP_COLORS[r['Group']])

    bars = ax.barh(y_pos, df['ens_MRE'].values * 100, color=colors,
                   edgecolor='#555', lw=0.5, height=0.65)

    # 误差棒
    for i, (_, r) in enumerate(df.iterrows()):
        ax.errorbar(r['ens_MRE'] * 100, i, xerr=r['fold_MRE_std'] * 100,
                    fmt='none', color='#555', capsize=2, capthick=0.6, lw=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['Label'].values, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Ensemble MRE (%)')
    ax.set_title('Ablation study', fontsize=9)

    # 标注数值 + delta
    for i, bar in enumerate(bars):
        val = df.iloc[i]['ens_MRE'] * 100
        delta = df.iloc[i]['delta_MRE_pct']
        if df.iloc[i]['Variant'] == 'P-A1':
            txt = f'{val:.1f}% (baseline)'
        elif np.isfinite(delta):
            sign = '+' if delta > 0 else ''
            txt = f'{val:.1f}% ({sign}{delta:.0f}%)'
        else:
            txt = f'{val:.1f}%'
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                txt, va='center', fontsize=5.5)

    # 组分隔线
    cum = 0
    for g in group_order:
        cnt = len(df[df['Group'] == g])
        if cum > 0:
            ax.axhline(cum - 0.5, color='#bbb', lw=0.6, ls='--')
        cum += cnt

    # 图例
    patches = [
        mpatches.Patch(color=BASELINE_COLOR, label='P-A1 (baseline)'),
        mpatches.Patch(color=GROUP_COLORS['P'], label='§5.1.1 Physics injection'),
        mpatches.Patch(color=GROUP_COLORS['G'], label='§5.1.2 Graph structure'),
        mpatches.Patch(color=GROUP_COLORS['I'], label='§5.1.3 Input & augmentation'),
    ]
    ax.legend(handles=patches, fontsize=5.5, loc='lower right')

    # x 轴范围
    max_val = df['ens_MRE'].max() * 100
    ax.set_xlim(0, min(max_val * 1.4, 70))

    savefig_multi(fig, out_path)
    plt.close(fig)


# ============================================================ #
# 图: Basquin 参数对比 (A vs m 散点)
# ============================================================ #
def plot_basquin_scatter(df_sum, out_path):
    df = df_sum.dropna(subset=['A_mean', 'm_mean']).copy()
    if len(df) < 2: return

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.9), constrained_layout=True)

    for _, r in df.iterrows():
        if r['Variant'] == 'P-A1':
            c, z, s = BASELINE_COLOR, 10, 80
        else:
            c, z, s = GROUP_COLORS[r['Group']], 5, 50
        ax.scatter(r['A_mean'], r['m_mean'], s=s, color=c, edgecolor='#555',
                   lw=0.5, zorder=z)
        ax.annotate(r['Variant'], (r['A_mean'], r['m_mean']),
                    fontsize=5.5, xytext=(4, 4), textcoords='offset points')
        # A 误差棒
        if np.isfinite(r['A_std']):
            ax.errorbar(r['A_mean'], r['m_mean'], xerr=r['A_std'], yerr=r['m_std'],
                        fmt='none', color=c, alpha=0.4, capsize=2, lw=0.6)

    # IIW 参考
    ax.axhline(3.0, color='#aaa', ls=':', lw=0.6, label='IIW m = 3.0')
    ax.axvline(12.01, color='#aaa', ls=':', lw=0.6, label='IIW A = 12.01')

    ax.set_xlabel(r'Basquin intercept $A_g$', fontsize=8)
    ax.set_ylabel(r'Basquin slope $m_g$', fontsize=8)
    ax.set_title('Learned Basquin parameters', fontsize=9)
    ax.legend(fontsize=6)

    savefig_multi(fig, out_path)
    plt.close(fig)


# ============================================================ #
# 主入口
# ============================================================ #
def main():
    ap = argparse.ArgumentParser(description='§5.1 消融实验分析')
    ap.add_argument('--v11_dir', default='./v11_final')
    ap.add_argument('--abl_dir', default='./ablation_5_1')
    ap.add_argument('--out_dir', default='./v11_analysis')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_tbl = out_dir / 'tables'
    out_prev = out_dir / 'previews'
    ensure_dir(out_tbl); ensure_dir(out_prev)

    print('=' * 60)
    print('  §5.1 消融实验分析')
    print('=' * 60)

    print('\n[1] 加载数据...')
    data = load_all(args.v11_dir, args.abl_dir)

    print('\n[2] 汇总表...')
    df_sum = build_summary(data)
    save_table(df_sum, out_tbl / 'ablation_summary.csv')

    # 控制台打印
    print(f'\n  {"Var":<8} {"MRE±std":>14} {"R²":>7} {"ens_MRE":>8} '
          f'{"A":>7} {"m":>7} {"Δ%":>6}')
    print('-' * 65)
    for _, r in df_sum.iterrows():
        A_str = f'{r["A_mean"]:.2f}' if np.isfinite(r['A_mean']) else '  N/A'
        m_str = f'{r["m_mean"]:.2f}' if np.isfinite(r['m_mean']) else ' N/A'
        d_str = f'{r["delta_MRE_pct"]:+.0f}%' if np.isfinite(r['delta_MRE_pct']) else '  —'
        print(f'  {r["Variant"]:<8} '
              f'{r["fold_MRE_mean"]*100:.1f}±{r["fold_MRE_std"]*100:.1f}% '
              f'{r["fold_R2_logN"]:.4f} '
              f'{r["ens_MRE"]*100:7.1f}% '
              f'{A_str:>7} {m_str:>7} {d_str:>6}')

    # 图 1: MRE 排名
    print('\n[3] 图: 消融 MRE 排名...')
    plot_ablation_bar(df_sum, out_prev / 'fig_5_1_ablation_bar')

    # 图 2: Basquin 参数散点
    print('[4] 图: Basquin 参数散点...')
    plot_basquin_scatter(df_sum, out_prev / 'fig_5_1_basquin_scatter')

    print(f'\n{"="*60}')
    print(f'  完成! 输出: {out_dir}/')
    print(f'    tables/:   ablation_summary.csv')
    print(f'    previews/: fig_5_1_ablation_bar, fig_5_1_basquin_scatter')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
