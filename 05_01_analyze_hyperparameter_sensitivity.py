# -*- coding: utf-8 -*-
"""
analyze_sensitivity.py v3 — §5.3 超参数敏感性 (纯 CSV, 无需重跑)

三种方法, 全部从 tpe_results.csv 的 55 行数据计算:

  (1) Random Forest 特征重要性 (sklearn)
  (2) Kruskal-Wallis 非参数方差分析 + η² 效应量
  (3) Partial Dependence + Scatter/Boxplot

用法:
  python analyze_sensitivity.py --tpe_dir ./tpe_search_v11 --out_dir ./v11_analysis
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import kruskal
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

from plot_style import (
    apply_style, savefig_multi,
    SINGLE_COL, ONE_HALF_COL, DOUBLE_COL,
    MACARON_LAVENDER, MACARON_APRICOT, MACARON_GREEN,
    MACARON_ROSE, MACARON_BROWN, IIW_REF_COLOR,
)
apply_style()

HP_ORDER = ['HIDDEN_DIM', 'HEADS', 'DROPOUT', 'LR', 'WEIGHT_DECAY']
HP_CONFIG = {
    'HIDDEN_DIM':   {'short': '$d_h$',   'type': 'int'},
    'HEADS':        {'short': '$H$',     'type': 'int'},
    'DROPOUT':      {'short': 'Dropout', 'type': 'float'},
    'LR':           {'short': 'LR',      'type': 'sci'},
    'WEIGHT_DECAY': {'short': 'WD',      'type': 'sci'},
}

# 柔和色板, 与消融实验图风格统一
HP_COLORS = {
    'HIDDEN_DIM':   '#8BABC4',   # 柔蓝
    'HEADS':        '#E8B89C',   # 柔橙
    'DROPOUT':      '#A3C9A0',   # 柔绿
    'LR':           '#D4A0A0',   # 柔粉
    'WEIGHT_DECAY': '#C4B09C',   # 柔棕
}

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)
def save_table(df, path):
    df.to_csv(path, index=False, encoding='utf-8-sig', float_format='%.6g')
    print(f'  ✓ {path.name}  ({len(df)} rows × {len(df.columns)} cols)')
def fmt_val(v, tp):
    if tp == 'sci': return f'{v:.0e}'
    elif tp == 'float': return f'{v:.1f}'
    else: return str(int(v))


# ================================================================ #
# 方法 1: Random Forest Permutation Importance
# ================================================================ #
def run_rf_importance(df, out_tbl):
    print('\n[方法 1] Random Forest Permutation Importance')
    X = df[HP_ORDER].values
    y = df['MRE_mean'].values

    rf = RandomForestRegressor(n_estimators=500, max_depth=5,
                               random_state=42, n_jobs=-1)
    rf.fit(X, y)
    r2_train = rf.score(X, y)
    print(f'  RF R² (in-sample) = {r2_train:.3f}')

    perm = permutation_importance(rf, X, y, n_repeats=50,
                                  random_state=42, n_jobs=-1)
    rows = []
    for i, hp in enumerate(HP_ORDER):
        rows.append({
            'hp_name': hp,
            'perm_importance_mean': perm.importances_mean[i],
            'perm_importance_std':  perm.importances_std[i],
            'mdi_importance': rf.feature_importances_[i],
        })
    df_imp = pd.DataFrame(rows).sort_values('perm_importance_mean',
                                             ascending=False).reset_index(drop=True)
    total = df_imp['perm_importance_mean'].sum()
    df_imp['perm_importance_pct'] = df_imp['perm_importance_mean'] / total * 100 if total > 0 else 0
    df_imp.insert(0, 'rank', range(1, len(df_imp) + 1))
    save_table(df_imp, out_tbl / 'table_5.3_rf_importance.csv')
    return df_imp


# ================================================================ #
# 方法 2: Kruskal-Wallis 检验 + η² 效应量
# ================================================================ #
def run_kruskal_wallis(df, out_tbl):
    print('\n[方法 2] Kruskal-Wallis 非参数方差分析')
    rows = []
    n_total = len(df)
    for hp in HP_ORDER:
        groups = [grp['MRE_mean'].values for _, grp in df.groupby(hp)]
        k = len(groups)
        if k < 2:
            rows.append({'hp_name': hp, 'H_statistic': 0, 'p_value': 1,
                         'eta_squared': 0, 'effect_size': 'N/A', 'n_levels': k})
            continue
        H, p = kruskal(*groups)
        eta_sq = max(0, (H - k + 1) / (n_total - k)) if n_total > k else 0
        if eta_sq >= 0.14:   effect = 'Large'
        elif eta_sq >= 0.06: effect = 'Medium'
        else:                effect = 'Small'
        rows.append({
            'hp_name': hp, 'n_levels': k, 'H_statistic': H,
            'p_value': p, 'significant': p < 0.05,
            'eta_squared': eta_sq, 'effect_size': effect,
        })
    df_kw = pd.DataFrame(rows).sort_values('eta_squared', ascending=False).reset_index(drop=True)
    df_kw.insert(0, 'rank', range(1, len(df_kw) + 1))
    save_table(df_kw, out_tbl / 'table_5.3_kruskal_wallis.csv')
    return df_kw


# ================================================================ #
# 方法 3: Partial Dependence + Scatter/Boxplot
# ================================================================ #
def run_partial_dependence(df, best_params, out_tbl):
    print('\n[方法 3] Partial Dependence + 经验分布')
    rows_pd = []
    for hp in HP_ORDER:
        cfg = HP_CONFIG[hp]
        for val, grp in df.groupby(hp):
            rows_pd.append({
                'hp_name': hp, 'hp_value': val,
                'hp_value_str': fmt_val(val, cfg['type']),
                'n_trials': len(grp),
                'MRE_pd_mean':   grp['MRE_mean'].mean(),
                'MRE_pd_median': grp['MRE_mean'].median(),
                'MRE_pd_std':    grp['MRE_mean'].std(),
                'MRE_pd_min':    grp['MRE_mean'].min(),
                'MRE_pd_max':    grp['MRE_mean'].max(),
                'MRE_pd_q25':    grp['MRE_mean'].quantile(0.25),
                'MRE_pd_q75':    grp['MRE_mean'].quantile(0.75),
            })
    df_pd = pd.DataFrame(rows_pd)
    save_table(df_pd, out_tbl / 'fig_5.3_partial_dependence.csv')

    for hp in HP_ORDER:
        df_sc = df[['trial', hp, 'MRE_mean']].copy()
        df_sc.rename(columns={hp: 'hp_value'}, inplace=True)
        df_sc['hp_name'] = hp
        if 'MRE_std' in df.columns:
            df_sc['MRE_std'] = df['MRE_std']
        save_table(df_sc, out_tbl / f'fig_5.3_scatter_{hp}.csv')

    return df_pd


# ================================================================ #
# 2D 交互热力图
# ================================================================ #
def run_heatmaps(df, best_params, out_tbl, out_prev):
    print('\n[交互热力图]')
    pairs = [('HIDDEN_DIM', 'HEADS'), ('LR', 'DROPOUT'), ('LR', 'WEIGHT_DECAY')]
    for hp_x, hp_y in pairs:
        suffix = f'{hp_x.lower()}_{hp_y.lower()}'
        vals_x = sorted(df[hp_x].unique())
        vals_y = sorted(df[hp_y].unique())
        hm = np.full((len(vals_y), len(vals_x)), np.nan)
        hm_n = np.zeros_like(hm, dtype=int)
        rows_2d = []
        for i, vy in enumerate(vals_y):
            for j, vx in enumerate(vals_x):
                sub = df[(df[hp_x] == vx) & (df[hp_y] == vy)]
                med = sub['MRE_mean'].median() if len(sub) > 0 else np.nan
                hm[i, j] = med
                hm_n[i, j] = len(sub)
                rows_2d.append({hp_x: vx, hp_y: vy,
                                'MRE_median': med, 'n_trials': len(sub)})
        save_table(pd.DataFrame(rows_2d), out_tbl / f'fig_5.3_heatmap_{suffix}.csv')

        fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.85),
                               constrained_layout=True)
        vmin = np.nanpercentile(hm, 10)
        vmax = np.nanpercentile(hm, 90)
        im = ax.imshow(hm, cmap='RdYlGn_r', aspect='auto', vmin=vmin, vmax=vmax)
        cx, cy = HP_CONFIG[hp_x], HP_CONFIG[hp_y]
        ax.set_xticks(range(len(vals_x)))
        ax.set_xticklabels([fmt_val(v, cx['type']) for v in vals_x], fontsize=7)
        ax.set_yticks(range(len(vals_y)))
        ax.set_yticklabels([fmt_val(v, cy['type']) for v in vals_y], fontsize=7)
        ax.set_xlabel(cx['short'], fontsize=8)
        ax.set_ylabel(cy['short'], fontsize=8)
        for i in range(len(vals_y)):
            for j in range(len(vals_x)):
                v = hm[i, j]; n = hm_n[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f'{v:.3f}\n(n={n})', ha='center', va='center',
                            fontsize=6, color='white' if v > (vmin+vmax)/2 else '#333')
        if best_params.get(hp_x) in vals_x and best_params.get(hp_y) in vals_y:
            bx = vals_x.index(best_params[hp_x])
            by = vals_y.index(best_params[hp_y])
            ax.plot(bx, by, 's', ms=18, mec=MACARON_BROWN, mew=2, mfc='none')
        cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label('Median MRE', fontsize=8)
        cbar.ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f'{v:.1%}'))
        savefig_multi(fig, out_prev / f'fig_5.3d_heatmap_{suffix}')
        plt.close(fig)
        print(f'  ✓ fig_5.3d_heatmap_{suffix}')


# ================================================================ #
# 综合可视化
# ================================================================ #
def run_visualization(df, df_imp, df_kw, df_pd, best_params, out_prev):
    print('\n[综合可视化]')

    # -------- 图 (a): RF importance + KW η² -------- #
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(ONE_HALF_COL, SINGLE_COL * 0.85),
                                   constrained_layout=True)
    n = len(df_imp)
    colors = [HP_COLORS[r['hp_name']] for _, r in df_imp.iterrows()]
    ax1.barh(range(n), df_imp['perm_importance_pct'].values,
             color=colors, edgecolor='#555', lw=0.4, height=0.6)
    ax1.set_yticks(range(n))
    ax1.set_yticklabels([HP_CONFIG[r]['short'] for r in df_imp['hp_name']], fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel('Permutation importance (%)', fontsize=7.5)
    ax1.set_title('(a) RF importance', fontsize=9, loc='left')
    for i, pct in enumerate(df_imp['perm_importance_pct'].values):
        ax1.text(pct + 0.5, i, f'{pct:.1f}%', va='center', fontsize=7)

    df_kw_sorted = df_kw.sort_values('eta_squared', ascending=False)
    colors2 = [HP_COLORS[r['hp_name']] for _, r in df_kw_sorted.iterrows()]
    ax2.barh(range(len(df_kw_sorted)), df_kw_sorted['eta_squared'].values,
             color=colors2, edgecolor='#555', lw=0.4, height=0.6)
    ax2.set_yticks(range(len(df_kw_sorted)))
    ax2.set_yticklabels([HP_CONFIG[r]['short'] for r in df_kw_sorted['hp_name']], fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel(r'Kruskal-Wallis $\eta^2$ (effect size)', fontsize=7.5)
    ax2.set_title(r'(b) KW $\eta^2$', fontsize=9, loc='left')
    ax2.axvline(0.06, color='#aaa', ls=':', lw=0.6)
    ax2.axvline(0.14, color='#aaa', ls=':', lw=0.6)
    ax2.text(0.06, -0.5, 'medium', fontsize=5.5, color='#888', ha='center')
    ax2.text(0.14, -0.5, 'large', fontsize=5.5, color='#888', ha='center')
    for i, (_, r) in enumerate(df_kw_sorted.iterrows()):
        sig = '*' if r['p_value'] < 0.05 else ''
        ax2.text(r['eta_squared'] + 0.003, i,
                 f'{r["eta_squared"]:.3f}{sig}', va='center', fontsize=7)

    savefig_multi(fig, out_prev / 'fig_5.3a_importance_comparison')
    plt.close(fig)
    print('  ✓ fig_5.3a_importance_comparison')

    # -------- 图 (b): Partial Dependence -------- #
    fig, axes = plt.subplots(1, 5, figsize=(DOUBLE_COL, SINGLE_COL * 0.75),
                             constrained_layout=True, sharey=True)
    for ax_i, (ax, hp) in enumerate(zip(axes, HP_ORDER)):
        cfg = HP_CONFIG[hp]
        c = HP_COLORS[hp]
        sub = df_pd[df_pd['hp_name'] == hp].sort_values('hp_value')
        x = range(len(sub))
        ax.plot(x, sub['MRE_pd_mean'].values, 'o-', color=c,
                lw=1.2, ms=6, mfc='white', mew=1.5, zorder=3)
        ax.fill_between(x,
                        sub['MRE_pd_mean'].values - sub['MRE_pd_std'].values,
                        sub['MRE_pd_mean'].values + sub['MRE_pd_std'].values,
                        color=c, alpha=0.15, lw=0)
        ax.set_xticks(x)
        ax.set_xticklabels(sub['hp_value_str'].values, fontsize=6.5,
                           rotation=30 if cfg['type'] == 'sci' else 0,
                           ha='right' if cfg['type'] == 'sci' else 'center')
        ax.set_xlabel(cfg['short'], fontsize=8)
        if ax_i == 0:
            ax.set_ylabel('Partial dependence\n(mean MRE across other HPs)', fontsize=7)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0%}'))
        for xi, val in zip(x, sub['MRE_pd_mean'].values):
            ax.text(xi, val + 0.003, f'{val:.3f}', ha='center', va='bottom', fontsize=6)
    savefig_multi(fig, out_prev / 'fig_5.3b_partial_dependence')
    plt.close(fig)
    print('  ✓ fig_5.3b_partial_dependence')

    # -------- 图 (c): Scatter + Boxplot -------- #
    fig, axes = plt.subplots(1, 5, figsize=(DOUBLE_COL, SINGLE_COL * 0.85),
                             constrained_layout=True, sharey=True)
    for ax_i, (ax, hp) in enumerate(zip(axes, HP_ORDER)):
        cfg = HP_CONFIG[hp]
        c = HP_COLORS[hp]
        vals = sorted(df[hp].unique())
        val_to_x = {v: i for i, v in enumerate(vals)}

        np.random.seed(42)
        x_jit = [val_to_x[v] + np.random.uniform(-0.18, 0.18) for v in df[hp]]
        ax.scatter(x_jit, df['MRE_mean'], s=14, alpha=0.5,
                   color=c, edgecolor='none', zorder=2)

        bp_data = [df[df[hp] == v]['MRE_mean'].values for v in vals]
        bp = ax.boxplot(bp_data, positions=range(len(vals)), widths=0.45,
                        patch_artist=True, showfliers=False, zorder=3,
                        medianprops=dict(color=MACARON_BROWN, lw=1.2),
                        whiskerprops=dict(color='#555', lw=0.6),
                        capprops=dict(color='#555', lw=0.6))
        for patch in bp['boxes']:
            patch.set_facecolor(c); patch.set_alpha(0.35)
            patch.set_edgecolor('#555'); patch.set_linewidth(0.5)

        best_val = best_params.get(hp)
        if best_val in val_to_x:
            bx = val_to_x[best_val]
            ax.plot(bx, df['MRE_mean'].min() - 0.005, marker='^',
                    ms=7, color=MACARON_BROWN, zorder=5, clip_on=False)

        for i, v_data in enumerate(bp_data):
            if len(v_data) > 0:
                med = np.median(v_data)
                ax.text(i, med - 0.004, f'{med:.3f}', ha='center', va='top',
                        fontsize=5.5, color='#333')

        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([fmt_val(v, cfg['type']) for v in vals], fontsize=6.5,
                           rotation=30 if cfg['type'] == 'sci' else 0,
                           ha='right' if cfg['type'] == 'sci' else 'center')
        ax.set_xlabel(cfg['short'], fontsize=8)
        if ax_i == 0:
            ax.set_ylabel('10-fold CV MRE', fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0%}'))

    y_lo = max(0, np.percentile(df['MRE_mean'], 2) - 0.02)
    y_hi = np.percentile(df['MRE_mean'], 98) + 0.02
    for ax in axes:
        ax.set_ylim(y_lo, y_hi)
    savefig_multi(fig, out_prev / 'fig_5.3c_scatter_boxplot')
    plt.close(fig)
    print('  ✓ fig_5.3c_scatter_boxplot')


# ================================================================ #
# 主入口
# ================================================================ #
def main():
    ap = argparse.ArgumentParser(description='§5.3 超参数敏感性 (纯 CSV)')
    ap.add_argument('--tpe_dir', default='./tpe_search_v11')
    ap.add_argument('--out_dir', default='./v11_analysis')
    args = ap.parse_args()

    tpe_dir = Path(args.tpe_dir)
    out_dir = Path(args.out_dir)
    out_tbl = out_dir / 'tables'; out_prev = out_dir / 'previews'
    ensure_dir(out_tbl); ensure_dir(out_prev)

    tpe_csv = tpe_dir / 'tpe_results.csv'
    if not tpe_csv.exists():
        raise FileNotFoundError(f'{tpe_csv} not found.')
    df = pd.read_csv(tpe_csv)
    print(f'Reading: {tpe_csv}  ({len(df)} valid trials)')

    meta_path = tpe_dir / 'tpe_meta.json'
    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        best_params = meta.get('best_params', {})
        print(f'Best: trial #{meta.get("best_trial")}  MRE={meta.get("best_mre"):.4f}')
    else:
        best_row = df.loc[df['MRE_mean'].idxmin()]
        best_params = {hp: best_row[hp] for hp in HP_ORDER}

    df_imp = run_rf_importance(df, out_tbl)
    df_kw  = run_kruskal_wallis(df, out_tbl)
    df_pd  = run_partial_dependence(df, best_params, out_tbl)

    run_heatmaps(df, best_params, out_tbl, out_prev)
    run_visualization(df, df_imp, df_kw, df_pd, best_params, out_prev)

    # 综合排名表
    df_rank = df_imp[['hp_name', 'perm_importance_pct']].merge(
        df_kw[['hp_name', 'eta_squared', 'p_value', 'effect_size']], on='hp_name')
    for hp in HP_ORDER:
        meds = df.groupby(hp)['MRE_mean'].median()
        df_rank.loc[df_rank['hp_name'] == hp, 'MRE_range'] = meds.max() - meds.min()
    df_rank = df_rank.sort_values('perm_importance_pct', ascending=False).reset_index(drop=True)
    df_rank.insert(0, 'rank', range(1, len(df_rank) + 1))
    save_table(df_rank, out_tbl / 'table_5.3_combined_ranking.csv')

    print('\n' + '=' * 75)
    print('  §5.3 超参数敏感性综合排名')
    print('=' * 75)
    print(f'  {"#":>2} {"HP":<16} {"RF imp%":>8} {"KW η²":>8} {"p-val":>8} '
          f'{"Effect":>8} {"MRE rng":>8}')
    print('-' * 75)
    for _, r in df_rank.iterrows():
        print(f'  {int(r["rank"]):2d} {r["hp_name"]:<16} '
              f'{r["perm_importance_pct"]:7.1f}% '
              f'{r["eta_squared"]:8.4f} '
              f'{r["p_value"]:8.4f} '
              f'{r["effect_size"]:>8} '
              f'{r["MRE_range"]:8.4f}')
    print('=' * 75)

    print(f'\nDone → {out_dir.resolve()}')


if __name__ == '__main__':
    main()