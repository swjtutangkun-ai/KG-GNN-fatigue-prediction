# -*- coding: utf-8 -*-
"""
analyze_baseline.py — V10 baseline 结果可视化与表格导出 (v2)

改进:
  - 输出 PNG (600 dpi) + PDF + SVG 三种格式
  - 马卡龙配色
  - 散点图标注 R², MRE, RMSE; N 空间用线性坐标
  - 分组 MRE 柱状图柱顶标注 mean ± std
  - S-N 曲线: 按腐蚀时长分标记, 加 ±2× 分散带, 标注关键参数

用法:
  python analyze_baseline.py --result_dir ./v10_final --out_dir ./v10_analysis
"""
import os, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from plot_style import (
    apply_style, savefig_multi,
    SINGLE_COL, ONE_HALF_COL, DOUBLE_COL,
    JT_COLORS, CT_COLORS, CT_MARKERS, PALETTE_MACARON,
    MACARON_LAVENDER, MACARON_APRICOT, MACARON_GREEN,
    MACARON_ROSE, MACARON_BROWN,
    MODEL_LINE_COLOR, IIW_REF_COLOR,
)

apply_style()

JT_ORDER = ['DJ', 'TX', 'UL']
CT_ORDER = [0, 20, 40, 60]
FLOAT_FMT = '%.6g'


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def save_table(df, path, sort_cols=None):
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    df.to_csv(path, index=False, encoding='utf-8-sig', float_format=FLOAT_FMT)
    print(f'  ✓ {path.name}  ({len(df)} rows × {len(df.columns)} cols)')


def iiw_basquin_ds_vs_n(fat, m=3.0, logN_range=(4, 7), n_pts=200):
    A = 6.301 + m * np.log10(fat)
    logN = np.linspace(logN_range[0], logN_range[1], n_pts)
    logDS = (A - logN) / m
    return 10 ** logDS, 10 ** logN


# ================================================================ #
# §4.2 整体预测性能
# ================================================================ #
def run_section_42(df_fold, df_sp, df_ens, meta, out_tbl, out_prev):
    print('\n[§4.2] 整体预测性能')

    # -------- 表 4.2 -------- #
    rows = []
    for s in sorted(df_fold['seed'].unique()):
        sub = df_fold[df_fold['seed'] == s]
        rows.append({
            'group': f'seed_{s}', 'n': len(sub),
            'MRE_mean': sub['MRE'].mean(), 'MRE_std': sub['MRE'].std(),
            'logMAE_mean': sub['logMAE'].mean(),
            'R2_logN_mean': sub['R2_logN'].mean(), 'R2_N_mean': sub['R2_N'].mean(),
            'P2X_mean': sub['P2X_cov'].mean(), 'RMSE_N_mean': sub['RMSE_N'].mean(),
            'A_mean': sub['A'].mean(), 'm_mean': sub['m'].mean(),
        })
    rows.append({
        'group': 'overall_fold_avg', 'n': len(df_fold),
        'MRE_mean': df_fold['MRE'].mean(), 'MRE_std': df_fold['MRE'].std(),
        'logMAE_mean': df_fold['logMAE'].mean(),
        'R2_logN_mean': df_fold['R2_logN'].mean(), 'R2_N_mean': df_fold['R2_N'].mean(),
        'P2X_mean': df_fold['P2X_cov'].mean(), 'RMSE_N_mean': df_fold['RMSE_N'].mean(),
        'A_mean': df_fold['A'].mean(), 'm_mean': df_fold['m'].mean(),
    })
    rows.append({
        'group': 'ensemble', 'n': len(df_ens),
        'MRE_mean': meta.get('ensemble_MRE'), 'MRE_std': np.nan,
        'logMAE_mean': np.nan,
        'R2_logN_mean': meta.get('ensemble_R2_logN'), 'R2_N_mean': meta.get('ensemble_R2_N'),
        'P2X_mean': meta.get('ensemble_P2X'), 'RMSE_N_mean': meta.get('ensemble_RMSE_N'),
        'A_mean': df_fold['A'].mean(), 'm_mean': df_fold['m'].mean(),
    })
    save_table(pd.DataFrame(rows), out_tbl / 'table_4.2_seed_summary.csv')

    # -------- 图 4.1 散点数据 -------- #
    df_4_1a = df_ens[['ID', 'joint_type', 'corrosion_hours',
                      'true_log10N', 'pred_log10N_mean', 'pred_log10N_std',
                      'rel_error_ensemble']].copy()
    df_4_1a.rename(columns={
        'pred_log10N_mean': 'pred_log10N',
        'pred_log10N_std': 'pred_log10N_std_across_seeds',
        'rel_error_ensemble': 'rel_error',
    }, inplace=True)
    df_4_1a['abs_err_log'] = np.abs(df_4_1a['pred_log10N'] - df_4_1a['true_log10N'])
    df_4_1a['within_2x'] = df_4_1a['abs_err_log'] <= np.log10(2.0)
    save_table(df_4_1a, out_tbl / 'fig_4.1a_scatter_logN.csv')

    df_4_1b = df_4_1a.copy()
    df_4_1b['true_N'] = 10 ** df_4_1b['true_log10N']
    df_4_1b['pred_N'] = 10 ** df_4_1b['pred_log10N']
    df_4_1b = df_4_1b[['ID', 'joint_type', 'corrosion_hours',
                       'true_N', 'pred_N', 'rel_error', 'within_2x']]
    save_table(df_4_1b, out_tbl / 'fig_4.1b_scatter_N.csv')

    # ---- 计算统计量 ---- #
    t_log = df_4_1a['true_log10N'].values
    p_log = df_4_1a['pred_log10N'].values
    ss_r = np.sum((t_log - p_log) ** 2)
    ss_t = np.sum((t_log - t_log.mean()) ** 2)
    r2_log = 1 - ss_r / ss_t if ss_t > 0 else 0

    t_N = df_4_1b['true_N'].values
    p_N = df_4_1b['pred_N'].values
    ss_rN = np.sum((t_N - p_N) ** 2)
    ss_tN = np.sum((t_N - t_N.mean()) ** 2)
    r2_N = 1 - ss_rN / ss_tN if ss_tN > 0 else 0
    mre_ens = float(df_4_1a['rel_error'].mean())
    rmse_N = float(np.sqrt(np.mean((t_N - p_N) ** 2)))
    p2x = float(df_4_1a['within_2x'].mean())

    # ---- 图 4.1(a) log10(N) 空间 + (b) N 空间 (线性坐标) ---- #
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, SINGLE_COL * 1.05),
                             constrained_layout=True)

    # --- (a) log10(N) 空间 ---
    ax = axes[0]
    lo = min(t_log.min(), p_log.min()) - 0.15
    hi = max(t_log.max(), p_log.max()) + 0.15
    ax.plot([lo, hi], [lo, hi], '-', color='#444', lw=0.9, zorder=1)
    ax.plot([lo, hi], [lo + np.log10(2), hi + np.log10(2)],
            '--', color='#aaa', lw=0.6, zorder=1)
    ax.plot([lo, hi], [lo - np.log10(2), hi - np.log10(2)],
            '--', color='#aaa', lw=0.6, zorder=1)
    for jt in JT_ORDER:
        mask = df_4_1a['joint_type'] == jt
        ax.scatter(df_4_1a.loc[mask, 'true_log10N'],
                   df_4_1a.loc[mask, 'pred_log10N'],
                   s=22, color=JT_COLORS[jt], edgecolor='white', lw=0.4,
                   alpha=0.9, label=jt, zorder=2)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r'True $\log_{10}(N)$')
    ax.set_ylabel(r'Predicted $\log_{10}(N)$')
    ax.set_aspect('equal', 'box')
    ax.legend(loc='upper left')
    # 统计量标注框
    stats_a = (f'$R^2$ = {r2_log:.4f}\n'
               f'MRE = {mre_ens:.2%}\n'
               f'$P_{{2\\times}}$ = {p2x:.1%}')
    ax.text(0.97, 0.03, stats_a, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=7.5,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc', alpha=0.85))
    ax.set_title('(a)', loc='left', fontsize=10, fontweight='bold')

    # --- (b) N 空间 (线性坐标) ---
    ax = axes[1]
    for jt in JT_ORDER:
        mask = df_4_1b['joint_type'] == jt
        ax.scatter(df_4_1b.loc[mask, 'true_N'],
                   df_4_1b.loc[mask, 'pred_N'],
                   s=22, color=JT_COLORS[jt], edgecolor='white', lw=0.4,
                   alpha=0.9, label=jt, zorder=2)
    # y=x 和 ±2× 带 (线性空间)
    n_max = max(t_N.max(), p_N.max()) * 1.1
    xs = np.array([0, n_max])
    ax.plot(xs, xs, '-', color='#444', lw=0.9, zorder=1)
    ax.plot(xs, xs * 2.0, '--', color='#aaa', lw=0.6, zorder=1, label=r'$\pm 2\times$ band')
    ax.plot(xs, xs / 2.0, '--', color='#aaa', lw=0.6, zorder=1)
    ax.set_xlim(0, n_max); ax.set_ylim(0, n_max)
    ax.set_xlabel(r'True $N$ (cycles)')
    ax.set_ylabel(r'Predicted $N$ (cycles)')
    # 科学计数法刻度
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v/1e6:.1f}' if v >= 1e6 else f'{v/1e5:.0f}'))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v/1e6:.1f}' if v >= 1e6 else f'{v/1e5:.0f}'))
    ax.set_xlabel(r'True $N$  ($\times 10^5$ cycles)')
    ax.set_ylabel(r'Predicted $N$  ($\times 10^5$ cycles)')
    ax.set_aspect('equal', 'box')
    ax.legend(loc='upper left', fontsize=7)
    stats_b = (f'$R^2_N$ = {r2_N:.4f}\n'
               f'RMSE = {rmse_N:.0f} cycles')
    ax.text(0.97, 0.03, stats_b, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=7.5,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc', alpha=0.85))
    ax.set_title('(b)', loc='left', fontsize=10, fontweight='bold')

    savefig_multi(fig, out_prev / 'fig_4.1_scatter')
    plt.close(fig)

    # -------- 图 4.2: 分组 MRE 柱状图 (柱顶标注 mean±std) -------- #
    rows_grp = []
    for jt in JT_ORDER:
        for ct in CT_ORDER:
            sub = df_ens[(df_ens['joint_type'] == jt) &
                         (df_ens['corrosion_hours'] == ct)]
            if len(sub) == 0:
                continue
            rows_grp.append({
                'joint_type': jt, 'corrosion_hours': ct, 'n': len(sub),
                'MRE_mean': sub['rel_error_ensemble'].mean(),
                'MRE_median': sub['rel_error_ensemble'].median(),
                'MRE_std': sub['rel_error_ensemble'].std(),
                'MRE_q25': sub['rel_error_ensemble'].quantile(0.25),
                'MRE_q75': sub['rel_error_ensemble'].quantile(0.75),
                'MRE_min': sub['rel_error_ensemble'].min(),
                'MRE_max': sub['rel_error_ensemble'].max(),
            })
    df_grp = pd.DataFrame(rows_grp)
    save_table(df_grp, out_tbl / 'fig_4.2_group_mre.csv')

    fig, ax = plt.subplots(figsize=(ONE_HALF_COL, SINGLE_COL * 0.8),
                           constrained_layout=True)
    x_idx = np.arange(len(CT_ORDER))
    bar_w = 0.24
    for i, jt in enumerate(JT_ORDER):
        sub = df_grp[df_grp['joint_type'] == jt].set_index('corrosion_hours')
        vals = [sub.loc[ct, 'MRE_mean'] if ct in sub.index else 0 for ct in CT_ORDER]
        stds = [sub.loc[ct, 'MRE_std'] if ct in sub.index else 0 for ct in CT_ORDER]
        bars = ax.bar(x_idx + (i - 1) * bar_w, vals, bar_w,
                      yerr=stds, capsize=2,
                      color=JT_COLORS[jt], edgecolor='#555', lw=0.4,
                      label=jt, error_kw={'lw': 0.5, 'color': '#555'})
        # 柱顶标注 mean ± std
        for j, (bar, v, s) in enumerate(zip(bars, vals, stds)):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + s + 0.005,
                        f'{v:.1%}\n±{s:.1%}',
                        ha='center', va='bottom', fontsize=5.5, color='#333',
                        linespacing=0.9)
    ax.set_xticks(x_idx)
    ax.set_xticklabels([f'{c} h' for c in CT_ORDER])
    ax.set_xlabel('Corrosion exposure (hours)')
    ax.set_ylabel('Ensemble MRE')
    ax.legend(title=None, loc='upper left')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0%}'))
    # 留出柱顶标注的空间
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
    savefig_multi(fig, out_prev / 'fig_4.2_group_mre')
    plt.close(fig)

    # -------- 图 4.3: S-N 曲线 (丰富版) -------- #
    A_final = df_fold['A'].mean()
    m_final = df_fold['m'].mean()
    fat_init = meta.get('FAT_INIT', {'DJ': 80, 'TX': 80, 'UL': 80})

    # 模型 S-N 曲线数据
    n_pts = 200
    logN_grid = np.linspace(4, 7, n_pts)
    logDS_model = (A_final - logN_grid) / m_final
    df_sn_model = pd.DataFrame({
        'logN': logN_grid, 'N': 10 ** logN_grid,
        'logDS': logDS_model, 'DS_MPa': 10 ** logDS_model,
        'A': A_final, 'm': m_final,
    })
    save_table(df_sn_model, out_tbl / 'fig_4.3_sn_model.csv')

    # IIW 参考曲线
    rows_iiw = []
    for jt in JT_ORDER:
        fat = fat_init.get(jt, 80)
        ds, n = iiw_basquin_ds_vs_n(fat, m=3.0, logN_range=(4, 7), n_pts=n_pts)
        for d, nv in zip(ds, n):
            rows_iiw.append({
                'joint_type': jt, 'FAT': fat,
                'DS_MPa': d, 'N': nv, 'logDS': np.log10(d), 'logN': np.log10(nv),
            })
    save_table(pd.DataFrame(rows_iiw), out_tbl / 'fig_4.3_sn_iiw_reference.csv')

    # 各试件散点 (含 corrosion_hours 以区分标记)
    df_sn_sc = df_ens[['ID', 'joint_type', 'corrosion_hours',
                       'true_N', 'pred_N_ensemble', 'DS_MPa_mean']].copy()
    df_sn_sc.rename(columns={
        'pred_N_ensemble': 'pred_N', 'DS_MPa_mean': 'DS_eq_MPa',
    }, inplace=True)
    df_sn_sc['logDS_eq'] = np.log10(df_sn_sc['DS_eq_MPa'])
    df_sn_sc['logN_true'] = np.log10(df_sn_sc['true_N'])
    df_sn_sc['logN_pred'] = np.log10(df_sn_sc['pred_N'])
    save_table(df_sn_sc, out_tbl / 'fig_4.3_sn_scatter.csv')

    # ---- 画图 ---- #
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.2, SINGLE_COL * 1.2),
                           constrained_layout=True)
    ax.set_xscale('log'); ax.set_yscale('log')

    # IIW 参考线
    fat_ref = list(fat_init.values())[0] if fat_init else 80
    ds_iiw, n_iiw = iiw_basquin_ds_vs_n(fat_ref, m=3.0)
    ax.plot(ds_iiw, n_iiw, '--', color=IIW_REF_COLOR, lw=1.0,
            label=f'IIW FAT{fat_ref}  ($m$=3.0, $A$={6.301+3*np.log10(fat_ref):.2f})')

    # 学到的 S-N 曲线
    ax.plot(10 ** logDS_model, 10 ** logN_grid, '-',
            color=MODEL_LINE_COLOR, lw=1.5,
            label=f'KG-GNN  ($A$={A_final:.2f}, $m$={m_final:.2f})')

    # ±2× 分散带 (以学到的曲线为中心)
    ax.fill_between(10 ** logDS_model,
                    10 ** (logN_grid - np.log10(2)),
                    10 ** (logN_grid + np.log10(2)),
                    color=MODEL_LINE_COLOR, alpha=0.08, lw=0,
                    label=r'$\pm 2\times$ scatter band')

    # 散点: 颜色 = 接头类型, 标记 = 腐蚀时长
    for jt in JT_ORDER:
        for ct in CT_ORDER:
            sub = df_sn_sc[(df_sn_sc['joint_type'] == jt) &
                           (df_sn_sc['corrosion_hours'] == ct)]
            if len(sub) == 0:
                continue
            ax.scatter(sub['DS_eq_MPa'], sub['true_N'],
                       s=28, color=JT_COLORS[jt],
                       marker=CT_MARKERS.get(ct, 'o'),
                       edgecolor='#555', lw=0.4, alpha=0.9,
                       label=f'{jt}-{ct}h', zorder=3)

    ax.set_xlabel(r'$\Delta S_{\mathrm{eq}}$  (MPa)')
    ax.set_ylabel(r'$N$  (cycles)')

    # 两列图例: 第一列曲线/带, 第二列散点
    ax.legend(loc='lower left', fontsize=6, ncol=2,
              columnspacing=1.0, handletextpad=0.4)

    # 参数标注框 (右上角)
    param_text = (
        f'Learned parameters:\n'
        f'  $A_g$ = {A_final:.3f}  (init {meta.get("A_INIT", 12.011):.3f})\n'
        f'  $m_g$ = {m_final:.3f}  (init {meta.get("M_INIT", 3.0):.3f})\n'
        f'  $\\Delta A$ = {A_final - meta.get("A_INIT", 12.011):+.3f}\n'
        f'  $\\Delta m$ = {m_final - meta.get("M_INIT", 3.0):+.3f}'
    )
    ax.text(0.98, 0.98, param_text, transform=ax.transAxes,
            ha='right', va='top', fontsize=6.5,
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#ccc', alpha=0.85),
            family='monospace')

    savefig_multi(fig, out_prev / 'fig_4.3_sn_curve')
    plt.close(fig)


# ================================================================ #
# §4.5 训练动力学
# ================================================================ #
def run_section_45(df_fold, df_hist, meta, out_tbl, out_prev):
    print('\n[§4.5] 训练动力学')

    # ---- 选代表 fold ---- #
    med = df_fold['MRE'].median()
    idx = (df_fold['MRE'] - med).abs().idxmin()
    rep_seed = int(df_fold.loc[idx, 'seed'])
    rep_fold = int(df_fold.loc[idx, 'fold'])
    sub = df_hist[(df_hist['seed'] == rep_seed) & (df_hist['fold'] == rep_fold)]
    df_4_5a = sub[['epoch', 'train_loss', 'val_mre', 'val_logmae',
                   'A_g', 'm_g', 'wait']].copy()
    df_4_5a.insert(0, 'fold', rep_fold)
    df_4_5a.insert(0, 'seed', rep_seed)
    save_table(df_4_5a, out_tbl / 'fig_4.5a_training_curve.csv')

    # 图 4.5(a) 训练曲线
    fig, ax1 = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.8),
                            constrained_layout=True)
    l1, = ax1.plot(df_4_5a['epoch'], df_4_5a['train_loss'],
                   color=MACARON_LAVENDER, lw=1.0, label='Train loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Smooth L1 (train)', color=MACARON_LAVENDER)
    ax1.tick_params(axis='y', colors=MACARON_LAVENDER)
    ax2 = ax1.twinx()
    l2, = ax2.plot(df_4_5a['epoch'], df_4_5a['val_mre'],
                   color=MACARON_APRICOT, lw=1.0, label='Val MRE')
    ax2.set_ylabel('Validation MRE', color=MACARON_APRICOT)
    ax2.tick_params(axis='y', colors=MACARON_APRICOT)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax1.legend(handles=[l1, l2], loc='upper right', fontsize=7)
    ax1.set_title(f'Seed {rep_seed}, Fold {rep_fold}  (representative)', fontsize=9)
    savefig_multi(fig, out_prev / 'fig_4.5a_training_curve')
    plt.close(fig)

    # ---- A_g, m_g 轨迹 ---- #
    df_A = df_hist[['seed', 'fold', 'epoch', 'A_g']].copy()
    df_m = df_hist[['seed', 'fold', 'epoch', 'm_g']].copy()
    save_table(df_A, out_tbl / 'fig_4.5b_A_trajectory.csv',
               sort_cols=['seed', 'fold', 'epoch'])
    save_table(df_m, out_tbl / 'fig_4.5c_m_trajectory.csv',
               sort_cols=['seed', 'fold', 'epoch'])

    def mean_band(df_long, col):
        return df_long.groupby('epoch')[col].agg(['mean', 'std', 'min', 'max']).reset_index()

    df_A_band = mean_band(df_A, 'A_g')
    df_m_band = mean_band(df_m, 'm_g')
    save_table(df_A_band, out_tbl / 'fig_4.5b_A_trajectory_band.csv')
    save_table(df_m_band, out_tbl / 'fig_4.5c_m_trajectory_band.csv')

    A_init = meta.get('A_INIT', 12.011)
    M_init = meta.get('M_INIT', 3.0)

    fig, (axA, axM) = plt.subplots(1, 2, figsize=(ONE_HALF_COL, SINGLE_COL * 0.8),
                                   constrained_layout=True)
    for ax, band, init, ylabel, finalv, color in [
        (axA, df_A_band, A_init, r'$A_{g}$', df_fold['A'].mean(), MACARON_LAVENDER),
        (axM, df_m_band, M_init, r'$m_{g}$', df_fold['m'].mean(), MACARON_GREEN),
    ]:
        ax.fill_between(band['epoch'], band['mean'] - band['std'],
                        band['mean'] + band['std'],
                        color=color, alpha=0.18, lw=0)
        ax.plot(band['epoch'], band['mean'], color=color, lw=1.2,
                label='Mean across folds')
        ax.axhline(init, ls='--', color=IIW_REF_COLOR, lw=0.7,
                   label=f'Init = {init:.2f}')
        ax.axhline(finalv, ls=':', color=MACARON_BROWN, lw=0.9,
                   label=f'Final = {finalv:.2f}')
        ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
        ax.legend(fontsize=6.5, loc='best')
    axA.set_title('(a) Basquin intercept', loc='left', fontsize=9)
    axM.set_title('(b) Basquin slope', loc='left', fontsize=9)
    savefig_multi(fig, out_prev / 'fig_4.5bc_basquin_trajectory')
    plt.close(fig)

    # ---- best_epoch 直方图 ---- #
    df_be = df_fold[['seed', 'fold', 'best_ep', 'final_ep', 'wait_after_best']].copy()
    save_table(df_be, out_tbl / 'fig_4.5d_best_epoch_hist.csv')

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.7),
                           constrained_layout=True)
    bins = np.arange(0, df_be['best_ep'].max() + 25, 25)
    ax.hist(df_be['best_ep'], bins=bins, color=MACARON_LAVENDER,
            edgecolor='#555', lw=0.5, alpha=0.85)
    ax.axvline(df_be['best_ep'].median(), color=MACARON_BROWN, ls='--', lw=0.9,
               label=f'Median = {int(df_be["best_ep"].median())}')
    ax.set_xlabel(r'Best epoch (arg min val MRE)')
    ax.set_ylabel('Number of folds')
    ax.legend(fontsize=7)
    savefig_multi(fig, out_prev / 'fig_4.5d_best_epoch_hist')
    plt.close(fig)


# ================================================================ #
# 主入口
# ================================================================ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result_dir', default='./v11_final')
    ap.add_argument('--out_dir', default='./v11_analysis')
    args = ap.parse_args()

    result_dir = Path(args.result_dir)
    out_dir = Path(args.out_dir)
    out_tbl = out_dir / 'tables'
    out_prev = out_dir / 'previews'
    ensure_dir(out_tbl); ensure_dir(out_prev)

    print(f'Reading from: {result_dir.resolve()}')
    df_fold = pd.read_csv(result_dir / 'results.csv')
    df_sp = pd.read_csv(result_dir / 'specimen_preds.csv')
    df_ens = pd.read_csv(result_dir / 'specimen_ensemble.csv')
    df_hist = pd.read_csv(result_dir / 'training_history.csv')
    with open(result_dir / 'meta.json', 'r', encoding='utf-8') as f:
        meta = json.load(f)
    print(f'  fold={len(df_fold)}, specimens={len(df_ens)}, history={len(df_hist)}')

    run_section_42(df_fold, df_sp, df_ens, meta, out_tbl, out_prev)
    run_section_45(df_fold, df_hist, meta, out_tbl, out_prev)

    print(f'\nDone. Outputs in: {out_dir.resolve()}')
    print('  tables/     — CSV for Origin')
    print('  previews/   — PNG (600dpi) + PDF + SVG')


if __name__ == '__main__':
    main()
