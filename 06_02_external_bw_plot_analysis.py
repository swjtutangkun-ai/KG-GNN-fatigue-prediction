# -*- coding: utf-8 -*-
"""
BW_4_plot_analysis.py — BW 外部泛化测试图表生成

输入:
  ./v11_bw_final/bw_ensemble.csv    (跨 seed 集成结果, 14 行)
  ./v11_bw_final/bw_preds.csv       (每 seed 原始预测, 70 行)
  ./v11_bw_final/meta.json          (汇总指标)

输出:
  ./v11_bw_final/figs/
    fig_bw_scatter.{png,pdf,svg}    预测 vs 真值散点 (log10 + linear, 2 panel)
    fig_bw_spec_mre.{png,pdf,svg}   逐试件 MRE bar + 集成 ± 跨 seed std
  ./v11_bw_final/tabs/
    tab_bw_detail.csv               14 根逐根明细 (可直接插入论文附录)
    tab_bw_summary.csv              BW vs §4.2 CV vs §4.4 LOO 指标对比

与论文图 5 风格一致 (图 5(a) log10 散点, 图 5(b) 线性空间散点), 用于 §4.4 后
新增小节 "§4.5 外部试件泛化测试 (BWJ)".
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ============================================================ #
# 绘图风格 (与论文图 5 一致)
# ============================================================ #
mpl.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
    'font.size':         10,
    'axes.labelsize':    11,
    'axes.titlesize':    11,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   9,
    'axes.linewidth':    0.9,
    'xtick.major.width': 0.9,
    'ytick.major.width': 0.9,
    'savefig.dpi':       600,
    'savefig.bbox':      'tight',
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
})

# 腐蚀等级配色 (BW 只有 20 和 60 两级)
CORR_COLOR = {
    20: '#3b7bb8',   # 蓝
    60: '#c63a4a',   # 红
}
CORR_MARKER = {20: 'o', 60: 's'}

IN_DIR  = './v11_bw_final'
FIG_DIR = os.path.join(IN_DIR, 'figs')
TAB_DIR = os.path.join(IN_DIR, 'tabs')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)


def save_fig(fig, name):
    """统一保存 3 种格式."""
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'))
    print(f'  保存 {name}.{{png,pdf,svg}}')


# ============================================================ #
# 读数据
# ============================================================ #
df_ens = pd.read_csv(os.path.join(IN_DIR, 'bw_ensemble.csv'))
df_preds = pd.read_csv(os.path.join(IN_DIR, 'bw_preds.csv'))
with open(os.path.join(IN_DIR, 'meta.json'), 'r', encoding='utf-8') as f:
    meta = json.load(f)

print(f'加载 bw_ensemble.csv: {len(df_ens)} 行')
print(f'加载 bw_preds.csv   : {len(df_preds)} 行 '
      f'({df_preds["seed"].nunique()} seeds × {df_preds["ID"].nunique()} specimens)\n')

bw_mre  = meta['bw_ensemble_MRE']
bw_r2   = meta['bw_ensemble_R2_logN']
bw_r2N  = meta['bw_ensemble_R2_N']
bw_rmse = meta['bw_ensemble_RMSE_N']
bw_p2x  = meta['bw_ensemble_P2X']


# ============================================================ #
# Fig. 15 — 预测 vs 真值散点 (双子图, 对标论文图 5(a)(b))
# ============================================================ #
fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4))

# ---- (a) log10 空间 ----
ax = axes[0]
t_log = df_ens['true_log10N'].values
p_log = df_ens['pred_log10N_mean'].values
p_std = df_ens['pred_log10N_std'].values

# 2× band = log10(2) ≈ 0.301 宽度 (在 log10N 空间等价于 factor 2)
lo = min(t_log.min(), p_log.min()) - 0.15
hi = max(t_log.max(), p_log.max()) + 0.15
xs = np.linspace(lo, hi, 200)

ax.fill_between(xs, xs - np.log10(2.0), xs + np.log10(2.0),
                color='0.88', alpha=0.6, zorder=0, label='2× scatter band')
ax.plot(xs, xs, color='0.30', lw=1.2, ls='--', zorder=1, label='Ideal')

for corr in [20, 60]:
    sub = df_ens[df_ens['corrosion_hours'] == corr]
    if len(sub) == 0:
        continue
    ax.errorbar(sub['true_log10N'], sub['pred_log10N_mean'],
                yerr=sub['pred_log10N_std'],
                fmt=CORR_MARKER[corr],
                color=CORR_COLOR[corr],
                mec='white', mew=0.8,
                ms=7, elinewidth=1.0, capsize=2.5, alpha=0.92,
                label=f'BWJ {corr}-day',
                zorder=3 if corr == 20 else 2)

ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_aspect('equal')
ax.set_xlabel(r'True $\log_{10}(N)$')
ax.set_ylabel(r'Predicted $\log_{10}(N)$')
ax.grid(True, ls=':', lw=0.5, color='0.75', zorder=0)
ax.text(0.045, 0.955,
        f'$R^2$ = {bw_r2:.3f}\nMRE = {bw_mre*100:.2f}%\n$P_{{2\\times}}$ = {bw_p2x*100:.1f}%',
        transform=ax.transAxes, va='top', ha='left',
        fontsize=9.5,
        bbox=dict(facecolor='white', edgecolor='0.6', boxstyle='round,pad=0.35', lw=0.6))
ax.legend(loc='lower right', frameon=True, framealpha=0.95,
          edgecolor='0.6', fancybox=False, fontsize=8.5)
ax.set_title('(a)', loc='left', fontweight='bold', fontsize=11)

# ---- (b) 线性空间 ----
ax = axes[1]
t_N = df_ens['true_N'].values
p_N = df_ens['pred_N_ensemble'].values

lo_N = 0
hi_N = max(t_N.max(), p_N.max()) * 1.08
xs_N = np.linspace(lo_N, hi_N, 200)

ax.fill_between(xs_N, xs_N / 2.0, xs_N * 2.0,
                color='0.88', alpha=0.6, zorder=0, label='2× scatter band')
ax.plot(xs_N, xs_N, color='0.30', lw=1.2, ls='--', zorder=1, label='Ideal')

for corr in [20, 60]:
    sub = df_ens[df_ens['corrosion_hours'] == corr]
    if len(sub) == 0:
        continue
    ax.scatter(sub['true_N'], sub['pred_N_ensemble'],
               s=60, marker=CORR_MARKER[corr],
               color=CORR_COLOR[corr], edgecolors='white', linewidth=0.8,
               alpha=0.92, label=f'BWJ {corr}-day',
               zorder=3 if corr == 20 else 2)

ax.set_xlim(lo_N, hi_N); ax.set_ylim(lo_N, hi_N)
ax.set_aspect('equal')
ax.set_xlabel(r'True $N$ (cycles)')
ax.set_ylabel(r'Predicted $N$ (cycles)')

def sci_fmt(x, _):
    if x == 0: return '0'
    return f'{x/1e6:.1f}×10⁶' if x >= 1e6 else f'{x/1e3:.0f}×10³'
ax.xaxis.set_major_formatter(plt.FuncFormatter(sci_fmt))
ax.yaxis.set_major_formatter(plt.FuncFormatter(sci_fmt))
ax.tick_params(axis='x', rotation=0)

ax.grid(True, ls=':', lw=0.5, color='0.75', zorder=0)
ax.text(0.045, 0.955,
        f'$R^2$ = {bw_r2N:.3f}\nRMSE = {bw_rmse:.0f} cycles',
        transform=ax.transAxes, va='top', ha='left',
        fontsize=9.5,
        bbox=dict(facecolor='white', edgecolor='0.6', boxstyle='round,pad=0.35', lw=0.6))
ax.legend(loc='lower right', frameon=True, framealpha=0.95,
          edgecolor='0.6', fancybox=False, fontsize=8.5)
ax.set_title('(b)', loc='left', fontweight='bold', fontsize=11)

plt.tight_layout()
save_fig(fig, 'fig_bw_scatter')
plt.close(fig)


# ============================================================ #
# Fig. 16 — 逐试件 MRE bar, 附 ±2× 可视化
# ============================================================ #
fig, ax = plt.subplots(figsize=(10.5, 4.2))

df_plot = df_ens.sort_values(['corrosion_hours', 'ID']).reset_index(drop=True)
x = np.arange(len(df_plot))

# 每根试件 seed-to-seed 的 MRE 分布 (误差棒)
seed_mre = df_preds.groupby('ID')['rel_error'].agg(['mean', 'std']).reset_index()
seed_mre = df_plot.merge(seed_mre, on='ID', how='left', suffixes=('', '_seed'))

colors = [CORR_COLOR[c] for c in df_plot['corrosion_hours'].values]

bars = ax.bar(x, df_plot['rel_error_ensemble'] * 100,
              color=colors, edgecolor='white', linewidth=0.6,
              alpha=0.88, zorder=2)
# 把每个 seed 各自的 MRE 画成小散点 (透明度 0.35), 显示 seed 间一致性
for i, sid in enumerate(df_plot['ID']):
    per_seed = df_preds[df_preds['ID'] == sid]['rel_error'].values * 100
    ax.scatter(np.full_like(per_seed, x[i], dtype=float), per_seed,
               s=12, color='0.15', alpha=0.45, zorder=3, linewidth=0)

# 参考线: BW 集成 MRE, 论文 §4.4 LOO MRE=18.66%, §4.2 CV MRE=15.35%
ax.axhline(bw_mre * 100, color='#c63a4a', ls='-', lw=1.1, alpha=0.85,
           zorder=1, label=f'BW ensemble MRE = {bw_mre*100:.2f}%')
ax.axhline(18.66, color='#3b7bb8', ls='--', lw=1.0, alpha=0.8,
           zorder=1, label='§4.4 LOO MRE = 18.66%')
ax.axhline(15.35, color='#2a7a44', ls=':', lw=1.0, alpha=0.8,
           zorder=1, label='§4.2 CV ensemble MRE = 15.35%')

ax.set_xticks(x)
ax.set_xticklabels(df_plot['ID'].values, rotation=40, ha='right', fontsize=8.5)
ax.set_ylabel('Relative error (%)')
ax.set_xlabel('BWJ specimen')
ax.grid(True, axis='y', ls=':', lw=0.5, color='0.75', zorder=0)

# 图例 (corrosion + 参考线)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_items = [
    Patch(facecolor=CORR_COLOR[20], edgecolor='white', label='BWJ 20-day'),
    Patch(facecolor=CORR_COLOR[60], edgecolor='white', label='BWJ 60-day'),
    Line2D([0], [0], color='0.15', marker='o', ls='', ms=4, label='per-seed MRE'),
    Line2D([0], [0], color='#c63a4a', lw=1.1, label=f'BW ensemble ({bw_mre*100:.2f}%)'),
    Line2D([0], [0], color='#3b7bb8', ls='--', lw=1.0, label='§4.4 LOO (18.66%)'),
    Line2D([0], [0], color='#2a7a44', ls=':', lw=1.0, label='§4.2 CV (15.35%)'),
]
ax.legend(handles=legend_items, loc='upper left',
          ncol=2, frameon=True, framealpha=0.95,
          edgecolor='0.6', fancybox=False, fontsize=8.2)

y_top = max(df_plot['rel_error_ensemble'].max() * 100 * 1.25, 40)
ax.set_ylim(0, y_top)
ax.set_xlim(-0.7, len(x) - 0.3)

plt.tight_layout()
save_fig(fig, 'fig_bw_spec_mre')
plt.close(fig)


# ============================================================ #
# Table 1 — 14 根试件逐根明细
# ============================================================ #
df_tab = df_ens.copy()
df_tab = df_tab.rename(columns={
    'ID':                    'Specimen',
    'corrosion_hours':       'Corrosion (day)',
    'true_N':                'True N (cycles)',
    'pred_N_ensemble':       'Predicted N (cycles)',
    'rel_error_ensemble':    'Relative error',
    'pred_log10N_mean':      'log10(N) predicted',
    'pred_log10N_std':       'log10(N) std (seed)',
    'DS_MPa_mean':           'ΔS_eq (MPa)',
})
cols_keep = ['Specimen', 'Corrosion (day)', 'True N (cycles)',
             'Predicted N (cycles)', 'Relative error',
             'log10(N) predicted', 'log10(N) std (seed)', 'ΔS_eq (MPa)']
df_tab = df_tab[cols_keep].copy()
df_tab['True N (cycles)']       = df_tab['True N (cycles)'].astype(int)
df_tab['Predicted N (cycles)']  = df_tab['Predicted N (cycles)'].round(0).astype(int)
df_tab['Relative error']        = (df_tab['Relative error'] * 100).round(2).astype(str) + '%'
df_tab['log10(N) predicted']    = df_tab['log10(N) predicted'].round(3)
df_tab['log10(N) std (seed)']   = df_tab['log10(N) std (seed)'].round(4)
df_tab['ΔS_eq (MPa)']           = df_tab['ΔS_eq (MPa)'].round(1)
df_tab.to_csv(os.path.join(TAB_DIR, 'tab_bw_detail.csv'), index=False, encoding='utf-8-sig')
print(f'  保存 tab_bw_detail.csv')


# ============================================================ #
# Table 2 — 三种评估协议汇总对比 (§4.2 / §4.4 / 本测试)
# ============================================================ #
A_mean = meta['A_mean']; A_std = meta['A_std']
m_mean = meta['m_mean']; m_std = meta['m_std']

tab_summary = pd.DataFrame([
    {
        'Evaluation protocol': '§4.2 Ensemble CV (5 seeds × 10 folds)',
        'Training set': 'DJ/TX/UL, fold-dependent (~102)',
        'Test set': 'DJ/TX/UL, held-out fold (~12)',
        'n (test)': 120,
        'MRE (%)': 15.35,
        'R²(log₁₀N)': 0.917,
        'R²(N)': 0.876,
        'P_2× (%)': 97.4,
        'A': 11.93,
        'm': 3.03,
    },
    {
        'Evaluation protocol': '§4.4 LOO (t* = 69)',
        'Training set': 'DJ/TX/UL, leave-one-out (119)',
        'Test set': 'DJ/TX/UL, held-out specimen (1)',
        'n (test)': 120,
        'MRE (%)': 18.66,
        'R²(log₁₀N)': 0.884,
        'R²(N)': np.nan,
        'P_2× (%)': np.nan,
        'A': 11.96,
        'm': 3.00,
    },
    {
        'Evaluation protocol': '§4.5 External BWJ test (5-seed ensemble)',
        'Training set': 'DJ/TX/UL, full (114)',
        'Test set': 'BWJ, unseen joint type (14)',
        'n (test)': len(df_ens),
        'MRE (%)': round(bw_mre * 100, 2),
        'R²(log₁₀N)': round(bw_r2, 3),
        'R²(N)': round(bw_r2N, 3),
        'P_2× (%)': round(bw_p2x * 100, 1),
        'A': round(A_mean, 2),
        'm': round(m_mean, 2),
    },
])
tab_summary.to_csv(os.path.join(TAB_DIR, 'tab_bw_summary.csv'),
                   index=False, encoding='utf-8-sig')
print(f'  保存 tab_bw_summary.csv')


# ============================================================ #
# 终端漂亮打印两张表
# ============================================================ #
print('\n' + '=' * 92)
print('Table — BW 外部泛化测试逐根明细 (14 specimens, 5-seed ensemble):')
print('=' * 92)
print(df_tab.to_string(index=False))

print('\n' + '=' * 92)
print('Table — 评估协议对比: §4.2 CV vs §4.4 LOO vs §4.5 外部 BWJ 测试')
print('=' * 92)
print(tab_summary.to_string(index=False))

print('\n输出:')
print(f'  Figures  -> {FIG_DIR}/')
print(f'    fig_bw_scatter.{{png,pdf,svg}}    — 预测 vs 真值散点 (对标论文图 5(a)(b))')
print(f'    fig_bw_spec_mre.{{png,pdf,svg}}   — 逐试件 MRE bar (含 per-seed 散点 + 参考线)')
print(f'  Tables   -> {TAB_DIR}/')
print(f'    tab_bw_detail.csv     — 14 行逐试件明细')
print(f'    tab_bw_summary.csv    — §4.2 vs §4.4 vs §4.5 汇总对比')
