# -*- coding: utf-8 -*-
"""
plot_style.py — 顶刊级 matplotlib 配置 (马卡龙配色版)

输出: PNG (600 dpi) + PDF (矢量, 字体嵌入) + SVG (矢量)
字体: Arial 正文, STIX (≈ Times New Roman) 数学符号
色板: 马卡龙系列 5 色 (薰衣草 / 抹茶绿 / 杏橘 / 玫瑰粉 / 可可棕)
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ==================== 栏宽常量 (英寸, Elsevier 标准) ==================== #
SINGLE_COL   = 3.5    # 单栏 ≈ 90 mm
ONE_HALF_COL = 5.5    # 1.5 栏
DOUBLE_COL   = 7.2    # 双栏 ≈ 190 mm

# ==================== 马卡龙色卡 ==================== #
MACARON_LAVENDER = '#9593B0'
MACARON_GREEN    = '#8FBF84'
MACARON_APRICOT  = '#DDA46E'
MACARON_ROSE     = '#C0968C'
MACARON_BROWN    = '#7D4838'

PALETTE_MACARON = [
    MACARON_LAVENDER, MACARON_GREEN, MACARON_APRICOT,
    MACARON_ROSE, MACARON_BROWN,
    '#6B6B8D', '#5A8A50', '#B07840',
]

JT_COLORS = {
    'DJ': MACARON_LAVENDER,
    'TX': MACARON_APRICOT,
    'UL': MACARON_GREEN,
}

# 按腐蚀时长的标记符号 (scatter 用)
CT_MARKERS = {0: 'o', 20: 's', 40: '^', 60: 'D'}

CT_COLORS = {0: '#C8C6D8', 20: '#9593B0', 40: '#6B6B8D', 60: '#45455E'}

MODEL_LINE_COLOR = MACARON_BROWN
IIW_REF_COLOR    = '#888888'


def apply_style():
    available = {f.name for f in font_manager.fontManager.ttflist}
    sans = 'Arial' if 'Arial' in available else 'DejaVu Sans'
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': [sans, 'Helvetica', 'DejaVu Sans'],
        'font.size': 9,
        'mathtext.fontset': 'stix',
        'mathtext.default': 'it',
        'axes.titlesize': 10, 'axes.labelsize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'legend.fontsize': 8, 'figure.titlesize': 10,
        'axes.linewidth': 0.75, 'lines.linewidth': 1.25,
        'lines.markersize': 4, 'patch.linewidth': 0.5, 'grid.linewidth': 0.4,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.major.width': 0.75, 'ytick.major.width': 0.75,
        'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
        'xtick.top': True, 'ytick.right': True,
        'axes.prop_cycle': mpl.cycler(color=PALETTE_MACARON),
        'axes.edgecolor': '#333333',
        'figure.dpi': 150, 'savefig.dpi': 600,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
        'savefig.facecolor': 'white', 'savefig.transparent': False,
        'legend.frameon': False, 'legend.handlelength': 1.6,
        'axes.spines.top': True, 'axes.spines.right': True,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none',   # SVG 里保留可编辑文字
    })


def savefig_multi(fig, path_noext, formats=('png', 'pdf', 'svg')):
    """同时导出 PNG (600 dpi) + PDF (矢量) + SVG (矢量)."""
    for fmt in formats:
        fig.savefig(f'{path_noext}.{fmt}', format=fmt)


# 向后兼容
savefig_dual = savefig_multi


def italic_math(s):
    if '_' in s:
        base, sub = s.split('_', 1)
        return rf'${base}_{{{sub}}}$'
    return rf'${s}$'
