# -*- coding: utf-8 -*-
"""
run_sweep_N.py — Node sampling size (N) sensitivity analysis.

────────────────────────────────────────────────────────────────
目的:
    为正文 §2.2.1 中 "N=500 为超参数" 的选择提供 paper-ready 的定量依据.
    扫描 N ∈ {200, 1000, 2000}, K_NEIGHBORS 固定 = 8, 与正文配置一致.
    N=500 的基线直接复用 ./v11_final/ 已有结果, 不重跑.

协议:
    每个 N 值用 3 seeds × 10 fold = 30 fold 交叉验证,
    所有其余超参数(HIDDEN_DIM=64, HEADS=4, DROPOUT=0.6, LR=4e-4, ...)
    与正文 V11 baseline 严格一致.

    3 配置 × 30 fold = 90 fold, 单 fold ~2 min, 预计 ~3 小时.

用法:
    # 放置位置: 与 02_*/04_* 阶段脚本及 config_single.py 同一目录
    python 05_01_node_sampling_size_sweep.py                     # 默认 3 seeds, 跳过 N=500
    python 05_01_node_sampling_size_sweep.py --n-seeds 5         # 与正文完全对齐 (50 fold/组, ~10h)
    python 05_01_node_sampling_size_sweep.py --include-500       # 同时重跑 N=500 (一致性校验用)
    python 05_01_node_sampling_size_sweep.py --skip-if-done      # 断点续跑
    python 05_01_node_sampling_size_sweep.py --dry-run           # 只打印命令, 不执行
    python 05_01_node_sampling_size_sweep.py --n-values 200 400  # 自定义 N 列表

机制:
    将 4 个阶段脚本复制到 ./sweep_N/_shim/ 下, 并在同目录放置一份
    config_single.py (shim). 从 shim 目录启动脚本时 sys.path[0] = shim,
    `from config_single import ...` 会命中 shim 而非原版; shim 按
    SWEEP_* 环境变量覆盖 TOP_N 和输出路径. cwd 保持在项目根, 相对路径
    (./DJ.xlsx, 原始 CSV 目录等) 正常解析.

    训练脚本 04_02_train_kggnn_overall_performance.py 的硬编码会被驱动 sed 替换:
        OUT_DIR = './v11_final'
        → OUT_DIR = os.environ.get('SWEEP_OUT_DIR', './v11_final')

    以及 HP 里的种子列表与 fold 数:
        'RANDOM_SEEDS':     [11, 22, 33, 44, 55],
        → 从 SWEEP_SEEDS env 解析

输出:
    ./sweep_N/
      N0200/
        data/                  ← top-N CSV (extract 产物)
        graphs/                ← k-NN 图 (build 产物)
        processed.pkl          ← preprocess 产物
        train/
          meta.json, results.csv, specimen_ensemble.csv, ...
      N1000/ ... N2000/
      sweep_summary.csv        ← 每个 N 一行
      sweep_summary.md         ← 可读表格 (含 N=500 正文基线参考行)
      sweep_curve.png          ← N vs MRE 曲线图 (可选, 需 matplotlib)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# ───────────────────── 默认扫描配置 ──────────────────────────
# N=500 默认不扫: 正文基线结果直接从 ./v11_final/meta.json 复用
DEFAULT_N_VALUES_NO_500 = [200, 1000, 2000]
SKIP_N_DEFAULT = 500

K_FIXED = 8  # 不扫描, 与正文一致 (仅用于输出目录命名与日志; 真正的 K 从原 config 读)

# 脚本名
SCRIPT_EXTRACT     = '02_02_01_node_feature_extraction.py'
SCRIPT_BUILD       = '02_02_02_graph_topology_construction.py'
SCRIPT_PREPROCESS  = '02_02_03_graph_preprocessing_and_augmentation.py'
SCRIPT_TRAIN       = '04_02_train_kggnn_overall_performance.py'
SCRIPT_TRAIN_SWEEP = '04_02_train_kggnn_sweep.py'
ORIG_CONFIG        = 'config_single.py'

# 正文基线目录 (N=500 复用)
V11_BASELINE_DIR = Path('./v11_final')

SWEEP_ROOT = Path('./sweep_N').resolve()
SHIM_DIR = (SWEEP_ROOT / '_shim').resolve()


# ───────────────────── 命令行参数 ────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(description='Node sampling size (N) sensitivity sweep')
    ap.add_argument('--n-values', type=int, nargs='+', default=DEFAULT_N_VALUES_NO_500,
                    help='要扫的 N 列表 (默认: 200 1000 2000; N=500 从 v11_final 复用)')
    ap.add_argument('--n-seeds', type=int, default=3,
                    help='每个 N 的种子数 (默认 3, 对齐正文 consistency)')
    ap.add_argument('--n-splits', type=int, default=10,
                    help='每 seed 的 fold 数 (默认 10)')
    ap.add_argument('--include-500', action='store_true',
                    help='同时重跑 N=500 做一致性校验 (不推荐: 正文已有结果)')
    ap.add_argument('--skip-if-done', action='store_true',
                    help='已有 meta.json 的 N 直接跳过')
    ap.add_argument('--dry-run', action='store_true',
                    help='只打印命令, 不执行')
    return ap.parse_args()


def banner(msg: str) -> None:
    line = '═' * 74
    print(f'\n{line}\n  {msg}\n{line}', flush=True)


# ───────────────────── shim 生成 ───────────────────────────
SHIM_CONFIG_CONTENT = """# AUTO-GENERATED shim — do not hand-edit.
# 加载原始 config_single.py 的全部符号, 再按 SWEEP_* env var 覆盖.
import os
import importlib.util as _iu

_real_path = os.environ.get('SWEEP_REAL_CONFIG')
if not _real_path or not os.path.isfile(_real_path):
    raise RuntimeError(f'SWEEP_REAL_CONFIG not set or missing: {_real_path!r}')

_spec = _iu.spec_from_file_location('config_single_real', _real_path)
_real = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_real)
for _n in dir(_real):
    if not _n.startswith('_'):
        globals()[_n] = getattr(_real, _n)

# env-var 覆盖
TOP_N = int(os.environ.get('SWEEP_TOP_N', globals().get('TOP_N', 500)))

DATA_DIR     = os.environ.get('SWEEP_DATA_DIR',     globals().get('DATA_DIR'))
GRAPH_DIR    = os.environ.get('SWEEP_GRAPH_DIR',    globals().get('GRAPH_DIR'))
AUG_PKL_PATH = os.environ.get('SWEEP_AUG_PKL_PATH', globals().get('AUG_PKL_PATH'))
RESULT_DIR   = os.environ.get('SWEEP_RESULT_DIR',   globals().get('RESULT_DIR'))

# K_NEIGHBORS 不在扫描范围内, 保持原 config 默认值 (通常 = 8)

if os.environ.get('SWEEP_VERBOSE_CFG', '0') == '1':
    print(f'[shim] TOP_N={TOP_N}  K_NEIGHBORS={K_NEIGHBORS}')
    print(f'[shim] DATA_DIR={DATA_DIR}')
    print(f'[shim] GRAPH_DIR={GRAPH_DIR}')
    print(f'[shim] AUG_PKL_PATH={AUG_PKL_PATH}')
    print(f'[shim] RESULT_DIR={RESULT_DIR}')
"""


def ensure_shim(cwd: Path) -> None:
    """写入 shim 配置 + 复制 4 个阶段脚本到 shim 目录."""
    SHIM_DIR.mkdir(parents=True, exist_ok=True)

    # (a) shim config
    (SHIM_DIR / 'config_single.py').write_text(SHIM_CONFIG_CONTENT, encoding='utf-8')
    print(f'[shim] 写入: {SHIM_DIR / "config_single.py"}')

    # (b) stage 0/1/2: 原样复制
    for sc in [SCRIPT_EXTRACT, SCRIPT_BUILD, SCRIPT_PREPROCESS]:
        src = cwd / sc
        if not src.is_file():
            raise FileNotFoundError(f'脚本缺失: {src}')
        (SHIM_DIR / sc).write_text(src.read_text(encoding='utf-8'), encoding='utf-8')
        print(f'[shim] 复制: {SHIM_DIR / sc}')

    # (c) stage 3: sed 替换生成 sweep 版本
    orig_train = cwd / SCRIPT_TRAIN
    if not orig_train.is_file():
        raise FileNotFoundError(f'训练脚本缺失: {orig_train}')
    code = orig_train.read_text(encoding='utf-8')

    # OUT_DIR 硬编码 → env var
    code_new = code.replace(
        "OUT_DIR = './v11_final'",
        "OUT_DIR = os.environ.get('SWEEP_OUT_DIR', './v11_final')",
    )
    if code_new == code:
        code_new = code.replace(
            'OUT_DIR = "./v11_final"',
            "OUT_DIR = os.environ.get('SWEEP_OUT_DIR', './v11_final')",
        )
    if code_new == code:
        raise RuntimeError(
            f"未能在 {SCRIPT_TRAIN} 中找到 OUT_DIR 硬编码行. "
            f"驱动依赖 OUT_DIR = './v11_final' 这一行."
        )
    code = code_new

    # RANDOM_SEEDS / N_SPLITS → env var
    before = code
    code = code.replace(
        "'RANDOM_SEEDS':     [11, 22, 33, 44, 55],",
        "'RANDOM_SEEDS':     [int(s) for s in "
        "os.environ.get('SWEEP_SEEDS', '11,22,33,44,55').split(',')],"
    )
    if code == before:
        raise RuntimeError("未能匹配 RANDOM_SEEDS 行, 请检查训练脚本 HP dict.")
    before = code
    code = code.replace(
        "'N_SPLITS':         10,",
        "'N_SPLITS':         int(os.environ.get('SWEEP_N_SPLITS', 10)),"
    )
    if code == before:
        raise RuntimeError("未能匹配 N_SPLITS 行, 请检查训练脚本 HP dict.")

    (SHIM_DIR / SCRIPT_TRAIN_SWEEP).write_text(code, encoding='utf-8')
    print(f'[shim] 写入: {SHIM_DIR / SCRIPT_TRAIN_SWEEP}')


# ───────────────────── 子进程调用 ──────────────────────────
def build_env(N: int, paths: dict, cwd: Path, n_seeds: int, n_splits: int) -> dict:
    env = os.environ.copy()
    env['SWEEP_REAL_CONFIG']  = str((cwd / ORIG_CONFIG).resolve())
    env['SWEEP_TOP_N']        = str(N)
    env['SWEEP_DATA_DIR']     = str(paths['data_dir'])
    env['SWEEP_GRAPH_DIR']    = str(paths['graph_dir'])
    env['SWEEP_AUG_PKL_PATH'] = str(paths['aug_pkl'])
    env['SWEEP_RESULT_DIR']   = str(paths['train_dir'])
    env['SWEEP_OUT_DIR']      = str(paths['train_dir'])
    env['SWEEP_SEEDS']        = ','.join(str(11 * (i + 1)) for i in range(n_seeds))
    env['SWEEP_N_SPLITS']     = str(n_splits)
    env['SWEEP_VERBOSE_CFG']  = '1'
    return env


def run_stage(script_name: str, env: dict, cwd: Path,
              dry_run: bool, log_path: Path) -> int:
    script_abs = SHIM_DIR / script_name
    if not dry_run and not script_abs.is_file():
        raise FileNotFoundError(f'shim 脚本缺失: {script_abs}')
    cmd = [sys.executable, '-u', str(script_abs)]
    print(f'  $ (cwd={cwd}) ' + ' '.join(cmd))
    if dry_run:
        return 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'ab') as f:
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        for line in iter(proc.stdout.readline, b''):
            sys.stdout.buffer.write(line)
            sys.stdout.flush()
            f.write(line)
        proc.wait()
    return proc.returncode


# ───────────────────── 单 N 流水线 ─────────────────────────
def run_one_N(N: int, args, cwd: Path) -> dict | None:
    tag = f'N{N:04d}'
    n_dir = SWEEP_ROOT / tag
    data_dir  = n_dir / 'data'
    graph_dir = n_dir / 'graphs'
    aug_pkl   = n_dir / 'processed.pkl'
    train_dir = n_dir / 'train'
    log_path  = n_dir / 'pipeline.log'

    paths = dict(data_dir=data_dir, graph_dir=graph_dir,
                 aug_pkl=aug_pkl, train_dir=train_dir)
    for p in paths.values():
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    meta_path = train_dir / 'meta.json'
    if args.skip_if_done and meta_path.is_file():
        print(f'[skip] {tag}: 已存在 {meta_path}')
        return read_combo_result(N, train_dir)

    env = build_env(N, paths, cwd, args.n_seeds, args.n_splits)

    # Stage 0: extract
    data_dir.mkdir(parents=True, exist_ok=True)
    extract_done = any(data_dir.glob(f'*_top{N}.csv'))
    if extract_done:
        print(f'[{tag}] Stage 0 extract: 已有 CSV, 复用')
    else:
        banner(f'[{tag}] Stage 0: extract top-{N}  →  {data_dir}')
        rc = run_stage(SCRIPT_EXTRACT, env, cwd, args.dry_run, log_path)
        if rc != 0:
            print(f'[fail] {tag} extract rc={rc}')
            return None

    # Stage 1: build
    graph_dir.mkdir(parents=True, exist_ok=True)
    banner(f'[{tag}] Stage 1: build k-NN graphs (K from config)  →  {graph_dir}')
    rc = run_stage(SCRIPT_BUILD, env, cwd, args.dry_run, log_path)
    if rc != 0:
        print(f'[fail] {tag} build rc={rc}')
        return None

    # Stage 2: preprocess
    banner(f'[{tag}] Stage 2: preprocess  →  {aug_pkl}')
    rc = run_stage(SCRIPT_PREPROCESS, env, cwd, args.dry_run, log_path)
    if rc != 0:
        print(f'[fail] {tag} preprocess rc={rc}')
        return None

    # Stage 3: train
    train_dir.mkdir(parents=True, exist_ok=True)
    banner(f'[{tag}] Stage 3: train  '
           f'({args.n_seeds} seeds × {args.n_splits} fold = '
           f'{args.n_seeds*args.n_splits} folds)  →  {train_dir}')
    rc = run_stage(SCRIPT_TRAIN_SWEEP, env, cwd, args.dry_run, log_path)
    if rc != 0:
        print(f'[fail] {tag} train rc={rc}')
        return None

    if args.dry_run:
        return None
    return read_combo_result(N, train_dir)


# ───────────────────── 结果读取 ────────────────────────────
def read_combo_result(N: int, train_dir: Path) -> dict | None:
    meta_path = Path(train_dir) / 'meta.json'
    if not meta_path.is_file():
        print(f'[warn] 未找到 {meta_path}')
        return None
    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)
    return {
        'N':                  N,
        'n_failed':           meta.get('n_failed_specimens'),
        'ensemble_MRE':       meta.get('ensemble_MRE'),
        'ensemble_R2_logN':   meta.get('ensemble_R2_logN'),
        'ensemble_R2_N':      meta.get('ensemble_R2_N'),
        'ensemble_RMSE_N':    meta.get('ensemble_RMSE_N'),
        'ensemble_P2X':       meta.get('ensemble_P2X'),
        'per_fold_MRE_mean':  meta.get('per_fold_MRE_mean'),
        'per_fold_MRE_std':   meta.get('per_fold_MRE_std'),
        'best_ep_median':     meta.get('best_ep_median'),
        'source':             str(train_dir),
    }


def try_read_baseline_500() -> dict | None:
    """从 ./v11_final/meta.json 读正文 N=500 基线, 加入汇总表."""
    meta_path = V11_BASELINE_DIR / 'meta.json'
    if not meta_path.is_file():
        print(f'[baseline] 未找到 {meta_path} —— N=500 参考行将不可用')
        return None
    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)
    return {
        'N':                  500,
        'n_failed':           meta.get('n_failed_specimens'),
        'ensemble_MRE':       meta.get('ensemble_MRE'),
        'ensemble_R2_logN':   meta.get('ensemble_R2_logN'),
        'ensemble_R2_N':      meta.get('ensemble_R2_N'),
        'ensemble_RMSE_N':    meta.get('ensemble_RMSE_N'),
        'ensemble_P2X':       meta.get('ensemble_P2X'),
        'per_fold_MRE_mean':  meta.get('per_fold_MRE_mean'),
        'per_fold_MRE_std':   meta.get('per_fold_MRE_std'),
        'best_ep_median':     meta.get('best_ep_median'),
        'source':             str(V11_BASELINE_DIR) + '  (paper baseline)',
    }


# ───────────────────── 汇总与出图 ─────────────────────────
def _df_to_markdown(df: pd.DataFrame) -> str:
    """手写 markdown 表格 (避免 pandas.to_markdown 对 tabulate 的依赖)."""
    cols = list(df.columns)
    rows = df.astype(object).where(pd.notna(df), '—').values.tolist()

    # 每列宽度 = max(header, 所有 cell)
    widths = [max(len(str(c)), *(len(str(r[i])) for r in rows)) for i, c in enumerate(cols)]

    def fmt_row(vals):
        return '| ' + ' | '.join(str(v).ljust(w) for v, w in zip(vals, widths)) + ' |'

    header = fmt_row(cols)
    sep = '| ' + ' | '.join('-' * w for w in widths) + ' |'
    body = '\n'.join(fmt_row(r) for r in rows)
    return '\n'.join([header, sep, body])


def write_summary(rows: list[dict], out_dir: Path) -> None:
    if not rows:
        print('[summary] 无结果')
        return
    df = pd.DataFrame(rows).sort_values('N').reset_index(drop=True)

    csv_path = out_dir / 'sweep_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f'[summary] {csv_path}')

    lines = ['# N Sensitivity Sweep — Summary', '']
    lines.append(f'Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'K_NEIGHBORS fixed = {K_FIXED} (read from original config)')
    lines.append('')
    lines.append('## Results table')
    lines.append('')

    cols_show = ['N', 'ensemble_MRE', 'ensemble_R2_logN',
                 'per_fold_MRE_mean', 'per_fold_MRE_std',
                 'best_ep_median', 'source']
    df_show = df[[c for c in cols_show if c in df.columns]].copy()
    fmt = {'ensemble_MRE': '{:.4f}', 'ensemble_R2_logN': '{:.3f}',
           'per_fold_MRE_mean': '{:.4f}', 'per_fold_MRE_std': '{:.4f}'}
    for col, f in fmt.items():
        if col in df_show.columns:
            df_show[col] = df_show[col].apply(
                lambda x: f.format(x) if pd.notna(x) else '—')
    lines.append(_df_to_markdown(df_show))
    lines.append('')

    md_path = out_dir / 'sweep_summary.md'
    md_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'[summary] {md_path}')

    try_make_curve(df, out_dir)


def try_make_curve(df: pd.DataFrame, out_dir: Path) -> None:
    """可选: 画 N vs MRE 曲线图. matplotlib 缺失时跳过."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[plot] matplotlib 不可用, 跳过出图')
        return

    if 'ensemble_MRE' not in df.columns or df['ensemble_MRE'].isna().all():
        return
    d = df.dropna(subset=['ensemble_MRE']).sort_values('N')

    fig, ax1 = plt.subplots(figsize=(6.2, 4.2))
    ax1.plot(d['N'], d['ensemble_MRE'] * 100, 'o-', color='#2e86ab',
             linewidth=2, markersize=9, label='Ensemble MRE')
    if 'per_fold_MRE_mean' in d.columns and 'per_fold_MRE_std' in d.columns:
        ax1.errorbar(d['N'], d['per_fold_MRE_mean'] * 100,
                     yerr=d['per_fold_MRE_std'] * 100,
                     fmt='s', color='#e76f51', alpha=0.75,
                     markersize=7, capsize=5, label='Per-fold MRE (mean ± std)')
    ax1.set_xscale('log')
    ax1.set_xlabel('N (top-N node sampling size)', fontsize=11)
    ax1.set_ylabel('MRE (%)', fontsize=11)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_xticks(d['N'].tolist())
    ax1.set_xticklabels([str(n) for n in d['N']])

    # 高亮 N=500 (正文基线)
    if 500 in d['N'].values:
        y500 = float(d[d['N'] == 500]['ensemble_MRE'].iloc[0]) * 100
        ax1.axvline(500, color='gray', linestyle=':', alpha=0.5)
        ax1.annotate(f'baseline\nN=500\nMRE={y500:.2f}%',
                     xy=(500, y500), xytext=(640, y500 + 2),
                     fontsize=9, color='dimgray',
                     arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
    ax1.legend(loc='best', fontsize=10)
    ax1.set_title('Node sampling size (N) sensitivity — K fixed',
                  fontsize=11)
    fig.tight_layout()
    fig_path = out_dir / 'sweep_curve.png'
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] {fig_path}')


# ───────────────────── 主流程 ─────────────────────────────
def main():
    args = parse_args()
    cwd = Path.cwd().resolve()

    # 工作目录健全性检查
    for sc in [SCRIPT_EXTRACT, SCRIPT_BUILD, SCRIPT_PREPROCESS,
               SCRIPT_TRAIN, ORIG_CONFIG]:
        if not (cwd / sc).is_file():
            sys.exit(f'[error] 缺少 {sc} (应与本驱动同目录)')

    # 决定扫描列表
    n_list = list(args.n_values)
    if args.include_500 and 500 not in n_list:
        n_list.append(500)
    elif (not args.include_500) and 500 in n_list:
        print(f'[info] N=500 从扫描列表移除 (正文基线, 直接复用 {V11_BASELINE_DIR})')
        n_list = [n for n in n_list if n != 500]

    n_list = sorted(set(n_list))

    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_shim(cwd)

    total_folds = len(n_list) * args.n_seeds * args.n_splits
    banner(f'开始扫描: N ∈ {n_list}  (K 固定 = {K_FIXED}, 从原 config 读)\n'
           f'  {args.n_seeds} seeds × {args.n_splits} fold = '
           f'{args.n_seeds*args.n_splits} fold / N\n'
           f'  总 fold 数: {total_folds}  (单 fold ~2min, 预计 {total_folds*2/60:.1f} h)')

    rows = []
    # 参考: 把正文 N=500 基线作为第一行 (不参与扫描)
    if not args.include_500:
        base = try_read_baseline_500()
        if base is not None:
            rows.append(base)

    t_all = time.time()
    for i, N in enumerate(n_list, 1):
        banner(f'[{i}/{len(n_list)}] N = {N}')
        t0 = time.time()
        r = run_one_N(N, args, cwd)
        dt_min = (time.time() - t0) / 60
        if r is not None:
            r['wall_min'] = round(dt_min, 2)
            rows.append(r)
            rows = sorted(rows, key=lambda x: x['N'])
            write_summary(rows, SWEEP_ROOT)  # 增量保存
        else:
            print(f'[skip/fail] N={N} ({dt_min:.1f} min)')

    banner(f'完成. 总耗时 {(time.time()-t_all)/60:.1f} min')
    write_summary(rows, SWEEP_ROOT)

    # 终端友好总结
    if rows:
        print('\n最终汇总:')
        for r in rows:
            tag = '  ← baseline' if str(r.get('source','')).endswith('(paper baseline)') else ''
            mre = r.get('ensemble_MRE')
            r2 = r.get('ensemble_R2_logN')
            mre_s = f'{mre*100:6.2f}%' if mre is not None else '   —  '
            r2_s  = f'{r2:.3f}' if r2 is not None else ' — '
            print(f'  N={r["N"]:>4d}   MRE={mre_s}   R²={r2_s}{tag}')


if __name__ == '__main__':
    main()
