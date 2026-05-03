# -*- coding: utf-8 -*-
"""
4_3b_baseline_handcrafted.py — §4.3 基准方法对比 (B 组: 最佳手工特征)

输入: hotspot 6D + 应力统计量 18D = 24D
  应力统计量: 每个分量 (σ₁,τ_max,σ_vm) × 6 统计量 (mean,std,max,min,p95,median)
  hotspot: F_concentration, F_volume, F_gradient, F_radius, F_n_hotspots, F_separation

4 个 baseline (传统 ML 最佳状态):
  B-1b: Random Forest   (24D)
  B-2b: XGBoost          (24D)
  B-3b: SVR-RBF          (24D)
  B-4b: GPR              (24D)

CV: 复用 v11_final/cv_splits.json
Seeds: [11, 22, 33] (3 seeds × 10 fold = 30 folds)
增强: 与 V11 相同分层策略

用法:
  python 4_3b_baseline_handcrafted.py [--cv_splits ./v11_final/cv_splits.json]
                                       [--out_dir ./baseline_4_3b]
                                       [--baselines B-1b,B-2b,B-3b,B-4b]
"""
import os, json, random, pickle, time, copy, argparse
import numpy as np
import torch
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️ xgboost 未安装, B-2b 将跳过. pip install xgboost")

from config_single import (
    FEAT_DIM, HOTSPOT_DIM, AUG_PKL_PATH,
    parse_id, USE_LOG10,
)

assert USE_LOG10 and FEAT_DIM == 3

# ============================================================ #
# 配置
# ============================================================ #
HP = {
    'AUG_LOW': (2, 0.03), 'AUG_MID': (3, 0.05), 'AUG_HIGH': (6, 0.08),
}
ABLATION_SEEDS = [11, 22, 33]
N_SPLITS = 10


# ==================== 增强 ==================== #
def augment(data, noise_std):
    d = copy.deepcopy(data)
    d['x'] = d['x'].clone() + torch.randn_like(d['x']) * noise_std
    return d


def compute_life_bins(failed_data):
    y_all = np.array([d['y'].item() for d in failed_data])
    q33, q67 = np.percentile(y_all, [33, 67])
    return [(0.0, q33, *HP['AUG_LOW']), (q33, q67, *HP['AUG_MID']),
            (q67, 999.0, *HP['AUG_HIGH'])]


def get_aug_params(y_val, bins):
    for lo, hi, n, ns in bins:
        if lo <= y_val < hi: return n, ns
    raise ValueError


def do_augmentation(train_raw):
    bins = compute_life_bins(train_raw)
    aug = []
    for d in train_raw:
        n_aug, noise = get_aug_params(d['y'].item(), bins)
        for _ in range(n_aug):
            aug.append(augment(d, noise))
    return aug


# ==================== 特征提取: 24D ==================== #
def extract_handcrafted(specimens):
    """24D = hotspot 6D + 应力统计量 18D.
    应力统计量: (σ₁,τ_max,σ_vm) × (mean,std,max,min,p95,median) = 18D.
    """
    X, y, ids = [], [], []
    for sp in specimens:
        stress = sp['x'].numpy()  # (500, 3)
        # 18D 应力统计量
        stats = []
        for col in range(3):  # σ₁, τ_max, σ_vm
            s = stress[:, col]
            stats.extend([
                s.mean(), s.std(), s.max(), s.min(),
                np.percentile(s, 95), np.median(s),
            ])
        # 6D hotspot
        hs = sp.get('hotspot_desc', torch.zeros(HOTSPOT_DIM)).numpy()
        # 拼接
        X.append(np.concatenate([np.array(stats), hs]))  # 24D
        y.append(sp['y'].item())
        ids.append(sp.get('ID', ''))
    return np.array(X), np.array(y), ids


# ==================== 评估 ==================== #
def compute_metrics(y_pred, y_true):
    vp, vt = np.array(y_pred), np.array(y_true)
    logmae = float(np.mean(np.abs(vp - vt)))
    ss_r = np.sum((vt - vp) ** 2); ss_t = np.sum((vt - vt.mean()) ** 2)
    r2_log = float(1 - ss_r / ss_t) if ss_t > 0 else 0.0
    pN, tN = 10.0 ** vp, 10.0 ** vt
    mre = float(np.mean(np.abs(pN - tN) / (tN + 1e-8)))
    rmse_N = float(np.sqrt(np.mean((pN - tN) ** 2)))
    ss_rN = np.sum((tN - pN) ** 2); ss_tN = np.sum((tN - tN.mean()) ** 2)
    r2_N = float(1 - ss_rN / ss_tN) if ss_tN > 0 else 0.0
    ratio = pN / (tN + 1e-8)
    p2x = float(np.mean((ratio >= 0.5) & (ratio <= 2.0)))
    return {'MRE': mre, 'logMAE': logmae, 'R2_logN': r2_log,
            'R2_N': r2_N, 'RMSE_N': rmse_N, 'P2X_cov': p2x}


# ==================== 4 个 baseline ==================== #
def train_rf(X_train, y_train, X_val, y_val):
    model = RandomForestRegressor(
        n_estimators=500, max_depth=None, min_samples_leaf=3,
        max_features='sqrt', random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return y_pred, sum(tree.tree_.node_count for tree in model.estimators_)


def train_xgb(X_train, y_train, X_val, y_val):
    model = XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbosity=0)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)
    y_pred = model.predict(X_val)
    return y_pred, model.get_booster().trees_to_dataframe().shape[0]


def train_svr(X_train, y_train, X_val, y_val):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf', C=10.0, gamma='scale', epsilon=0.1)),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_val)
    n_sv = pipe.named_steps['svr'].support_vectors_.shape[0]
    return y_pred, n_sv


def train_gpr(X_train, y_train, X_val, y_val):
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
    ])
    X_tr_s = pipe.fit_transform(X_train)
    X_va_s = pipe.transform(X_val)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                    random_state=42, normalize_y=True)
    gpr.fit(X_tr_s, y_train)
    y_pred, y_std = gpr.predict(X_va_s, return_std=True)
    return y_pred, len(y_train)


# ==================== 统一 train_fold ==================== #
def train_fold(baseline_name, train_raw, val_raw, seed, fold_i):
    random.seed(seed); np.random.seed(seed)
    t0 = time.time()

    # 增强
    aug = do_augmentation(train_raw)
    all_train = train_raw + aug

    # 提取 24D 手工特征
    X_train, y_train, _ = extract_handcrafted(all_train)
    X_val, y_val, val_ids = extract_handcrafted(val_raw)

    n_raw, n_aug = len(train_raw), len(aug)

    if baseline_name == 'B-1b':
        y_pred, n_params = train_rf(X_train, y_train, X_val, y_val)
    elif baseline_name == 'B-2b':
        if not HAS_XGB:
            return None, None, None
        y_pred, n_params = train_xgb(X_train, y_train, X_val, y_val)
    elif baseline_name == 'B-3b':
        y_pred, n_params = train_svr(X_train, y_train, X_val, y_val)
    elif baseline_name == 'B-4b':
        y_pred, n_params = train_gpr(X_train, y_train, X_val, y_val)
    else:
        raise ValueError(f'未知 baseline: {baseline_name}')

    time_s = time.time() - t0
    metrics = compute_metrics(y_pred, y_val)

    fold_row = {**metrics, 'time_s': time_s, 'n_params': n_params,
                'n_train_raw': n_raw, 'n_train_aug': n_aug}

    spec_rows = []
    for i in range(len(y_pred)):
        true_N = float(10.0 ** y_val[i]); pred_N = float(10.0 ** y_pred[i])
        jt, k_h = parse_id(val_ids[i])
        spec_rows.append({
            'seed': seed, 'fold': fold_i, 'ID': val_ids[i],
            'joint_type': jt, 'corrosion_hours': int(k_h),
            'true_N': true_N, 'pred_N': pred_N,
            'rel_error': abs(pred_N - true_N) / (true_N + 1e-8),
            'pred_log10N': float(y_pred[i]), 'true_log10N': float(y_val[i]),
        })

    return fold_row, spec_rows, time_s


# ==================== 运行单个 baseline ==================== #
BASELINE_NAMES = {
    'B-1b': 'Random Forest (24D handcrafted)',
    'B-2b': 'XGBoost (24D handcrafted)',
    'B-3b': 'SVR-RBF (24D handcrafted)',
    'B-4b': 'GPR (24D handcrafted)',
}


def run_baseline(baseline_name, failed, cv_splits, out_dir):
    desc = BASELINE_NAMES[baseline_name]
    var_dir = os.path.join(out_dir, baseline_name)
    os.makedirs(var_dir, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'  {baseline_name}: {desc}')
    print(f'  输出: {var_dir}')
    print(f'{"="*60}')

    all_fold, all_spec = [], []
    for seed in ABLATION_SEEDS:
        print(f'\n  Seed {seed}:')
        for fold_i in range(N_SPLITS):
            key = f's{seed}_f{fold_i}'
            if key not in cv_splits: continue
            ti = cv_splits[key]['train_idx']
            vi = cv_splits[key]['val_idx']
            tf = [failed[i] for i in ti]; vf = [failed[i] for i in vi]

            print(f'  --- s={seed} f={fold_i} ---', end='')
            fr, sr, ts = train_fold(baseline_name, tf, vf, seed, fold_i)
            if fr is None:
                print(' SKIPPED'); continue
            fr['seed'] = seed; fr['fold'] = fold_i; fr['baseline'] = baseline_name
            all_fold.append(fr); all_spec.extend(sr)
            print(f' MRE={fr["MRE"]:.4f} R²={fr["R2_logN"]:.3f} | {ts:.0f}s')

            pd.DataFrame(all_fold).to_csv(os.path.join(var_dir, 'results.csv'), index=False)

    if not all_fold:
        print(f'  {baseline_name}: 无结果')
        return None

    df_fold = pd.DataFrame(all_fold)
    df_sp = pd.DataFrame(all_spec)
    df_fold.to_csv(os.path.join(var_dir, 'results.csv'), index=False)
    df_sp.to_csv(os.path.join(var_dir, 'specimen_preds.csv'), index=False)

    # 集成
    df_ens = df_sp.groupby('ID').agg(
        true_N=('true_N', 'first'), true_log10N=('true_log10N', 'first'),
        pred_log10N_mean=('pred_log10N', 'mean'), pred_log10N_std=('pred_log10N', 'std'),
        joint_type=('joint_type', 'first'), corrosion_hours=('corrosion_hours', 'first'),
        n_evals=('pred_log10N', 'count'),
    ).reset_index()
    df_ens['pred_N_ensemble'] = np.power(10.0, df_ens['pred_log10N_mean'])
    df_ens['rel_error_ensemble'] = np.abs(
        df_ens['pred_N_ensemble'] - df_ens['true_N']) / (df_ens['true_N'] + 1e-8)
    df_ens.to_csv(os.path.join(var_dir, 'specimen_ensemble.csv'), index=False)

    ens_mre = float(df_ens['rel_error_ensemble'].mean())
    t_log, p_log = df_ens['true_log10N'].values, df_ens['pred_log10N_mean'].values
    ens_r2 = float(1 - np.sum((p_log - t_log)**2) / np.sum((t_log - t_log.mean())**2))
    t_N, p_N = df_ens['true_N'].values, df_ens['pred_N_ensemble'].values
    ens_r2_N = float(1 - np.sum((p_N - t_N)**2) / np.sum((t_N - t_N.mean())**2))
    ens_p2x = float(np.mean((p_N / (t_N + 1e-8) >= 0.5) & (p_N / (t_N + 1e-8) <= 2.0)))

    meta = {
        'baseline': baseline_name, 'description': desc,
        'input_dim': 24, 'input_desc': 'hotspot 6D + stress stats 18D (mean,std,max,min,p95,median × 3)',
        'seeds': ABLATION_SEEDS, 'n_folds': N_SPLITS,
        'fold_MRE_mean': float(df_fold['MRE'].mean()),
        'fold_MRE_std':  float(df_fold['MRE'].std()),
        'fold_R2_logN_mean': float(df_fold['R2_logN'].mean()),
        'fold_R2_N_mean': float(df_fold['R2_N'].mean()),
        'fold_P2X_mean': float(df_fold['P2X_cov'].mean()),
        'ensemble_MRE': ens_mre, 'ensemble_R2_logN': ens_r2,
        'ensemble_R2_N': ens_r2_N, 'ensemble_P2X': ens_p2x,
        'n_params': int(df_fold['n_params'].iloc[0]),
        'avg_time_s': float(df_fold['time_s'].mean()),
    }
    with open(os.path.join(var_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f'\n  {baseline_name} 汇总:')
    print(f'    fold MRE:  {meta["fold_MRE_mean"]:.4f} ± {meta["fold_MRE_std"]:.4f}')
    print(f'    ensemble:  MRE={ens_mre:.4f}  R²_logN={ens_r2:.4f}  P_2X={ens_p2x:.4f}')
    print(f'    params={meta["n_params"]}  avg_time={meta["avg_time_s"]:.1f}s')
    return meta


# ==================== 主入口 ==================== #
def main():
    ap = argparse.ArgumentParser(description='§4.3 基准对比 B 组 (24D 手工特征)')
    ap.add_argument('--cv_splits', default='./v11_final/cv_splits.json')
    ap.add_argument('--out_dir', default='./baseline_4_3b')
    ap.add_argument('--baselines', default='B-1b,B-2b,B-3b,B-4b',
                    help='逗号分隔, 可选: B-1b,B-2b,B-3b,B-4b')
    args = ap.parse_args()

    with open(AUG_PKL_PATH, 'rb') as f:
        all_data = pickle.load(f)
    failed = [d for d in all_data if not d.get('censored', False)]
    print(f'失效样本: {len(failed)}')

    with open(args.cv_splits, 'r', encoding='utf-8') as f:
        cv_splits = json.load(f)
    print(f'CV 切分: {len(cv_splits)} folds loaded')

    for seed in ABLATION_SEEDS:
        if f's{seed}_f0' not in cv_splits:
            print(f'  ⚠️ seed {seed} 不在 cv_splits 中, 自动生成')
            y_all = np.array([d['y'].item() for d in failed])
            y_bins = pd.qcut(y_all, q=N_SPLITS, labels=False, duplicates='drop')
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            for fi, (ti, vi) in enumerate(skf.split(np.zeros(len(failed)), y_bins)):
                cv_splits[f's{seed}_f{fi}'] = {
                    'seed': int(seed), 'fold': int(fi),
                    'train_idx': ti.tolist(), 'val_idx': vi.tolist()}

    baseline_names = [b.strip() for b in args.baselines.split(',')]
    print(f'Baselines: {baseline_names}\n')

    all_meta = []
    for bn in baseline_names:
        if bn not in BASELINE_NAMES:
            print(f'⚠️ 未知 baseline {bn}, 跳过'); continue
        meta = run_baseline(bn, failed, cv_splits, args.out_dir)
        if meta is not None:
            all_meta.append(meta)

    # 汇总表
    if len(all_meta) > 1:
        summary_rows = [{
            'baseline': m['baseline'], 'description': m['description'],
            'fold_MRE_mean': m['fold_MRE_mean'], 'fold_MRE_std': m['fold_MRE_std'],
            'fold_R2_logN': m['fold_R2_logN_mean'],
            'fold_R2_N': m['fold_R2_N_mean'],
            'fold_P2X': m['fold_P2X_mean'],
            'ensemble_MRE': m['ensemble_MRE'],
            'ensemble_R2_logN': m['ensemble_R2_logN'],
            'ensemble_P2X': m['ensemble_P2X'],
            'n_params': m['n_params'],
            'avg_time': m['avg_time_s'],
        } for m in all_meta]

        df_sum = pd.DataFrame(summary_rows)
        sum_path = os.path.join(args.out_dir, 'baseline_b_summary.csv')
        df_sum.to_csv(sum_path, index=False)

        print(f'\n{"="*80}')
        print(f'  §4.3 B 组汇总 → {sum_path}')
        print(f'{"="*80}')
        print(f'  {"#":<6} {"Model":<30} {"MRE±std":>14} {"R²_logN":>8} '
              f'{"P_2X":>6} {"ens_MRE":>8} {"time":>6}')
        print('-' * 80)
        for _, r in df_sum.iterrows():
            print(f'  {r["baseline"]:<6} {r["description"]:<30} '
                  f'{r["fold_MRE_mean"]:.4f}±{r["fold_MRE_std"]:.4f} '
                  f'{r["fold_R2_logN"]:8.4f} {r["fold_P2X"]:6.3f} '
                  f'{r["ensemble_MRE"]:8.4f} {r["avg_time"]:5.0f}s')
        print(f'{"="*80}')

    print(f'\n完成! 输出: {args.out_dir}/')


if __name__ == '__main__':
    main()
