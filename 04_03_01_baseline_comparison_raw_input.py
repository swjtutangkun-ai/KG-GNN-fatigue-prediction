# -*- coding: utf-8 -*-
"""
4_3_baseline_comparison.py — §4.3 基准方法对比

6 个 baseline, 输入统一为 TOP500×6 展平 = 3000D (坐标 + 应力):
  B-1: Random Forest     — 3000D 直接, 树集成
  B-2: XGBoost           — 3000D 直接, 梯度提升
  B-3: SVR (RBF)         — PCA→20D, 核方法
  B-4: GPR               — PCA→20D, 高斯过程 (小样本+UQ)
  B-5: MLP               — 3000D 直接, 3 层全连接 NN
  B-6: PointNet          — 500×6 点集 [坐标+应力], 置换不变 NN

CV: 复用 v11_final/cv_splits.json (与消融配对)
Seeds: [11, 22, 33] (3 seeds × 10 fold = 30 folds)
增强: 与 V11 相同分层策略 (文献惯例: 架构无关增强统一给所有方法)

用法:
  python 4_3_baseline_comparison.py [--cv_splits ./v11_final/cv_splits.json]
                                     [--out_dir ./baseline_4_3]
                                     [--baselines B-1,B-2,B-3,B-4,B-5,B-6]
"""
import os, json, random, pickle, time, copy, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️ xgboost 未安装, B-2 将跳过. pip install xgboost")

from config_single import (
    FEAT_DIM, HOTSPOT_DIM, AUG_PKL_PATH,
    parse_id, USE_LOG10, FAT_INIT,
)

assert USE_LOG10 and FEAT_DIM == 3

# ============================================================ #
# 配置
# ============================================================ #
HP = {
    # MLP / PointNet 训练参数 (与 V11 一致)
    'LR': 4e-4, 'WEIGHT_DECAY': 1e-4,
    'EPOCHS': 500, 'BATCH_SIZE': 4,
    'PATIENCE': 50, 'EMA_DECAY': 0.995, 'ES_MIN_DELTA': 5e-4,
    # 增强 (与 V11 一致)
    'AUG_LOW': (2, 0.03), 'AUG_MID': (3, 0.05), 'AUG_HIGH': (6, 0.08),
    # PCA
    'PCA_N_COMPONENTS': 20,
}
ABLATION_SEEDS = [11, 22, 33]
N_SPLITS = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== 数据准备 ==================== #
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


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
    """V11 分层增强."""
    bins = compute_life_bins(train_raw)
    aug = []
    for d in train_raw:
        n_aug, noise = get_aug_params(d['y'].item(), bins)
        for _ in range(n_aug):
            aug.append(augment(d, noise))
    return aug


def extract_tabular(specimens):
    """提取 3000D 表格特征: TOP500×3 坐标 + TOP500×3 应力 展平."""
    X, y, ids = [], [], []
    for sp in specimens:
        coords_flat = sp['coords'].numpy().flatten()  # (500*3,) = 1500D [x,y,z]
        stress_flat = sp['x'].numpy().flatten()        # (500*3,) = 1500D [σ₁,τ_max,σ_vm]
        X.append(np.concatenate([coords_flat, stress_flat]))  # 3000D
        y.append(sp['y'].item())
        ids.append(sp.get('ID', ''))
    return np.array(X), np.array(y), ids


def extract_pointnet_batch(specimens):
    """提取 PointNet 输入: (B, 500, 6) = [x,y,z,σ₁,τ_max,σ_vm]."""
    pts = torch.stack([
        torch.cat([sp['coords'], sp['x']], dim=1)  # (500, 6)
        for sp in specimens])  # (B, 500, 6)
    y = torch.tensor([sp['y'].item() for sp in specimens])
    return pts, y


# ==================== EMA (MLP/PointNet 用) ==================== #
class EMAModel:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.backup = {}
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)
    def apply(self, model):
        self.backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
    def restore(self, model):
        model.load_state_dict(self.backup); self.backup = {}
    def state_dict(self):
        return {k: v.cpu().clone() for k, v in self.shadow.items()}


class EarlyStopping:
    def __init__(self):
        self.patience, self.counter = HP['PATIENCE'], 0
        self.best_score, self.early_stop = None, False
        self.best_ema_state = None
    def __call__(self, mre, ema):
        s = -mre
        if self.best_score is None or s > self.best_score + HP['ES_MIN_DELTA']:
            self.best_score = s; self.best_ema_state = ema.state_dict(); self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ==================== 评估 (通用) ==================== #
def compute_metrics(y_pred, y_true):
    """输入 log10(N) 空间的预测和真值."""
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


# ============================================================ #
# B-1: Random Forest
# ============================================================ #
def train_rf(X_train, y_train, X_val, y_val):
    model = RandomForestRegressor(
        n_estimators=500, max_depth=None, min_samples_leaf=3,
        max_features='sqrt', random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return y_pred, sum(tree.tree_.node_count for tree in model.estimators_)


# ============================================================ #
# B-2: XGBoost
# ============================================================ #
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


# ============================================================ #
# B-3: SVR (RBF) + PCA
# ============================================================ #
def train_svr(X_train, y_train, X_val, y_val):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=HP['PCA_N_COMPONENTS'])),
        ('svr', SVR(kernel='rbf', C=10.0, gamma='scale', epsilon=0.1)),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_val)
    n_sv = pipe.named_steps['svr'].support_vectors_.shape[0]
    return y_pred, n_sv


# ============================================================ #
# B-4: GPR + PCA
# ============================================================ #
def train_gpr(X_train, y_train, X_val, y_val):
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=HP['PCA_N_COMPONENTS'])),
    ])
    X_tr_pca = pipe.fit_transform(X_train)
    X_va_pca = pipe.transform(X_val)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                    random_state=42, normalize_y=True)
    gpr.fit(X_tr_pca, y_train)
    y_pred, y_std = gpr.predict(X_va_pca, return_std=True)
    return y_pred, len(y_train)  # GPR stores all training points


# ============================================================ #
# B-5: MLP (1506D → 256 → 128 → 1)
# ============================================================ #
class BaselineMLP(nn.Module):
    def __init__(self, in_dim=3000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LeakyReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.LeakyReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.LeakyReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_train, y_train, X_val, y_val, seed):
    set_seed(seed)
    # 标准化
    scaler = StandardScaler()
    X_tr = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
    X_va = torch.tensor(scaler.transform(X_val), dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    y_va = torch.tensor(y_val, dtype=torch.float32)

    model = BaselineMLP(X_tr.shape[1]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=HP['LR'], weight_decay=HP['WEIGHT_DECAY'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=HP['EPOCHS'], eta_min=1e-5)
    ema = EMAModel(model, HP['EMA_DECAY'])
    es = EarlyStopping()
    bs = HP['BATCH_SIZE']

    for epoch in range(HP['EPOCHS']):
        model.train()
        idx = torch.randperm(len(X_tr))
        ep_loss, nb = 0.0, 0
        for s in range(0, len(idx), bs):
            bi = idx[s:s+bs]
            xb, yb = X_tr[bi].to(device), y_tr[bi].to(device)
            optimizer.zero_grad()
            loss = F.smooth_l1_loss(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); ema.update(model)
            ep_loss += loss.item(); nb += 1
        scheduler.step()

        ema.apply(model)
        model.eval()
        with torch.no_grad():
            pred_va = model(X_va.to(device)).cpu().numpy()
        mre_val = compute_metrics(pred_va, y_val)['MRE']
        ema.restore(model)
        es(mre_val, ema)
        if es.early_stop:
            break

    if es.best_ema_state is not None:
        model.load_state_dict(es.best_ema_state); model.to(device)
    model.eval()
    with torch.no_grad():
        y_pred = model(X_va.to(device)).cpu().numpy()
    return y_pred, n_params


# ============================================================ #
# B-6: PointNet (500×3 → shared MLP → max pool → FC)
# ============================================================ #
class BaselinePointNet(nn.Module):
    """PointNet: 置换不变点云回归.
    shared MLP per point → max pool → concat hotspot → FC head.
    """
    def __init__(self):
        super().__init__()
        # 逐点 MLP (shared weights)
        self.point_mlp = nn.Sequential(
            nn.Linear(6, 64), nn.LeakyReLU(), nn.BatchNorm1d(64),  # ★ 6D: [x,y,z,σ₁,τ_max,σ_vm]
            nn.Linear(64, 128), nn.LeakyReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 256), nn.LeakyReLU(), nn.BatchNorm1d(256),
        )
        # 回归头 (max-pooled 256D, 不含 hotspot)
        self.reg_head = nn.Sequential(
            nn.Linear(256, 128), nn.LeakyReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.LeakyReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, pts):
        """pts: (B, N, 6) = [x,y,z,σ₁,τ_max,σ_vm]."""
        B, N, _ = pts.shape
        x = pts.reshape(B * N, 6)
        x = self.point_mlp(x)           # (B*N, 256)
        x = x.reshape(B, N, 256)
        x = x.max(dim=1).values         # (B, 256)  max pool
        return self.reg_head(x).squeeze(-1)


def train_pointnet(train_raw, val_raw, seed):
    set_seed(seed)
    # ★ 与 V11 相同的分层增强
    aug = do_augmentation(train_raw)
    all_train = train_raw + aug

    model = BaselinePointNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=HP['LR'], weight_decay=HP['WEIGHT_DECAY'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=HP['EPOCHS'], eta_min=1e-5)
    ema = EMAModel(model, HP['EMA_DECAY'])
    es = EarlyStopping()
    bs = HP['BATCH_SIZE']

    for epoch in range(HP['EPOCHS']):
        model.train()
        idx = list(range(len(all_train))); random.shuffle(idx)
        ep_loss, nb = 0.0, 0
        for s in range(0, len(idx), bs):
            batch = [all_train[i] for i in idx[s:s+bs]]
            pts, yt = extract_pointnet_batch(batch)
            pts, yt = pts.to(device), yt.to(device)
            optimizer.zero_grad()
            loss = F.smooth_l1_loss(model(pts), yt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); ema.update(model)
            ep_loss += loss.item(); nb += 1
        scheduler.step()

        # 验证
        ema.apply(model)
        model.eval()
        with torch.no_grad():
            pts_v, yt_v = extract_pointnet_batch(val_raw)
            pred_v = model(pts_v.to(device)).cpu().numpy()
        mre_val = compute_metrics(pred_v, yt_v.numpy())['MRE']
        ema.restore(model)
        es(mre_val, ema)
        if es.early_stop:
            break

    if es.best_ema_state is not None:
        model.load_state_dict(es.best_ema_state); model.to(device)
    model.eval()
    with torch.no_grad():
        pts_v, yt_v = extract_pointnet_batch(val_raw)
        y_pred = model(pts_v.to(device)).cpu().numpy()
    return y_pred, n_params


# ============================================================ #
# 统一 train_fold
# ============================================================ #
def train_fold(baseline_name, train_raw, val_raw, seed, fold_i):
    set_seed(seed)
    t0 = time.time()
    sp_id = parse_id

    # ★ 与 V11 相同的分层增强 (文献惯例: 表格级增强统一给所有方法)
    aug = do_augmentation(train_raw)
    all_train = train_raw + aug
    X_train, y_train, _ = extract_tabular(all_train)
    X_val, y_val, val_ids = extract_tabular(val_raw)

    n_raw, n_aug = len(train_raw), len(aug)

    # 训练
    if baseline_name == 'B-1':
        y_pred, n_params = train_rf(X_train, y_train, X_val, y_val)
    elif baseline_name == 'B-2':
        if not HAS_XGB:
            return None, None, None
        y_pred, n_params = train_xgb(X_train, y_train, X_val, y_val)
    elif baseline_name == 'B-3':
        y_pred, n_params = train_svr(X_train, y_train, X_val, y_val)
    elif baseline_name == 'B-4':
        y_pred, n_params = train_gpr(X_train, y_train, X_val, y_val)
    elif baseline_name == 'B-5':
        y_pred, n_params = train_mlp(X_train, y_train, X_val, y_val, seed)
    elif baseline_name == 'B-6':
        y_pred, n_params = train_pointnet(train_raw, val_raw, seed)
        _, y_val, val_ids = extract_tabular(val_raw)  # 获取 y_val 和 ids
    else:
        raise ValueError(f'未知 baseline: {baseline_name}')

    time_s = time.time() - t0
    metrics = compute_metrics(y_pred, y_val)

    # 构建结果
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


# ============================================================ #
# 运行单个 baseline
# ============================================================ #
BASELINE_NAMES = {
    'B-1': 'Random Forest (3000D)',
    'B-2': 'XGBoost (3000D)',
    'B-3': 'SVR-RBF (PCA→20D)',
    'B-4': 'GPR (PCA→20D)',
    'B-5': 'MLP (3000D)',
    'B-6': 'PointNet (500×6)',
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


# ============================================================ #
# 主入口
# ============================================================ #
def main():
    ap = argparse.ArgumentParser(description='§4.3 基准方法对比')
    ap.add_argument('--cv_splits', default='./v11_final/cv_splits.json')
    ap.add_argument('--out_dir', default='./baseline_4_3')
    ap.add_argument('--baselines', default='B-1,B-2,B-3,B-4,B-5,B-6',
                    help='逗号分隔, 可选: B-1,B-2,B-3,B-4,B-5,B-6')
    args = ap.parse_args()

    print(f'Device: {device}')

    with open(AUG_PKL_PATH, 'rb') as f:
        all_data = pickle.load(f)
    failed = [d for d in all_data if not d.get('censored', False)]
    print(f'失效样本: {len(failed)}')

    with open(args.cv_splits, 'r', encoding='utf-8') as f:
        cv_splits = json.load(f)
    print(f'CV 切分: {len(cv_splits)} folds loaded')

    # 生成缺失的 seed 切分
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

    # 汇总对比表
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
        sum_path = os.path.join(args.out_dir, 'baseline_summary.csv')
        df_sum.to_csv(sum_path, index=False)

        print(f'\n{"="*80}')
        print(f'  §4.3 基准对比汇总 → {sum_path}')
        print(f'{"="*80}')
        print(f'  {"#":<5} {"Model":<25} {"MRE±std":>14} {"R²_logN":>8} '
              f'{"P_2X":>6} {"ens_MRE":>8} {"time":>6}')
        print('-' * 80)
        for _, r in df_sum.iterrows():
            print(f'  {r["baseline"]:<5} {r["description"]:<25} '
                  f'{r["fold_MRE_mean"]:.4f}±{r["fold_MRE_std"]:.4f} '
                  f'{r["fold_R2_logN"]:8.4f} {r["fold_P2X"]:6.3f} '
                  f'{r["ensemble_MRE"]:8.4f} {r["avg_time"]:5.0f}s')
        print(f'{"="*80}')

    print(f'\n完成! 输出: {args.out_dir}/')
    for bn in baseline_names:
        if bn in BASELINE_NAMES:
            print(f'  {bn}/  — results.csv, specimen_preds.csv, specimen_ensemble.csv, meta.json')
    if len(all_meta) > 1:
        print(f'  baseline_summary.csv — 全部 baseline 对比汇总表')


if __name__ == '__main__':
    main()