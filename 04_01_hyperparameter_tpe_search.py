# -*- coding: utf-8 -*-
"""
3_train_v11_tpe_search.py — V11 贝叶斯超参数搜索 (无边特征)

★ 与 V10 TPE 唯一区别: GATv2Conv(edge_dim=None), 不使用边特征
★ 其余完全一致: 搜索空间, CV 协议, 固定 HP, 增强策略

★ 搜索空间 (5 维离散, 420 有效组合):
    HIDDEN_DIM:   [64, 128, 256]
    HEADS:        [2, 4, 8]
    DROPOUT:      [0.2, 0.4, 0.6, 0.8]
    LR:           [1e-4, 2e-4, 4e-4, 8e-4]
    WEIGHT_DECAY: [1e-5, 1e-4, 1e-3]

★ CV 协议: 1 seed (42) × 10 fold = 10 folds / trial
★ trial 数: N_TRIALS (默认 60)
★ 断点续跑: 直接重新运行即可
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = '0'

import json, random, pickle, time, copy, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.aggr import AttentionalAggregation
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)

from config_single import (
    FEAT_DIM, EDGE_DIM, HOTSPOT_DIM, AUG_PKL_PATH, RESULT_DIR,
    parse_id, USE_LOG10, FAT_INIT,
)

assert USE_LOG10 and FEAT_DIM == 3  # ★ V11: 不再 assert EDGE_DIM

# ============================================================ #
# 固定 HP (不搜索的部分)
# ============================================================ #
FIXED_HP = {
    'EPOCHS':           500,
    'PATIENCE':         50,
    'EMA_DECAY':        0.995,
    'ES_MIN_DELTA':     5e-4,
    'BASQUIN_M':        3.0,
    'LOG_DS_CENTER':    2.0,
    'LOG_DS_SCALE':     1.0,
    'AUG_LOW':          (2, 0.03),
    'AUG_MID':          (3, 0.05),
    'AUG_HIGH':         (6, 0.08),
}

SEARCH_SEED   = 11
N_SPLITS      = 10
N_TRIALS      = 60
TPE_STARTUP   = 20

JT_NAMES = ['DJ', 'TX', 'UL']
A_PER_JT = np.array([6.301 + FIXED_HP['BASQUIN_M'] * np.log10(FAT_INIT[k])
                      for k in JT_NAMES], dtype=np.float32)
A_INIT = float(A_PER_JT.mean())
M_INIT = FIXED_HP['BASQUIN_M']

OUT_DIR = './tpe_search_v11'  # ★ V11: 输出目录
os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================ #
# 搜索空间
# ============================================================ #
SEARCH_SPACE = {
    'HIDDEN_DIM':   [64, 128, 256],
    'HEADS':        [2, 4, 8],
    'DROPOUT':      [0.2, 0.4, 0.6, 0.8],
    'LR':           [1e-4, 2e-4, 4e-4, 8e-4],
    'WEIGHT_DECAY': [1e-5, 1e-4, 1e-3],
    'BATCH_SIZE':   [4, 8, 16],
}

N_TOTAL = 1
for v in SEARCH_SPACE.values():
    N_TOTAL *= len(v)
# 排除 HIDDEN_DIM=256 × HEADS=8
N_EXCLUDED = (len(SEARCH_SPACE['DROPOUT']) * len(SEARCH_SPACE['LR']) *
              len(SEARCH_SPACE['WEIGHT_DECAY']) * len(SEARCH_SPACE['BATCH_SIZE']))
N_VALID = N_TOTAL - N_EXCLUDED
print(f'搜索空间: 全组合={N_TOTAL}, 排除 256×8={N_EXCLUDED}, 有效={N_VALID}')


# ==================== 通用工具 ==================== #
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
        self.patience = FIXED_HP['PATIENCE']
        self.counter = 0
        self.best_score, self.early_stop = None, False
        self.best_ema_state = None
    def __call__(self, mre, ema):
        s = -mre
        if self.best_score is None or s > self.best_score + FIXED_HP['ES_MIN_DELTA']:
            self.best_score = s; self.best_ema_state = ema.state_dict(); self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


def augment(data, noise_std):
    d = copy.deepcopy(data)
    d['x'] = d['x'].clone() + torch.randn_like(d['x']) * noise_std
    # ★ V11: 不再增强 edge_attr
    return d


def compute_life_bins(failed_data):
    y_all = np.array([d['y'].item() for d in failed_data])
    q33, q67 = np.percentile(y_all, [33, 67])
    return [(0.0, q33, *FIXED_HP['AUG_LOW']),
            (q33, q67, *FIXED_HP['AUG_MID']),
            (q67, 999.0, *FIXED_HP['AUG_HIGH'])]


def get_aug_params(y_val, bins):
    for lo, hi, n, ns in bins:
        if lo <= y_val < hi:
            return n, ns
    raise ValueError


def preprocess(sp):
    # ★ V11: 不传 edge_attr
    g = Data(x=sp['x'], edge_index=sp['edge_index'])
    return {'graph': g, 'y': sp['y'], 'ID': sp.get('ID', ''),
            'hotspot_desc': sp.get('hotspot_desc', torch.zeros(HOTSPOT_DIM))}


# ==================== 模型 (HP 由外部注入) ==================== #
class PhysicsGNN(nn.Module):
    def __init__(self, hp):
        super().__init__()
        h = hp['HIDDEN_DIM']
        self.drop = nn.Dropout(hp['DROPOUT'])
        # ★ V11: edge_dim=None
        self.conv1 = GATv2Conv(FEAT_DIM, h, heads=hp['HEADS'], edge_dim=None)
        self.bn1 = nn.BatchNorm1d(h * hp['HEADS'])
        self.conv2 = GATv2Conv(h * hp['HEADS'], h, heads=1, edge_dim=None)
        self.bn2 = nn.BatchNorm1d(h)
        self.pool = AttentionalAggregation(
            nn.Sequential(nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1)))
        in_dim = h + HOTSPOT_DIM
        self.sigma_head = nn.Sequential(
            nn.Linear(in_dim, h), nn.LeakyReLU(), nn.Dropout(hp['DROPOUT']),
            nn.Linear(h, h // 2), nn.LeakyReLU(), nn.Dropout(hp['DROPOUT']),
            nn.Linear(h // 2, 1))
        self.A_g = nn.Parameter(torch.tensor(A_INIT, dtype=torch.float32))
        self.m_g = nn.Parameter(torch.tensor(M_INIT, dtype=torch.float32))

    def forward(self, batch_data):
        graphs = [sp['graph'] for sp in batch_data]
        batch = Batch.from_data_list(graphs).to(device)
        # ★ V11: 不传 edge_attr
        x = self.drop(F.leaky_relu(self.bn1(self.conv1(batch.x, batch.edge_index))))
        x = self.drop(F.leaky_relu(self.bn2(self.conv2(x, batch.edge_index))))
        z = self.pool(x, batch.batch)
        hs = torch.stack([sp['hotspot_desc'] for sp in batch_data]).to(device)
        zh = torch.cat([z, hs], dim=1)
        log_DS = FIXED_HP['LOG_DS_CENTER'] + FIXED_HP['LOG_DS_SCALE'] * torch.tanh(
            self.sigma_head(zh).squeeze(-1))
        DS = torch.pow(10.0, log_DS)
        preds = self.A_g.expand(len(batch_data)) - self.m_g * log_DS
        return preds, {'A_g': self.A_g.detach().clone(),
                       'm_g': self.m_g.detach().clone(),
                       'DS_per': DS.detach()}


# ==================== 评估 ==================== #
def evaluate(model, val_data, bs):
    model.eval()
    vp, vt = [], []
    A_snap, m_snap = None, None
    with torch.no_grad():
        for s in range(0, len(val_data), bs):
            batch = val_data[s:s+bs]
            preds, info = model(batch)
            vp.extend(preds.cpu().numpy())
            vt.extend([sp['y'].item() for sp in batch])
            if A_snap is None:
                A_snap = float(info['A_g'].cpu().numpy())
                m_snap = float(info['m_g'].cpu().numpy())
    vp, vt = np.array(vp), np.array(vt)
    logmae = float(np.mean(np.abs(vp - vt)))
    pN, tN = 10.0 ** vp, 10.0 ** vt
    mre = float(np.mean(np.abs(pN - tN) / (tN + 1e-8)))
    ss_r = np.sum((vt - vp) ** 2); ss_t = np.sum((vt - vt.mean()) ** 2)
    r2_log = float(1 - ss_r / ss_t) if ss_t > 0 else 0.0
    return mre, logmae, r2_log, A_snap, m_snap


# ==================== 训练单 fold ==================== #
def train_fold(hp, train_raw, val_raw, seed, fold_i):
    set_seed(seed)
    bins = compute_life_bins(train_raw)
    train_aug = []
    for d in train_raw:
        n_aug, noise = get_aug_params(d['y'].item(), bins)
        for _ in range(n_aug):
            train_aug.append(augment(d, noise_std=noise))
    train_data = [preprocess(d) for d in train_raw + train_aug]
    val_data = [preprocess(d) for d in val_raw]

    model = PhysicsGNN(hp).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=hp['LR'],
                                 weight_decay=hp['WEIGHT_DECAY'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FIXED_HP['EPOCHS'], eta_min=1e-5)
    es = EarlyStopping()
    ema = EMAModel(model, FIXED_HP['EMA_DECAY'])
    bs = hp['BATCH_SIZE']
    best_ep = 0

    for epoch in range(FIXED_HP['EPOCHS']):
        model.train()
        idx = list(range(len(train_data))); random.shuffle(idx)
        ep_loss, nb = 0.0, 0
        for s in range(0, len(idx), bs):
            batch = [train_data[i] for i in idx[s:s+bs]]
            tgt = torch.tensor([sp['y'].item() for sp in batch], device=device)
            optimizer.zero_grad()
            preds, _ = model(batch)
            loss = F.smooth_l1_loss(preds, tgt, reduction='mean')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); ema.update(model)
            ep_loss += loss.item(); nb += 1
        scheduler.step()

        ema.apply(model)
        mre, _, _, _, _ = evaluate(model, val_data, bs)
        ema.restore(model)
        prev = es.best_score
        es(mre, ema)
        if es.best_score != prev:
            best_ep = epoch + 1
        if es.early_stop:
            break

    if es.best_ema_state is not None:
        model.load_state_dict(es.best_ema_state); model.to(device)
    mre, logmae, r2_log, A_final, m_final = evaluate(model, val_data, bs)
    return mre, logmae, r2_log, A_final, m_final, best_ep


# ==================== Optuna objective ==================== #
print(f'Device: {device}  Output: {OUT_DIR}')
print(f'V11 TPE Search (edge_dim=None)')
print(f'加载数据...')
with open(AUG_PKL_PATH, 'rb') as f:
    ALL_DATA = pickle.load(f)
FAILED = [d for d in ALL_DATA if not d.get('censored', False)]
print(f'失效样本: {len(FAILED)}')

Y_ALL = np.array([d['y'].item() for d in FAILED])
Y_BINS = pd.qcut(Y_ALL, q=N_SPLITS, labels=False, duplicates='drop')
SKF = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEARCH_SEED)
CV_SPLITS = list(SKF.split(np.zeros(len(FAILED)), Y_BINS))

fold_detail_rows = []


def objective(trial):
    hidden_dim = trial.suggest_categorical('HIDDEN_DIM', SEARCH_SPACE['HIDDEN_DIM'])
    heads      = trial.suggest_categorical('HEADS',      SEARCH_SPACE['HEADS'])
    dropout    = trial.suggest_categorical('DROPOUT',    SEARCH_SPACE['DROPOUT'])
    lr         = trial.suggest_categorical('LR',         SEARCH_SPACE['LR'])
    wd         = trial.suggest_categorical('WEIGHT_DECAY', SEARCH_SPACE['WEIGHT_DECAY'])
    batch_size = trial.suggest_categorical('BATCH_SIZE', SEARCH_SPACE['BATCH_SIZE'])

    if hidden_dim == 256 and heads == 8:
        return 999.0

    hp = {
        'HIDDEN_DIM': hidden_dim, 'HEADS': heads,
        'DROPOUT': dropout, 'LR': lr, 'WEIGHT_DECAY': wd,
        'BATCH_SIZE': batch_size,
    }
    trial_t0 = time.time()

    mres, logmaes, r2s, As, ms, eps = [], [], [], [], [], []
    for fold_i, (ti, vi) in enumerate(CV_SPLITS):
        tf = [FAILED[i] for i in ti]; vf = [FAILED[i] for i in vi]
        try:
            mre, logmae, r2_log, A, m, best_ep = train_fold(hp, tf, vf, SEARCH_SEED, fold_i)
        except Exception as e:
            print(f'    [ERROR] fold {fold_i}: {e}')
            return 999.0

        mres.append(mre); logmaes.append(logmae); r2s.append(r2_log)
        As.append(A); ms.append(m); eps.append(best_ep)

        fold_detail_rows.append({
            'trial': trial.number, **hp,
            'fold': fold_i, 'MRE': mre, 'logMAE': logmae,
            'R2_logN': r2_log, 'A': A, 'm': m, 'best_ep': best_ep,
        })

        trial.report(np.mean(mres), fold_i)

    trial_time = time.time() - trial_t0
    mean_mre = float(np.mean(mres))
    std_mre = float(np.std(mres))

    trial.set_user_attr('MRE_std', std_mre)
    trial.set_user_attr('logMAE_mean', float(np.mean(logmaes)))
    trial.set_user_attr('R2_logN_mean', float(np.mean(r2s)))
    trial.set_user_attr('A_mean', float(np.mean(As)))
    trial.set_user_attr('A_std', float(np.std(As)))
    trial.set_user_attr('m_mean', float(np.mean(ms)))
    trial.set_user_attr('m_std', float(np.std(ms)))
    trial.set_user_attr('best_ep_mean', float(np.mean(eps)))
    trial.set_user_attr('time_s', trial_time)

    print(f'  Trial {trial.number:3d}/{N_TRIALS} | '
          f'H={hidden_dim} heads={heads} D={dropout} '
          f'LR={lr:.0e} WD={wd:.0e} BS={batch_size} | '
          f'MRE={mean_mre:.4f}±{std_mre:.4f} '
          f'R²={np.mean(r2s):.3f} '
          f'A={np.mean(As):.3f}±{np.std(As):.3f} '
          f'm={np.mean(ms):.3f}±{np.std(ms):.3f} | '
          f'{trial_time:.0f}s')

    return mean_mre


# ==================== 主程序 ==================== #
if __name__ == '__main__':
    db_path = os.path.join(OUT_DIR, 'optuna_tpe.db')
    study = optuna.create_study(
        study_name='v11_tpe_search',
        storage=f'sqlite:///{db_path}',
        direction='minimize',
        sampler=TPESampler(
            seed=SEARCH_SEED,
            n_startup_trials=TPE_STARTUP,
            multivariate=True,
        ),
        load_if_exists=True,
    )

    n_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_todo = max(0, N_TRIALS - n_done)
    print(f'\n已完成: {n_done} / {N_TRIALS}  (TPE startup={TPE_STARTUP})')
    print(f'待跑:   {n_todo} trials × {N_SPLITS} folds')
    print(f'预计:   ~{n_todo * 10 * 30 / 3600:.1f} h\n')

    if n_todo > 0:
        study.optimize(objective, n_trials=n_todo, show_progress_bar=False)
    else:
        print('已达 N_TRIALS, 无需续跑. 直接汇总.\n')

    # ==================== 汇总结果 ==================== #
    print('\n' + '=' * 80)
    print('V11 TPE Search 完成 (edge_dim=None)')
    print('=' * 80)

    valid_trials = [t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE and t.value < 100]

    summary_rows = []
    for t in valid_trials:
        row = {
            'trial': t.number,
            'HIDDEN_DIM': t.params['HIDDEN_DIM'],
            'HEADS': t.params['HEADS'],
            'DROPOUT': t.params['DROPOUT'],
            'LR': t.params['LR'],
            'WEIGHT_DECAY': t.params['WEIGHT_DECAY'],
            'MRE_mean': t.value,
        }
        row.update(t.user_attrs)
        summary_rows.append(row)
    df_summary = pd.DataFrame(summary_rows).sort_values('MRE_mean')
    df_summary.to_csv(os.path.join(OUT_DIR, 'tpe_results.csv'), index=False)

    df_folds = pd.DataFrame(fold_detail_rows)
    df_folds.to_csv(os.path.join(OUT_DIR, 'fold_details.csv'), index=False)

    print('\nTop 10 配置:')
    print(f'{"#":>3} {"H":>4} {"Hd":>3} {"D":>4} {"LR":>8} {"WD":>8} '
          f'{"MRE":>8} {"±":>6} {"R²":>6} {"A":>7} {"m":>6} {"ep":>5}')
    print('-' * 80)
    for i, row in df_summary.head(10).iterrows():
        print(f'{row["trial"]:3.0f} '
              f'{row["HIDDEN_DIM"]:4.0f} '
              f'{row["HEADS"]:3.0f} '
              f'{row["DROPOUT"]:4.1f} '
              f'{row["LR"]:8.1e} '
              f'{row["WEIGHT_DECAY"]:8.1e} '
              f'{row["MRE_mean"]:8.4f} '
              f'{row.get("MRE_std", 0):6.4f} '
              f'{row.get("R2_logN_mean", 0):6.3f} '
              f'{row.get("A_mean", 0):7.3f} '
              f'{row.get("m_mean", 0):6.3f} '
              f'{row.get("best_ep_mean", 0):5.0f}')

    best = study.best_trial
    print(f'\n★ Best Trial {best.number}: MRE = {best.value:.4f}')
    print(f'  {best.params}')

    with open(os.path.join(OUT_DIR, 'tpe_meta.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'sampler': 'TPESampler',
            'architecture': 'V11: 2×GATv2(edge_dim=None) + AttentionalAggregation + Basquin',
            'n_trials_target': N_TRIALS,
            'n_startup_trials': TPE_STARTUP,
            'search_space': {k: [float(x) if isinstance(x, float) else x
                                  for x in v] for k, v in SEARCH_SPACE.items()},
            'fixed_hp': {k: (list(v) if isinstance(v, tuple) else v)
                         for k, v in FIXED_HP.items()},
            'search_seed': SEARCH_SEED,
            'n_splits': N_SPLITS,
            'n_valid_configs_in_space': N_VALID,
            'n_completed': len(valid_trials),
            'best_trial': best.number,
            'best_mre': best.value,
            'best_params': best.params,
            'A_INIT': A_INIT,
            'M_INIT': M_INIT,
        }, f, indent=2, ensure_ascii=False)

    print(f'\n输出: {OUT_DIR}/')
    print(f'  tpe_results.csv   — {len(valid_trials)} 配置汇总 (按 MRE 排序)')
    print(f'  fold_details.csv  — {len(fold_detail_rows)} 逐 fold 详情')
    print(f'  optuna_tpe.db     — Optuna study (可重跑/分析)')
    print(f'  tpe_meta.json     — 搜索空间与最佳配置')