# -*- coding: utf-8 -*-
"""
4_4b_sensitivity.py — §4.4 数据量敏感性 (缩减训练集的 LOO + t*)

与 4_4a_loo.py 完全相同的 114 次 LOO 外推,
区别: 每次从剩余 113 个试件中按 JT×CT 分层采样 40%/60%/80% 作为训练集.
测试集不变 (始终是留出的那 1 个), 可与 100% LOO 直接对比.

3 比例, 每比例 114 次 × 300 epochs

用法:
  python 4_4b_sensitivity.py [--out_dir ./generalization_4_4]
                              [--fractions 0.4,0.6,0.8]
"""
import os, json, random, pickle, time, copy, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.aggr import AttentionalAggregation
import pandas as pd

from config_single import (
    FEAT_DIM, HOTSPOT_DIM, AUG_PKL_PATH,
    parse_id, USE_LOG10, FAT_INIT,
)

assert USE_LOG10 and FEAT_DIM == 3

HP = {
    'HIDDEN_DIM': 64, 'HEADS': 4, 'DROPOUT': 0.6,
    'LR': 4e-4, 'WEIGHT_DECAY': 1e-4,
    'EPOCHS': 300, 'BATCH_SIZE': 4,
    'EMA_DECAY': 0.995,
    'BASQUIN_M': 3.0, 'LOG_DS_CENTER': 2.0, 'LOG_DS_SCALE': 1.0,
    'AUG_LOW': (2, 0.03), 'AUG_MID': (3, 0.05), 'AUG_HIGH': (6, 0.08),
}
SEED = 42
JT_NAMES = ['DJ', 'TX', 'UL']
A_PER_JT = np.array([6.301 + HP['BASQUIN_M'] * np.log10(FAT_INIT[k])
                      for k in JT_NAMES], dtype=np.float32)
A_INIT = float(A_PER_JT.mean())
M_INIT = HP['BASQUIN_M']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== 工具 ==================== #
class EMAModel:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
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


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


def augment(data, noise_std):
    d = copy.deepcopy(data)
    d['x'] = d['x'].clone() + torch.randn_like(d['x']) * noise_std
    return d


def preprocess(sp):
    g = Data(x=sp['x'], edge_index=sp['edge_index'])
    sp_id = sp.get('ID', '')
    jt, k_hours = parse_id(sp_id)
    return {'graph': g, 'y': sp['y'], 'ID': sp_id,
            'joint_type': jt, 'corrosion_hours': int(k_hours),
            'hotspot_desc': sp.get('hotspot_desc', torch.zeros(HOTSPOT_DIM))}


def compute_life_bins(failed_data):
    y_all = np.array([d['y'].item() for d in failed_data])
    if len(y_all) < 3:
        return [(0.0, 999.0, 3, 0.05)]
    q33, q67 = np.percentile(y_all, [33, 67])
    return [(0.0, q33, *HP['AUG_LOW']), (q33, q67, *HP['AUG_MID']),
            (q67, 999.0, *HP['AUG_HIGH'])]


def get_aug_params(y_val, bins):
    for lo, hi, n, ns in bins:
        if lo <= y_val < hi: return n, ns
    return bins[-1][2], bins[-1][3]


def do_augmentation(train_raw):
    bins = compute_life_bins(train_raw)
    aug = []
    for d in train_raw:
        n_aug, noise = get_aug_params(d['y'].item(), bins)
        for _ in range(n_aug):
            aug.append(augment(d, noise))
    return aug


# ==================== 分层子采样 ==================== #
def stratified_subsample(specimens, fraction, rng_seed):
    """从 specimens 按 JT×CT 等比例抽取, 每组至少 1 个."""
    rng = np.random.RandomState(rng_seed)
    groups = {}
    for i, sp in enumerate(specimens):
        jt, ct = parse_id(sp.get('ID', ''))
        key = f'{jt}_{int(ct)}'
        if key not in groups:
            groups[key] = []
        groups[key].append(i)

    selected = []
    for key, indices in groups.items():
        n_select = max(1, int(len(indices) * fraction + 0.5))
        chosen = rng.choice(indices, size=min(n_select, len(indices)), replace=False)
        selected.extend(chosen.tolist())

    return sorted(set(selected))


# ==================== 模型 ==================== #
class PhysicsGNN(nn.Module):
    def __init__(self):
        super().__init__()
        h = HP['HIDDEN_DIM']
        self.drop = nn.Dropout(HP['DROPOUT'])
        self.conv1 = GATv2Conv(FEAT_DIM, h, heads=HP['HEADS'], edge_dim=None)
        self.bn1 = nn.BatchNorm1d(h * HP['HEADS'])
        self.conv2 = GATv2Conv(h * HP['HEADS'], h, heads=1, edge_dim=None)
        self.bn2 = nn.BatchNorm1d(h)
        self.pool = AttentionalAggregation(
            nn.Sequential(nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1)))
        in_dim = h + HOTSPOT_DIM
        self.sigma_head = nn.Sequential(
            nn.Linear(in_dim, h), nn.LeakyReLU(), nn.Dropout(HP['DROPOUT']),
            nn.Linear(h, h // 2), nn.LeakyReLU(), nn.Dropout(HP['DROPOUT']),
            nn.Linear(h // 2, 1))
        self.A_g = nn.Parameter(torch.tensor(A_INIT))
        self.m_g = nn.Parameter(torch.tensor(M_INIT))

    def forward(self, batch_data):
        graphs = [sp['graph'] for sp in batch_data]
        batch = Batch.from_data_list(graphs).to(device)
        x = self.drop(F.leaky_relu(self.bn1(self.conv1(batch.x, batch.edge_index))))
        x = self.drop(F.leaky_relu(self.bn2(self.conv2(x, batch.edge_index))))
        z = self.pool(x, batch.batch)
        hs = torch.stack([sp['hotspot_desc'] for sp in batch_data]).to(device)
        zh = torch.cat([z, hs], dim=1)
        log_DS = HP['LOG_DS_CENTER'] + HP['LOG_DS_SCALE'] * torch.tanh(
            self.sigma_head(zh).squeeze(-1))
        preds = self.A_g - self.m_g * log_DS
        return preds, self.A_g.detach().item(), self.m_g.detach().item()


# ==================== LOO 单次训练 (轨迹模式) ==================== #
def train_loo_trajectory(train_raw, val_sp, seed):
    set_seed(seed)
    aug = do_augmentation(train_raw)
    train_data = [preprocess(d) for d in train_raw + aug]
    val_data = [preprocess(val_sp)]

    model = PhysicsGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=HP['LR'], weight_decay=HP['WEIGHT_DECAY'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=HP['EPOCHS'], eta_min=1e-5)
    ema = EMAModel(model, HP['EMA_DECAY'])
    bs = HP['BATCH_SIZE']
    n_ep = HP['EPOCHS']

    preds_per_ep = np.zeros(n_ep)
    A_per_ep = np.zeros(n_ep)
    m_per_ep = np.zeros(n_ep)

    for epoch in range(n_ep):
        model.train()
        idx = list(range(len(train_data))); random.shuffle(idx)
        for s in range(0, len(idx), bs):
            batch = [train_data[i] for i in idx[s:s + bs]]
            tgt = torch.tensor([sp['y'].item() for sp in batch], device=device)
            optimizer.zero_grad()
            preds, _, _ = model(batch)
            loss = F.smooth_l1_loss(preds, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); ema.update(model)
        scheduler.step()

        ema.apply(model)
        model.eval()
        with torch.no_grad():
            pred, A_now, m_now = model(val_data)
        preds_per_ep[epoch] = float(pred.cpu().numpy()[0])
        A_per_ep[epoch] = A_now
        m_per_ep[epoch] = m_now
        ema.restore(model)

    return preds_per_ep, A_per_ep, m_per_ep


# ==================== 单比例 LOO ==================== #
def run_fraction_loo(failed, fraction, out_dir):
    n = len(failed)
    n_ep = HP['EPOCHS']
    label = f'{int(fraction*100)}pct'
    sub_dir = os.path.join(out_dir, f'LOO_{label}')
    os.makedirs(sub_dir, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'  LOO {label}: 114 次, 每次从 113 中取 {fraction:.0%} 训练')
    print(f'{"="*60}')

    pred_matrix = np.full((n_ep, n), np.nan)
    A_matrix = np.full((n_ep, n), np.nan)
    m_matrix = np.full((n_ep, n), np.nan)
    true_log10N = np.zeros(n)
    spec_ids, spec_jt, spec_ct = [], [], []
    n_train_actual = []

    t0_all = time.time()
    for i in range(n):
        t0 = time.time()
        sp = failed[i]
        jt, ct = parse_id(sp.get('ID', ''))
        spec_ids.append(sp.get('ID', ''))
        spec_jt.append(jt)
        spec_ct.append(int(ct))
        true_log10N[i] = sp['y'].item()

        # 剩余 113 个
        remaining = [failed[j] for j in range(n) if j != i]

        # ★ 分层子采样: 用 SEED + i 保证每个 LOO 迭代的采样独立但可复现
        sel_idx = stratified_subsample(remaining, fraction, SEED + i)
        train_raw = [remaining[j] for j in sel_idx]
        n_train_actual.append(len(train_raw))

        preds_ep, A_ep, m_ep = train_loo_trajectory(train_raw, sp, SEED)
        pred_matrix[:, i] = preds_ep
        A_matrix[:, i] = A_ep
        m_matrix[:, i] = m_ep

        ts = time.time() - t0
        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - t0_all
            eta = elapsed / (i + 1) * (n - i - 1)
            done = i + 1
            tN_done = 10.0 ** true_log10N[:done]
            temp_mre = np.zeros(n_ep)
            for ep in range(n_ep):
                pN = 10.0 ** pred_matrix[ep, :done]
                temp_mre[ep] = np.mean(np.abs(pN - tN_done) / (tN_done + 1e-8))
            temp_tstar = int(np.argmin(temp_mre) + 1)
            print(f'    [{done:3d}/114] n_train={len(train_raw)} | '
                  f't*≈{temp_tstar} MRE≈{temp_mre[temp_tstar-1]:.4f} | '
                  f'{ts:.0f}s | ETA {eta/60:.0f}min')

        if (i + 1) % 20 == 0:
            np.savez(os.path.join(sub_dir, 'checkpoint.npz'),
                     pred_matrix=pred_matrix, true_log10N=true_log10N, n_done=i+1)

    # MRE 曲线 + t*
    true_N = 10.0 ** true_log10N
    mre_curve = np.zeros(n_ep)
    r2_curve = np.zeros(n_ep)
    p2x_curve = np.zeros(n_ep)

    for ep in range(n_ep):
        pN = 10.0 ** pred_matrix[ep]
        mre_curve[ep] = np.mean(np.abs(pN - true_N) / (true_N + 1e-8))
        ss_r = np.sum((pred_matrix[ep] - true_log10N) ** 2)
        ss_t = np.sum((true_log10N - true_log10N.mean()) ** 2)
        r2_curve[ep] = 1 - ss_r / ss_t if ss_t > 0 else 0.0
        ratio = pN / (true_N + 1e-8)
        p2x_curve[ep] = np.mean((ratio >= 0.5) & (ratio <= 2.0))

    t_star = int(np.argmin(mre_curve) + 1)
    best_mre = float(mre_curve[t_star - 1])
    best_r2 = float(r2_curve[t_star - 1])
    best_p2x = float(p2x_curve[t_star - 1])
    best_A = float(np.mean(A_matrix[t_star - 1]))
    best_m = float(np.mean(m_matrix[t_star - 1]))
    avg_ntrain = int(np.mean(n_train_actual))

    print(f'\n    ★ {label}: t*={t_star}  MRE={best_mre:.4f}  R²={best_r2:.4f}  '
          f'P2X={best_p2x:.3f}  平均训练={avg_ntrain}')

    # 导出
    df_curve = pd.DataFrame({
        'epoch': np.arange(1, n_ep + 1),
        'MRE': mre_curve, 'R2_logN': r2_curve, 'P2X_cov': p2x_curve,
    })
    df_curve.to_csv(os.path.join(sub_dir, 'mre_curve.csv'), index=False)

    best_preds = pred_matrix[t_star - 1]
    rows = []
    for i in range(n):
        pred_N = float(10.0 ** best_preds[i])
        tN = float(true_N[i])
        rows.append({
            'ID': spec_ids[i], 'joint_type': spec_jt[i],
            'corrosion_hours': spec_ct[i],
            'true_log10N': float(true_log10N[i]),
            'pred_log10N': float(best_preds[i]),
            'true_N': tN, 'pred_N': pred_N,
            'rel_error': abs(pred_N - tN) / (tN + 1e-8),
            'n_train': n_train_actual[i],
        })
    pd.DataFrame(rows).to_csv(os.path.join(sub_dir, 'loo_results.csv'), index=False)

    np.savez(os.path.join(sub_dir, 'pred_matrix.npz'),
             pred_matrix=pred_matrix, true_log10N=true_log10N,
             spec_ids=np.array(spec_ids))

    total_min = (time.time() - t0_all) / 60
    meta = {
        'fraction': label, 'n_loo': n, 'avg_n_train': avg_ntrain,
        'n_epochs': n_ep, 'seed': SEED,
        't_star': t_star,
        'MRE_at_t_star': best_mre, 'R2_at_t_star': best_r2,
        'P2X_at_t_star': best_p2x,
        'A_at_t_star': best_A, 'm_at_t_star': best_m,
        'total_time_min': total_min,
    }
    with open(os.path.join(sub_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return meta


# ==================== 主入口 ==================== #
def main():
    ap = argparse.ArgumentParser(description='§4.4 数据量敏感性 (缩减训练集 LOO)')
    ap.add_argument('--out_dir', default='./generalization_4_4')
    ap.add_argument('--fractions', default='0.4,0.6,0.8',
                    help='逗号分隔')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fractions = [float(f) for f in args.fractions.split(',')]

    print(f'Device: {device}')
    print(f'EPOCHS: {HP["EPOCHS"]}, 比例: {fractions}')

    with open(AUG_PKL_PATH, 'rb') as f:
        all_data = pickle.load(f)
    failed = [d for d in all_data if not d.get('censored', False)]
    print(f'失效样本: {len(failed)}')

    all_meta = []
    for frac in fractions:
        meta = run_fraction_loo(failed, frac, args.out_dir)
        all_meta.append(meta)

    # ★ 合并已有结果
    summary_path = os.path.join(args.out_dir, 'sensitivity_summary.csv')
    if os.path.exists(summary_path):
        df_existing = pd.read_csv(summary_path)
        # 去掉本次跑的比例, 保留其余
        run_fracs = [m['fraction'] for m in all_meta]
        df_existing = df_existing[~df_existing['fraction'].isin(run_fracs)]
        df_all = pd.concat([df_existing, pd.DataFrame(all_meta)], ignore_index=True)
    else:
        df_all = pd.DataFrame(all_meta)
    df_all = df_all.sort_values('avg_n_train').reset_index(drop=True)
    df_all.to_csv(summary_path, index=False)

    # 汇总打印 (全部比例)
    print(f'\n{"="*60}')
    print(f'  数据量敏感性汇总 (与 100% LOO 对比)')
    print(f'{"="*60}')
    print(f'  {"比例":>6} {"n_train":>8} {"t*":>4} {"MRE":>8} {"R²":>7} {"P2X":>6}')
    print('-' * 50)
    for _, m in df_all.iterrows():
        print(f'  {m["fraction"]:>6} {int(m["avg_n_train"]):8d} {int(m["t_star"]):4d} '
              f'{m["MRE_at_t_star"]*100:7.2f}% {m["R2_at_t_star"]:.4f} '
              f'{m["P2X_at_t_star"]:.3f}')
    print(f'  {"100%":>6} {"113":>8} {"(见LOO/)":>4}')


if __name__ == '__main__':
    main()