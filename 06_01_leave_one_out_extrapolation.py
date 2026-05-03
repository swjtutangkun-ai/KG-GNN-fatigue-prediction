# -*- coding: utf-8 -*-
"""
4_4a_loo.py — §4.4.1 留一试件外推 (LOO)

114 次留一, 每次跑满 300 epochs (无早停), 逐 epoch 记录预测.
构建 (300 × 114) 预测矩阵, 找全局最优 epoch t*.

t* 的意义: 用全部 114 个试件的 LOO 预测确定最优训练步数,
可作为模型部署时的固定 epoch 配置.

用法:
  python 4_4a_loo.py [--out_dir ./generalization_4_4]
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
    """训练 300 epochs, 逐 epoch 返回对 val_sp 的预测. 无早停."""
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

    # 逐 epoch 记录: pred_log10N, A, m
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

        # EMA 预测
        ema.apply(model)
        model.eval()
        with torch.no_grad():
            pred, A_now, m_now = model(val_data)
        preds_per_ep[epoch] = float(pred.cpu().numpy()[0])
        A_per_ep[epoch] = A_now
        m_per_ep[epoch] = m_now
        ema.restore(model)

    return preds_per_ep, A_per_ep, m_per_ep


# ==================== 主入口 ==================== #
def main():
    ap = argparse.ArgumentParser(description='§4.4.1 LOO 留一外推 + 最优 epoch 搜索')
    ap.add_argument('--out_dir', default='./generalization_4_4')
    args = ap.parse_args()

    loo_dir = os.path.join(args.out_dir, 'LOO')
    os.makedirs(loo_dir, exist_ok=True)

    print(f'Device: {device}')
    print(f'HP: H={HP["HIDDEN_DIM"]} heads={HP["HEADS"]} D={HP["DROPOUT"]} '
          f'LR={HP["LR"]} BS={HP["BATCH_SIZE"]} EPOCHS={HP["EPOCHS"]}')

    with open(AUG_PKL_PATH, 'rb') as f:
        all_data = pickle.load(f)
    failed = [d for d in all_data if not d.get('censored', False)]
    n = len(failed)
    n_ep = HP['EPOCHS']
    print(f'失效样本: {n}, 总迭代: {n} × {n_ep} = {n * n_ep} epochs')

    # 预分配矩阵
    pred_matrix = np.full((n_ep, n), np.nan)   # (epochs, specimens)
    A_matrix = np.full((n_ep, n), np.nan)
    m_matrix = np.full((n_ep, n), np.nan)
    true_log10N = np.zeros(n)
    spec_ids, spec_jt, spec_ct = [], [], []

    print(f'\n{"="*60}')
    print(f'  LOO: {n} 次 × {n_ep} epochs (无早停, 轨迹模式)')
    print(f'{"="*60}')
    t0_all = time.time()

    for i in range(n):
        t0 = time.time()
        sp = failed[i]
        jt, ct = parse_id(sp.get('ID', ''))
        spec_ids.append(sp.get('ID', ''))
        spec_jt.append(jt)
        spec_ct.append(int(ct))
        true_log10N[i] = sp['y'].item()

        train_raw = [failed[j] for j in range(n) if j != i]
        preds_ep, A_ep, m_ep = train_loo_trajectory(train_raw, sp, SEED)

        pred_matrix[:, i] = preds_ep
        A_matrix[:, i] = A_ep
        m_matrix[:, i] = m_ep
        ts = time.time() - t0

        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - t0_all
            eta = elapsed / (i + 1) * (n - i - 1)
            # 临时最优 epoch
            done = i + 1
            temp_mre = np.zeros(n_ep)
            tN_done = 10.0 ** true_log10N[:done]
            for ep in range(n_ep):
                pN = 10.0 ** pred_matrix[ep, :done]
                temp_mre[ep] = np.mean(np.abs(pN - tN_done) / (tN_done + 1e-8))
            temp_tstar = int(np.argmin(temp_mre) + 1)
            print(f'  [{done:3d}/{n}] {spec_ids[i]:20s} | '
                  f't*≈{temp_tstar} MRE≈{temp_mre[temp_tstar-1]:.4f} | '
                  f'{ts:.0f}s | ETA {eta/60:.0f}min')

        # 每 20 个试件增量保存
        if (i + 1) % 20 == 0:
            np.savez(os.path.join(loo_dir, 'checkpoint.npz'),
                     pred_matrix=pred_matrix, true_log10N=true_log10N,
                     A_matrix=A_matrix, m_matrix=m_matrix,
                     spec_ids=np.array(spec_ids), n_done=i+1)

    # ============================================================ #
    # 计算全局 MRE 曲线, 找 t*
    # ============================================================ #
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

    print(f'\n{"="*60}')
    print(f'  ★ 全局最优 epoch t* = {t_star}')
    print(f'    MRE  = {best_mre:.4f}')
    print(f'    R²   = {best_r2:.4f}')
    print(f'    P_2X = {best_p2x:.4f}')
    print(f'    A    = {best_A:.3f}')
    print(f'    m    = {best_m:.3f}')
    print(f'{"="*60}')

    # ============================================================ #
    # 导出
    # ============================================================ #
    # 1. MRE 曲线
    df_curve = pd.DataFrame({
        'epoch': np.arange(1, n_ep + 1),
        'MRE': mre_curve, 'R2_logN': r2_curve, 'P2X_cov': p2x_curve,
        'A_mean': np.mean(A_matrix, axis=1), 'm_mean': np.mean(m_matrix, axis=1),
    })
    df_curve.to_csv(os.path.join(loo_dir, 'loo_mre_curve.csv'), index=False)

    # 2. t* 处逐试件结果
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
            'A': float(A_matrix[t_star - 1, i]),
            'm': float(m_matrix[t_star - 1, i]),
        })
    df_results = pd.DataFrame(rows)
    df_results.to_csv(os.path.join(loo_dir, 'loo_results.csv'), index=False)

    # 3. 完整矩阵
    np.savez(os.path.join(loo_dir, 'pred_matrix.npz'),
             pred_matrix=pred_matrix, true_log10N=true_log10N,
             A_matrix=A_matrix, m_matrix=m_matrix,
             spec_ids=np.array(spec_ids),
             spec_jt=np.array(spec_jt),
             spec_ct=np.array(spec_ct))

    # 4. 分组统计
    print(f'\n  t*={t_star} 处分组:')
    for jt in JT_NAMES:
        sub = df_results[df_results['joint_type'] == jt]
        if len(sub) > 0:
            print(f'    {jt}: MRE={sub["rel_error"].mean():.4f}±{sub["rel_error"].std():.4f} (n={len(sub)})')

    # 5. meta
    total_min = (time.time() - t0_all) / 60
    meta = {
        'mode': 'LOO', 'n_specimens': n, 'n_epochs': n_ep, 'seed': SEED,
        'optimal_epoch_t_star': t_star,
        'MRE_at_t_star': best_mre,
        'R2_at_t_star': best_r2,
        'P2X_at_t_star': best_p2x,
        'A_at_t_star': best_A, 'm_at_t_star': best_m,
        'total_time_min': total_min,
    }
    with open(os.path.join(loo_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f'\n  总时间: {total_min:.1f} min')
    print(f'  输出: {loo_dir}/')
    print(f'    loo_results.csv      — t* 处 114 个试件预测')
    print(f'    loo_mre_curve.csv    — 逐 epoch MRE/R²/P2X 曲线')
    print(f'    pred_matrix.npz      — 完整 (300×114) 预测矩阵')
    print(f'    meta.json            — t* 及核心指标')


if __name__ == '__main__':
    main()
