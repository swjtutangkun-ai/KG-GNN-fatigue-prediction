# -*- coding: utf-8 -*-
"""
BW_3_train_v11.py — V11 全量训练 + BW 14 根外部泛化测试

策略:
  训练集: DJ/TX/UL 失效试件 114 根, 全量训练 (无 CV, 无早停)
  训练轮数: T_STAR = 69 (来自论文 §4.4 LOO 全局最优 epoch)
  集成: 5 seed, 跨 seed 对 BW 预测做 log10 空间平均
  测试集: BW 14 根 (每根已按实验观察的 KL 侧构图, 由 BW_1 产生)

这一协议与论文 §4.4 LOO 完全对等: 都是 "~114 训 + 外部试件推" 的设置.
因此 BW MRE 可直接与 LOO MRE=18.66% 对比.

输出:
  ./v11_bw_final/
    bw_preds.csv        5 seeds × 14 试件 = 70 行
    bw_ensemble.csv     14 行 (跨 seed 集成)
    seed_summary.csv    每 seed 的 A, m, BW MRE, 用时
    meta.json           HP + 汇总指标
"""
import os
import json, random, pickle, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.aggr import AttentionalAggregation
import pandas as pd

from config_single import (
    FEAT_DIM, EDGE_DIM, HOTSPOT_DIM, AUG_PKL_PATH,
    parse_id, USE_LOG10, FAT_INIT,
)

# ============================================================ #
HP = {
    'HIDDEN_DIM':       64,
    'HEADS':            4,
    'DROPOUT':          0.6,
    'LR':               4e-4,
    'WEIGHT_DECAY':     1e-4,
    'BATCH_SIZE':       4,

    # ★ 固定 epoch: 来自论文 §4.4 LOO 全局最优 (MRE=18.66%)
    'T_STAR':           69,

    'EMA_DECAY':        0.995,
    'RANDOM_SEEDS':     [11, 22, 33, 44, 55],

    'BASQUIN_M':        3.0,
    'LOG_DS_CENTER':    2.0,
    'LOG_DS_SCALE':     1.0,

    'AUG_LOW':          (2, 0.03),
    'AUG_MID':          (3, 0.05),
    'AUG_HIGH':         (6, 0.08),
}

assert USE_LOG10 and FEAT_DIM == 3

JT_NAMES = ['DJ', 'TX', 'UL']
A_PER_JT = np.array([6.301 + HP['BASQUIN_M'] * np.log10(FAT_INIT[k]) for k in JT_NAMES],
                    dtype=np.float32)
A_INIT = float(A_PER_JT.mean())
M_INIT = HP['BASQUIN_M']

OUT_DIR = './v11_bw_final'
os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BW_AUG_PKL_PATH = './bw_data_log10.pkl' if USE_LOG10 else './bw_data.pkl'


# ==================== EMA ==================== #
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


def augment(data, noise_std):
    d = copy.deepcopy(data)
    d['x'] = d['x'].clone() + torch.randn_like(d['x']) * noise_std
    return d


def compute_life_bins(failed_data):
    y_all = np.array([d['y'].item() for d in failed_data])
    q33, q67 = np.percentile(y_all, [33, 67])
    return [(0.0, q33, *HP['AUG_LOW']),
            (q33, q67, *HP['AUG_MID']),
            (q67, 999.0, *HP['AUG_HIGH'])]


def get_aug_params(y_val, bins):
    for lo, hi, n, ns in bins:
        if lo <= y_val < hi:
            return n, ns
    raise ValueError


def preprocess(sp):
    g = Data(x=sp['x'], edge_index=sp['edge_index'])
    sp_id = sp.get('ID', '')
    jt, k_hours = parse_id(sp_id)
    # BW 试件 parse_id 会返回 ('BW', 20/60), 与训练集 DJ/TX/UL 走相同路径
    return {
        'graph': g,
        'y': sp['y'],
        'ID': sp_id,
        'joint_type': jt,
        'corrosion_hours': int(k_hours),
        'hotspot_desc': sp.get('hotspot_desc', torch.zeros(HOTSPOT_DIM)),
        'x_raw': sp['x'].clone(),
    }


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


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

        self.pool_gate = nn.Sequential(nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))
        self.pool = AttentionalAggregation(self.pool_gate)

        in_dim = h + HOTSPOT_DIM
        self.sigma_head = nn.Sequential(
            nn.Linear(in_dim, h), nn.LeakyReLU(), nn.Dropout(HP['DROPOUT']),
            nn.Linear(h, h // 2), nn.LeakyReLU(), nn.Dropout(HP['DROPOUT']),
            nn.Linear(h // 2, 1))

        self.A_g = nn.Parameter(torch.tensor(A_INIT, dtype=torch.float32))
        self.m_g = nn.Parameter(torch.tensor(M_INIT, dtype=torch.float32))

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
        DS = torch.pow(10.0, log_DS)
        preds = self.A_g.expand(len(batch_data)) - self.m_g * log_DS
        info = {'A_g': self.A_g.detach().clone(),
                'm_g': self.m_g.detach().clone(),
                'DS_per': DS.detach()}
        return preds, info


def predict_bw(model, bw_data, bs, seed):
    """用 EMA 权重对 14 根 BW 做前向."""
    model.eval()
    rows = []
    A_val, m_val = None, None
    with torch.no_grad():
        for s in range(0, len(bw_data), bs):
            batch = bw_data[s:s+bs]
            preds, info = model(batch)
            if A_val is None:
                A_val = float(info['A_g'].cpu().numpy())
                m_val = float(info['m_g'].cpu().numpy())
            DS_vals = info['DS_per'].cpu().numpy()
            for i, sp in enumerate(batch):
                true_log10N = float(sp['y'].item())
                pred_log10N = float(preds[i].cpu().numpy())
                true_N = float(10.0 ** true_log10N)
                pred_N = float(10.0 ** pred_log10N)
                rows.append({
                    'seed': seed,
                    'ID': sp['ID'],
                    'joint_type': sp['joint_type'],
                    'corrosion_hours': sp['corrosion_hours'],
                    'true_N': true_N, 'pred_N': pred_N,
                    'rel_error': abs(pred_N - true_N) / (true_N + 1e-8),
                    'pred_log10N': pred_log10N,
                    'true_log10N': true_log10N,
                    'DS_MPa': float(DS_vals[i]),
                    'A_g': A_val, 'm_g': m_val,
                })
    return rows, A_val, m_val


def train_one_seed(train_raw, seed, bw_data):
    """单 seed 全量训练, 训完对 BW 推理."""
    set_seed(seed)
    t0 = time.time()

    bins = compute_life_bins(train_raw)
    train_aug = []
    for d in train_raw:
        n_aug, noise = get_aug_params(d['y'].item(), bins)
        for _ in range(n_aug):
            train_aug.append(augment(d, noise_std=noise))
    train_data = [preprocess(d) for d in train_raw + train_aug]
    n_f, n_a = len(train_raw), len(train_aug)
    print(f'  [seed={seed}] 训练: {n_f} 失效 + {n_a} 增强 = {n_f+n_a}, '
          f'固定 {HP["T_STAR"]} epoch (LOO t*=69)')

    model = PhysicsGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=HP['LR'], weight_decay=HP['WEIGHT_DECAY'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=HP['T_STAR'], eta_min=1e-5)
    ema = EMAModel(model, HP['EMA_DECAY'])
    bs = HP['BATCH_SIZE']

    for epoch in range(HP['T_STAR']):
        model.train()
        idx = list(range(len(train_data))); random.shuffle(idx)
        ep_loss = 0.0; nb = 0
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
        avg_train = ep_loss / max(nb, 1)

        A_live = float(model.A_g.detach().cpu().numpy())
        m_live = float(model.m_g.detach().cpu().numpy())
        if (epoch + 1) % 10 == 0 or epoch == HP['T_STAR'] - 1:
            print(f'    ep={epoch+1:3d}/{HP["T_STAR"]} | L_t={avg_train:.4f} | '
                  f'A={A_live:.3f} m={m_live:.3f}', flush=True)

    ema.apply(model)
    bw_rows, A_final, m_final = predict_bw(model, bw_data, bs, seed)
    time_s = time.time() - t0
    bw_mre = float(np.mean([r['rel_error'] for r in bw_rows]))
    print(f'  [seed={seed}] 完成, A={A_final:.3f} (drift={A_final-A_INIT:+.3f}) '
          f'm={m_final:.3f} (drift={m_final-M_INIT:+.3f}) | '
          f'该 seed BW MRE={bw_mre:.3f} | 用时 {time_s:.1f}s\n')

    return bw_rows, {
        'seed': seed, 'A_final': A_final, 'm_final': m_final,
        'bw_mre_this_seed': bw_mre, 'time_s': time_s,
    }


# ============================================================ #
# 主程序
# ============================================================ #
if __name__ == '__main__':
    print(f'Device: {device}  Output: {OUT_DIR}')
    print(f'V11 全量训练 (无 CV, 无早停) + BW 14 根外部泛化测试')
    print(f'  HIDDEN_DIM={HP["HIDDEN_DIM"]}, HEADS={HP["HEADS"]}, DROPOUT={HP["DROPOUT"]}')
    print(f'  LR={HP["LR"]:g}, WEIGHT_DECAY={HP["WEIGHT_DECAY"]:g}, BATCH_SIZE={HP["BATCH_SIZE"]}')
    print(f'  ★ T_STAR={HP["T_STAR"]} (论文 §4.4 LOO 全局最优 epoch)')
    print(f'  ★ 集成: {len(HP["RANDOM_SEEDS"])} seeds, log10 空间平均\n')

    # 训练集 (DJ/TX/UL)
    with open(AUG_PKL_PATH, 'rb') as f:
        all_data = pickle.load(f)
    failed = [d for d in all_data if not d.get('censored', False)]
    print(f'训练集 (DJ/TX/UL 失效, 全量): {len(failed)} 根')

    # BW 外部测试集 (14 根, 仅 KL 侧)
    if not os.path.exists(BW_AUG_PKL_PATH):
        raise FileNotFoundError(f'{BW_AUG_PKL_PATH} 不存在, 请先运行 BW_0/BW_1/BW_2')
    with open(BW_AUG_PKL_PATH, 'rb') as f:
        bw_raw = pickle.load(f)
    bw_data = [preprocess(d) for d in bw_raw]
    print(f'BW 外部测试集: {len(bw_data)} 根 (仅开裂侧 KL)\n')

    # 5 seed 全量训练 + BW 推理
    all_bw_rows = []
    all_seed_summary = []
    for seed in HP['RANDOM_SEEDS']:
        print(f'==== Seed {seed} ====')
        bw_rows, summary = train_one_seed(failed, seed, bw_data)
        all_bw_rows.extend(bw_rows)
        all_seed_summary.append(summary)

        pd.DataFrame(all_bw_rows).to_csv(os.path.join(OUT_DIR, 'bw_preds.csv'), index=False)
        pd.DataFrame(all_seed_summary).to_csv(os.path.join(OUT_DIR, 'seed_summary.csv'), index=False)

    # ============================================================ #
    # 跨 seed 集成 (log10 空间平均)
    # ============================================================ #
    df_bw = pd.DataFrame(all_bw_rows)
    df_ens = df_bw.groupby('ID').agg(
        true_N=('true_N', 'first'),
        true_log10N=('true_log10N', 'first'),
        pred_log10N_mean=('pred_log10N', 'mean'),
        pred_log10N_std=('pred_log10N', 'std'),
        joint_type=('joint_type', 'first'),
        corrosion_hours=('corrosion_hours', 'first'),
        DS_MPa_mean=('DS_MPa', 'mean'),
        n_seeds=('pred_log10N', 'count'),
    ).reset_index()
    df_ens['pred_N_ensemble'] = np.power(10.0, df_ens['pred_log10N_mean'])
    df_ens['rel_error_ensemble'] = (
        np.abs(df_ens['pred_N_ensemble'] - df_ens['true_N']) / (df_ens['true_N'] + 1e-8))
    df_ens = df_ens.sort_values('ID').reset_index(drop=True)
    df_ens.to_csv(os.path.join(OUT_DIR, 'bw_ensemble.csv'), index=False)

    # 汇总指标 (对标论文 §4.2 和 §4.4)
    t_log = df_ens['true_log10N'].values
    p_log = df_ens['pred_log10N_mean'].values
    t_N = df_ens['true_N'].values
    p_N = df_ens['pred_N_ensemble'].values

    ens_mre = float(df_ens['rel_error_ensemble'].mean())
    ens_r2_log = float(1 - np.sum((p_log - t_log) ** 2) / np.sum((t_log - t_log.mean()) ** 2))
    ens_r2_N = float(1 - np.sum((p_N - t_N) ** 2) / np.sum((t_N - t_N.mean()) ** 2))
    ens_rmse_N = float(np.sqrt(np.mean((p_N - t_N) ** 2)))
    ratio = p_N / (t_N + 1e-8)
    ens_p2x = float(np.mean((ratio >= 0.5) & (ratio <= 2.0)))

    # ==================== 终端打印汇总 ==================== #
    print('=' * 76)
    print(f'★ BW 外部泛化测试 ({len(df_ens)} 根, 每根 {len(HP["RANDOM_SEEDS"])} seed 集成):')
    print('=' * 76)
    print(f'{"ID":<10s} {"corr_d":>7s} {"true_N":>10s} {"pred_N":>12s} '
          f'{"rel_err":>9s} {"log10_std":>10s} {"ΔS (MPa)":>10s}')
    print('-' * 76)
    for _, r in df_ens.iterrows():
        print(f'{r["ID"]:<10s} {int(r["corrosion_hours"]):>7d} '
              f'{int(r["true_N"]):>10d} {int(r["pred_N_ensemble"]):>12d} '
              f'{r["rel_error_ensemble"]:>8.2%} '
              f'{r["pred_log10N_std"]:>10.3f} {r["DS_MPa_mean"]:>10.1f}')
    print('-' * 76)
    print(f'\nBW 集成整体指标:')
    print(f'  MRE           = {ens_mre:.4f}  ({ens_mre*100:.2f}%)')
    print(f'  R²_log10N     = {ens_r2_log:.4f}')
    print(f'  R²_N          = {ens_r2_N:.4f}')
    print(f'  RMSE_N        = {ens_rmse_N:.0f}')
    print(f'  P_2× coverage = {ens_p2x:.4f} ({int(ens_p2x*len(df_ens))}/{len(df_ens)})')
    print(f'\n对标:')
    print(f'  论文 §4.2 集成 (5×10 CV)  : MRE=15.35%, R²_logN=0.917, P_2×=97.4%')
    print(f'  论文 §4.4 LOO (t*=69)     : MRE=18.66%, R²_logN=0.884')
    print(f'  本测试 (BW, 5 seed 集成)  : MRE={ens_mre*100:.2f}%, R²_logN={ens_r2_log:.3f}, '
          f'P_2×={ens_p2x*100:.1f}%')

    A_vals = [s['A_final'] for s in all_seed_summary]
    m_vals = [s['m_final'] for s in all_seed_summary]
    print(f'\nBasquin 参数 (跨 seed):')
    print(f'  A = {np.mean(A_vals):.3f} ± {np.std(A_vals):.3f} '
          f'(drift vs IIW FAT80 A₀={A_INIT:.3f}: {np.mean(A_vals)-A_INIT:+.3f})')
    print(f'  m = {np.mean(m_vals):.3f} ± {np.std(m_vals):.3f} '
          f'(drift vs IIW m₀={M_INIT:.1f}: {np.mean(m_vals)-M_INIT:+.3f})')

    with open(os.path.join(OUT_DIR, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'HP': {k: (list(v) if isinstance(v, tuple) else v) for k, v in HP.items()},
            'A_INIT': A_INIT, 'M_INIT': M_INIT,
            'architecture': 'V11 full-train + BW external test',
            'training_strategy': f'no CV, no early stopping, fixed T_STAR={HP["T_STAR"]}',
            't_star_source': '§4.4 LOO global optimum (MRE=18.66%)',
            'n_train_specimens': len(failed),
            'n_bw_specimens': len(df_ens),
            'A_mean': float(np.mean(A_vals)), 'A_std': float(np.std(A_vals)),
            'm_mean': float(np.mean(m_vals)), 'm_std': float(np.std(m_vals)),
            'bw_ensemble_MRE':        ens_mre,
            'bw_ensemble_R2_logN':    ens_r2_log,
            'bw_ensemble_R2_N':       ens_r2_N,
            'bw_ensemble_RMSE_N':     ens_rmse_N,
            'bw_ensemble_P2X':        ens_p2x,
        }, f, indent=2, ensure_ascii=False)

    print(f'\n输出: {OUT_DIR}/')
    print('  bw_preds.csv       — 每 seed 每根 BW 原始预测 (70 行)')
    print('  bw_ensemble.csv    — 跨 seed 集成 (14 行)')
    print('  seed_summary.csv   — 每 seed 的 A/m/MRE/用时')
    print('  meta.json          — HP + 汇总指标')
    print(f'\n下一步: python BW_4_plot_analysis.py')
