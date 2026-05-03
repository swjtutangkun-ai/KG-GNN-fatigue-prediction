# -*- coding: utf-8 -*-
"""
3_train_v11.py — V11 final baseline (V10 去除边特征)

★ 与 V10 唯一区别: GATv2Conv(edge_dim=None), 不使用边特征 [dx, dy, dz]
  消融实验 G-3 验证: 去除边特征后 MRE 从 0.166 降至 0.160, 边特征为冗余信息.
  参考: Krokos et al. 2024 Sci Reports — 边拓扑而非边属性是关键信息载体.

★ HP: V11 TPE search 最优 (H=64, heads=4, D=0.6, LR=4e-4, WD=1e-4, BS=4)
  Trial 57, MRE=0.1554, 参数量 ~16K.

★ 其余全部与 V10 一致: 增强策略, CV 协议, EMA, 物理头, 注意力导出.

CV 协议: 5 seeds × 10 fold = 50 folds
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
from torch_geometric.utils import softmax as pyg_softmax
from sklearn.model_selection import StratifiedKFold
import pandas as pd

from config_single import (
    FEAT_DIM, EDGE_DIM, HOTSPOT_DIM, AUG_PKL_PATH, RESULT_DIR,
    parse_id, USE_LOG10, FAT_INIT,
)

# ============================================================ #
# HP — V11 TPE search best (Trial 57, MRE=0.1554)
# ============================================================ #
HP = {
    # 网络
    'HIDDEN_DIM':       64,
    'HEADS':            4,
    'DROPOUT':          0.6,

    # 优化
    'LR':               4e-4,
    'WEIGHT_DECAY':     1e-4,
    'EPOCHS':           500,
    'BATCH_SIZE':       4,

    # 早停
    'PATIENCE':         50,
    'EMA_DECAY':        0.995,
    'ES_MIN_DELTA':     5e-4,

    # CV (5 seeds final evaluation)
    'RANDOM_SEEDS':     [11, 22, 33, 44, 55],
    'N_SPLITS':         10,

    # 物理头初值
    'BASQUIN_M':        3.0,
    'LOG_DS_CENTER':    2.0,
    'LOG_DS_SCALE':     1.0,

    # 数据增强
    'AUG_LOW':          (2, 0.03),
    'AUG_MID':          (3, 0.05),
    'AUG_HIGH':         (6, 0.08),

    # 注意力导出 (§5.2 用)
    'EXPORT_ATTENTION': True,
}

assert USE_LOG10 and FEAT_DIM == 3  # ★ V11: 不再 assert EDGE_DIM == 3

JT_NAMES = ['DJ', 'TX', 'UL']
A_PER_JT = np.array([6.301 + HP['BASQUIN_M'] * np.log10(FAT_INIT[k]) for k in JT_NAMES],
                    dtype=np.float32)
A_INIT = float(A_PER_JT.mean())
M_INIT = HP['BASQUIN_M']

OUT_DIR = './v11_final'  # ★ V11: 输出目录
os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    def state_dict(self):
        return {k: v.cpu().clone() for k, v in self.shadow.items()}


def augment(data, noise_std):
    d = copy.deepcopy(data)
    d['x'] = d['x'].clone() + torch.randn_like(d['x']) * noise_std
    # ★ V11: 不再增强 edge_attr (模型不使用边特征)
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
    """保留原始节点特征 (§5.2 注意力关联用) + 分组字段 (§4.2.2 用)."""
    # ★ V11: 不传 edge_attr 给 Data (模型不使用)
    g = Data(x=sp['x'], edge_index=sp['edge_index'])
    sp_id = sp.get('ID', '')
    jt, k_hours = parse_id(sp_id)
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
    torch.backends.cudnn.benchmark = True  # ★ 速度优先, 不固定


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


# ==================== 模型 ==================== #
class PhysicsGNN(nn.Module):
    """V11: GATv2 backbone (无边特征) + physics-encoded output layer.
       log10(N) = A_g - m_g · log10(ΔS_eq)

       ★ V11 vs V10: edge_dim=None, 注意力仅基于节点特征计算.
    """
    def __init__(self):
        super().__init__()
        h = HP['HIDDEN_DIM']
        self.drop = nn.Dropout(HP['DROPOUT'])
        # ★ V11: edge_dim=None (不使用边特征)
        self.conv1 = GATv2Conv(FEAT_DIM, h, heads=HP['HEADS'], edge_dim=None)
        self.bn1 = nn.BatchNorm1d(h * HP['HEADS'])
        self.conv2 = GATv2Conv(h * HP['HEADS'], h, heads=1, edge_dim=None)
        self.bn2 = nn.BatchNorm1d(h)

        # gate_nn 独立命名, 方便从外部复用计算 gate 权重
        self.pool_gate = nn.Sequential(nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))
        self.pool = AttentionalAggregation(self.pool_gate)

        in_dim = h + HOTSPOT_DIM
        self.sigma_head = nn.Sequential(
            nn.Linear(in_dim, h), nn.LeakyReLU(), nn.Dropout(HP['DROPOUT']),
            nn.Linear(h, h // 2), nn.LeakyReLU(), nn.Dropout(HP['DROPOUT']),
            nn.Linear(h // 2, 1))

        self.A_g = nn.Parameter(torch.tensor(A_INIT, dtype=torch.float32))
        self.m_g = nn.Parameter(torch.tensor(M_INIT, dtype=torch.float32))
        self.return_attention = False

    def forward(self, batch_data):
        graphs = [sp['graph'] for sp in batch_data]
        batch = Batch.from_data_list(graphs).to(device)
        # ★ V11: 不传 edge_attr
        x = self.drop(F.leaky_relu(self.bn1(self.conv1(batch.x, batch.edge_index))))
        x = self.drop(F.leaky_relu(self.bn2(self.conv2(x, batch.edge_index))))

        # Pooling gate 权重 (per-graph softmax)
        gate_logits = self.pool_gate(x).squeeze(-1)
        gate_weights = pyg_softmax(gate_logits, batch.batch)

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
        if self.return_attention:
            info['gate_weights'] = gate_weights.detach()
            info['batch_idx'] = batch.batch.detach()
        return preds, info


def evaluate(model, val_data, bs, capture_attention=False):
    """capture_attention=True 时额外返回 per-node 注意力权重."""
    model.eval()
    old_return = model.return_attention
    model.return_attention = capture_attention

    vp, vt, v_ids, per_sp = [], [], [], []
    attention_rows = []
    A_snap, m_snap = None, None

    with torch.no_grad():
        for s in range(0, len(val_data), bs):
            batch = val_data[s:s+bs]
            preds, info = model(batch)
            vp.extend(preds.cpu().numpy())
            vt.extend([sp['y'].item() for sp in batch])
            v_ids.extend([sp['ID'] for sp in batch])
            A_val = float(info['A_g'].cpu().numpy())
            DS_vals = info['DS_per'].cpu().numpy()
            for i, sp in enumerate(batch):
                per_sp.append({'ID': sp['ID'], 'A': A_val, 'DS_MPa': float(DS_vals[i])})
            if A_snap is None:
                A_snap = A_val
                m_snap = float(info['m_g'].cpu().numpy())

            if capture_attention and 'gate_weights' in info:
                gw = info['gate_weights'].cpu().numpy()
                bi = info['batch_idx'].cpu().numpy()
                for local_i, sp in enumerate(batch):
                    node_mask = bi == local_i
                    x_raw = sp['x_raw'].numpy()
                    node_gates = gw[node_mask]
                    for node_idx in range(len(node_gates)):
                        attention_rows.append({
                            'ID': sp['ID'],
                            'joint_type': sp['joint_type'],
                            'corrosion_hours': sp['corrosion_hours'],
                            'node_idx': int(node_idx),
                            'attention': float(node_gates[node_idx]),
                            'sigma1':   float(x_raw[node_idx, 0]),
                            'tau_max':  float(x_raw[node_idx, 1]),
                            'sigma_vm': float(x_raw[node_idx, 2]),
                        })

    model.return_attention = old_return

    vp, vt = np.array(vp), np.array(vt)
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
            'R2_N': r2_N, 'RMSE_N': rmse_N, 'P2X_cov': p2x}, \
           vp, vt, v_ids, per_sp, A_snap, m_snap, attention_rows


def train_fold(train_failed_raw, val_failed_raw, seed, fold_i):
    set_seed(seed)
    t0 = time.time()

    bins = compute_life_bins(train_failed_raw)
    train_aug = []
    for d in train_failed_raw:
        n_aug, noise = get_aug_params(d['y'].item(), bins)
        for _ in range(n_aug):
            train_aug.append(augment(d, noise_std=noise))

    train_data = [preprocess(d) for d in train_failed_raw + train_aug]
    val_data = [preprocess(d) for d in val_failed_raw]

    n_f, n_a = len(train_failed_raw), len(train_aug)
    print(f'    训练: {n_f}失效+{n_a}增强={n_f+n_a}, 验证: {len(val_failed_raw)}')
    print(f'    自适应增强分界: q33={bins[0][1]:.3f} (N≈{10**bins[0][1]:.0f}), '
          f'q67={bins[1][1]:.3f} (N≈{10**bins[1][1]:.0f})')

    model = PhysicsGNN().to(device)
    n_total = sum(p.numel() for p in model.parameters())
    print(f'    模型参数: 总={n_total}')

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=HP['LR'],
                                 weight_decay=HP['WEIGHT_DECAY'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=HP['EPOCHS'], eta_min=1e-5)
    es = EarlyStopping()
    ema = EMAModel(model, HP['EMA_DECAY'])
    bs = HP['BATCH_SIZE']

    best_ep = 0
    final_ep = 0
    hist_rows = []

    for epoch in range(HP['EPOCHS']):
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
        final_ep = epoch + 1
        avg_train = ep_loss / max(nb, 1)

        ema.apply(model)
        metrics, _, _, _, val_per_sp, A_now, m_now, _ = evaluate(model, val_data, bs)
        ema.restore(model)
        prev_best = es.best_score
        es(metrics['MRE'], ema)
        if es.best_score != prev_best:
            best_ep = final_ep

        DS_arr = np.array([r['DS_MPa'] for r in val_per_sp])
        print(f'    ep={final_ep:3d} | L_t={avg_train:.4f} | '
              f'logMAE={metrics["logMAE"]:.4f} MRE={metrics["MRE"]:.4f} | '
              f'A={A_now:.3f} m={m_now:.3f} | '
              f'ΔS=[{DS_arr.min():.0f},{DS_arr.max():.0f}]MPa | '
              f'wait={es.counter}', flush=True)

        hist_rows.append({
            'seed': seed, 'fold': fold_i, 'epoch': final_ep,
            'train_loss': avg_train, 'val_mre': metrics['MRE'],
            'val_logmae': metrics['logMAE'],
            'A_g': float(A_now), 'm_g': m_now,
            'ds_min': float(DS_arr.min()), 'ds_max': float(DS_arr.max()),
            'wait': es.counter,
        })

        if es.early_stop:
            print(f'    ep={final_ep:3d} | 早停 (best MRE={-es.best_score:.4f}, '
                  f'best_ep={best_ep})', flush=True)
            break

    if es.best_ema_state is not None:
        model.load_state_dict(es.best_ema_state); model.to(device)

    # 最终评估 + 注意力导出
    metrics, vp, vt, v_ids, per_sp, A_final, m_final, attn_rows = evaluate(
        model, val_data, bs, capture_attention=HP['EXPORT_ATTENTION'])
    time_s = time.time() - t0
    DS_arr = np.array([r['DS_MPa'] for r in per_sp])
    wait_after_best = final_ep - best_ep

    print(f'    最终: MRE={metrics["MRE"]:.4f} R²_N={metrics["R2_N"]:.3f} '
          f'A={A_final:.3f} (drift={A_final-A_INIT:+.3f}) '
          f'm={m_final:.3f} (drift={m_final-M_INIT:+.3f})')
    print(f'    final_ep={final_ep} | best_ep={best_ep} | '
          f'wait_after_best={wait_after_best} | 时间={time_s:.1f}s')

    fold_row = {
        **metrics,
        'A': A_final, 'A_drift': A_final - A_INIT,
        'm': m_final, 'm_drift': m_final - M_INIT,
        'dS_min': float(DS_arr.min()), 'dS_max': float(DS_arr.max()),
        'dS_range': float(DS_arr.max() - DS_arr.min()),
        'best_ep': best_ep, 'final_ep': final_ep,
        'wait_after_best': wait_after_best,
        'time_s': time_s, 'n_params': n_total,
    }

    id_to_meta = {sp['ID']: (sp['joint_type'], sp['corrosion_hours']) for sp in val_data}
    spec_rows = []
    for i in range(len(vp)):
        true_N = float(10.0 ** vt[i]); pred_N = float(10.0 ** vp[i])
        jt, k_hours = id_to_meta.get(v_ids[i], ('UNKNOWN', -1))
        spec_rows.append({
            'seed': seed, 'fold': fold_i, 'ID': v_ids[i],
            'joint_type': jt,
            'corrosion_hours': k_hours,
            'true_N': true_N, 'pred_N': pred_N,
            'rel_error': abs(pred_N - true_N) / (true_N + 1e-8),
            'pred_log10N': float(vp[i]), 'true_log10N': float(vt[i]),
            'A': A_final, 'DS_MPa': per_sp[i]['DS_MPa'],
        })

    for r in attn_rows:
        r['seed'] = seed
        r['fold'] = fold_i

    return fold_row, spec_rows, hist_rows, attn_rows


if __name__ == '__main__':
    print(f'Device: {device}  Output: {OUT_DIR}')
    print(f'V11 Final (V10 去除边特征, edge_dim=None)')
    print(f'  HIDDEN_DIM = {HP["HIDDEN_DIM"]}, HEADS = {HP["HEADS"]}, '
          f'DROPOUT = {HP["DROPOUT"]}')
    print(f'  LR = {HP["LR"]:g}, WEIGHT_DECAY = {HP["WEIGHT_DECAY"]:g}')
    print(f'  BATCH_SIZE = {HP["BATCH_SIZE"]}')
    print(f'  edge_dim = None (★ V11 改动)')
    print(f'  CV: {len(HP["RANDOM_SEEDS"])} seeds × {HP["N_SPLITS"]} fold = '
          f'{len(HP["RANDOM_SEEDS"]) * HP["N_SPLITS"]} folds\n')

    with open(AUG_PKL_PATH, 'rb') as f:
        all_data = pickle.load(f)
    failed = [d for d in all_data if not d.get('censored', False)]
    print(f'失效样本: {len(failed)}\n')

    # ★ 预生成并保存 CV 切分 (消融实验复用, 保证配对)
    cv_splits = {}
    for seed in HP['RANDOM_SEEDS']:
        y_all = np.array([d['y'].item() for d in failed])
        y_bins = pd.qcut(y_all, q=HP['N_SPLITS'], labels=False, duplicates='drop')
        skf = StratifiedKFold(n_splits=HP['N_SPLITS'], shuffle=True, random_state=seed)
        for fold_i, (ti, vi) in enumerate(skf.split(np.zeros(len(failed)), y_bins)):
            cv_splits[f's{seed}_f{fold_i}'] = {
                'seed': int(seed), 'fold': int(fold_i),
                'train_idx': ti.tolist(), 'val_idx': vi.tolist(),
                'val_ids': [failed[i].get('ID', '') for i in vi],
            }
    with open(os.path.join(OUT_DIR, 'cv_splits.json'), 'w', encoding='utf-8') as f:
        json.dump(cv_splits, f, indent=2, ensure_ascii=False)
    print(f'CV 切分已保存: {OUT_DIR}/cv_splits.json ({len(cv_splits)} folds)\n')

    all_fold_rows, all_spec_rows, all_hist_rows, all_attn_rows = [], [], [], []
    for seed in HP['RANDOM_SEEDS']:
        print(f'==== Seed {seed} ====')
        for fold_i in range(HP['N_SPLITS']):
            key = f's{seed}_f{fold_i}'
            ti = cv_splits[key]['train_idx']
            vi = cv_splits[key]['val_idx']
            print(f'--- seed={seed} fold={fold_i} ---')
            tf = [failed[i] for i in ti]; vf = [failed[i] for i in vi]
            fr, sr, hr, ar = train_fold(tf, vf, seed, fold_i)
            fr['seed'] = seed; fr['fold'] = fold_i
            all_fold_rows.append(fr)
            all_spec_rows.extend(sr)
            all_hist_rows.extend(hr)
            all_attn_rows.extend(ar)
            print(f'  >> s={seed} f={fold_i} | MRE={fr["MRE"]:.4f} R²_N={fr["R2_N"]:.3f} '
                  f'A={fr["A"]:.3f} m={fr["m"]:.3f} best_ep={fr["best_ep"]} '
                  f'final_ep={fr["final_ep"]} | {fr["time_s"]:.1f}s\n')

            # 增量保存
            pd.DataFrame(all_fold_rows).to_csv(
                os.path.join(OUT_DIR, 'results.csv'), index=False)
            pd.DataFrame(all_spec_rows).to_csv(
                os.path.join(OUT_DIR, 'specimen_preds.csv'), index=False)
            pd.DataFrame(all_hist_rows).to_csv(
                os.path.join(OUT_DIR, 'training_history.csv'), index=False)
            if all_attn_rows:
                pd.DataFrame(all_attn_rows).to_csv(
                    os.path.join(OUT_DIR, 'attention_export.csv'), index=False)

    # ============================================================ #
    # 最终汇总
    # ============================================================ #
    df = pd.DataFrame(all_fold_rows)
    df_sp = pd.DataFrame(all_spec_rows)

    print('\n' + '=' * 70)
    print(f'V11 Final 汇总 ({len(HP["RANDOM_SEEDS"])} seeds × '
          f'{HP["N_SPLITS"]} fold = {len(df)} folds):')
    print('=' * 70)
    for col, name, prec in [('MRE', 'MRE', 4), ('logMAE', 'logMAE', 4),
                             ('R2_logN', 'R²_logN', 3), ('R2_N', 'R²_N', 3),
                             ('P2X_cov', 'P_2X_cov', 3),
                             ('RMSE_N', 'RMSE_N', 0),
                             ('A', 'A', 3), ('m', 'm', 3),
                             ('best_ep', 'best_epoch', 0),
                             ('wait_after_best', 'wait_after_best', 0)]:
        v = df[col]
        print(f'  {name:18s}: {v.mean():.{prec}f} ± {v.std():.{prec}f}')

    # 集成 (跨 seed log-space 平均)
    df_ens = df_sp.groupby('ID').agg(
        true_N=('true_N', 'first'),
        true_log10N=('true_log10N', 'first'),
        pred_log10N_mean=('pred_log10N', 'mean'),
        pred_log10N_std=('pred_log10N', 'std'),
        joint_type=('joint_type', 'first'),
        corrosion_hours=('corrosion_hours', 'first'),
        DS_MPa_mean=('DS_MPa', 'mean'),
        n_evals=('pred_log10N', 'count'),
    ).reset_index()
    df_ens['pred_N_ensemble'] = np.power(10.0, df_ens['pred_log10N_mean'])
    df_ens['rel_error_ensemble'] = (
        np.abs(df_ens['pred_N_ensemble'] - df_ens['true_N']) / (df_ens['true_N'] + 1e-8))

    ens_mre = float(df_ens['rel_error_ensemble'].mean())
    t_log, p_log = df_ens['true_log10N'].values, df_ens['pred_log10N_mean'].values
    t_N, p_N = df_ens['true_N'].values, df_ens['pred_N_ensemble'].values
    ens_r2_log = float(1 - np.sum((p_log - t_log) ** 2) / np.sum((t_log - t_log.mean()) ** 2))
    ens_r2_N = float(1 - np.sum((p_N - t_N) ** 2) / np.sum((t_N - t_N.mean()) ** 2))
    ens_rmse_N = float(np.sqrt(np.mean((p_N - t_N) ** 2)))
    ens_p2x = float(np.mean((p_N / (t_N + 1e-8) >= 0.5) & (p_N / (t_N + 1e-8) <= 2.0)))

    print(f'\n集成 (跨 seed log-space 平均):')
    print(f'  ensemble MRE    : {ens_mre:.4f}')
    print(f'  ensemble R²_logN: {ens_r2_log:.4f}')
    print(f'  ensemble R²_N   : {ens_r2_N:.4f}')
    print(f'  ensemble RMSE_N : {ens_rmse_N:.0f}')
    print(f'  ensemble P_2X   : {ens_p2x:.4f}')

    df_ens.to_csv(os.path.join(OUT_DIR, 'specimen_ensemble.csv'), index=False)

    # PATIENCE 诊断
    pct_at_patience = (df['wait_after_best'] >= HP['PATIENCE']).mean() * 100
    print(f'\n[PATIENCE 合理性诊断]')
    print(f'  fold 中 wait_after_best ≥ PATIENCE ({HP["PATIENCE"]}) 的比例: '
          f'{pct_at_patience:.1f}%')
    print(f'  best_ep 范围: [{df["best_ep"].min()}, {df["best_ep"].max()}], '
          f'中位数 {int(df["best_ep"].median())}')

    with open(os.path.join(OUT_DIR, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'HP': {k: (list(v) if isinstance(v, tuple) else v) for k, v in HP.items()},
            'A_INIT': A_INIT, 'M_INIT': M_INIT,
            'JT_NAMES': JT_NAMES,
            'FAT_INIT': FAT_INIT,
            'architecture': 'V11: 2×GATv2(edge_dim=None) + AttentionalAggregation + Basquin',
            'hp_source': 'V11 TPE search best (Trial 57, MRE=0.1554)',
            'cv_strategy': f'{len(HP["RANDOM_SEEDS"])} seeds × {HP["N_SPLITS"]} fold StratifiedKFold',
            'n_failed_specimens': len(failed),
            'per_fold_MRE_mean':   float(df['MRE'].mean()),
            'per_fold_MRE_std':    float(df['MRE'].std()),
            'ensemble_MRE':        ens_mre,
            'ensemble_R2_logN':    ens_r2_log,
            'ensemble_R2_N':       ens_r2_N,
            'ensemble_RMSE_N':     ens_rmse_N,
            'ensemble_P2X':        ens_p2x,
            'best_ep_max':         int(df['best_ep'].max()),
            'best_ep_median':      int(df['best_ep'].median()),
        }, f, indent=2, ensure_ascii=False)

    print(f'\n输出: {OUT_DIR}/')
    print('  results.csv            — 每 fold 指标 (50 行)')
    print('  specimen_preds.csv     — 每 specimen 每 fold')
    print('  specimen_ensemble.csv  — 跨 seed 集成')
    print('  training_history.csv   — 每 epoch 训练曲线')
    print('  attention_export.csv   — 每节点注意力权重 (§5.2)')
    print('  cv_splits.json         — CV 切分记录 (消融复用)')
    print('  meta.json              — HP + 核心指标')
    print(f'\n下一步: python analyze_baseline.py --result_dir {OUT_DIR}')
