# -*- coding: utf-8 -*-
"""
5_1_ablation_all.py — §5.1 消融实验 (合并 §5.1.1 + §5.1.2 + §5.1.3)

§5.1.1 物理信息注入方式 (P 系列, 4 个变体):
  P-A1: (= V11 baseline, 不重跑, 从 v11_final 读取)
  P-A2: 无先验初始化 (A/m 随机初始化)            — IIW 先验的价值
  P-B:  损失层软约束 (MLP + λ·Basquin 惩罚)     — Karniadakis 2021
  P-C:  不注入 (纯 MLP 回归)                     — 数据驱动 baseline

§5.1.2 图结构 (G 系列, 4+2 个变体):
  G-1: 去 GNN (仅 hotspot → Basquin)            — GNN 是否必要
  G-2: GCN 替代 GATv2                           — 注意力机制的贡献
  G-3: 加边特征 (edge_dim=EDGE_DIM)              — 反向消融
  G-4: 均值池化替代 AttentionalAggregation       — 注意力池化的贡献
  G-5: 1 层 GATv2 (可选)                         — GNN 深度
  G-6: 3 层 GATv2 (可选)                         — GNN 深度

§5.1.3 输入与增强 (I 系列, 4 个变体):
  I-1: 去 hotspot 描述符                         — hotspot 统计量的贡献
  I-2: 无增强 (仅原始 114 样本)                   — 增强的必要性
  I-3: 均匀增强 (3×, noise=0.05)                 — 分层 vs 均匀
  I-4: 节点只用 σ₁ (去掉 τ_max, σ_vm)           — 验证 §5.2 注意力发现

CV: 复用 v11_final/cv_splits.json (配对)
Seeds: [11, 22, 33] (3 seeds × 10 fold = 30 folds)

用法:
  python 5_1_ablation_all.py [--cv_splits ./v11_final/cv_splits.json]
                              [--out_dir ./ablation_5_1]
                              [--variants P-A1,P-A2,P-B,P-C,G-1,G-2,G-3,G-4,I-1,I-2,I-3,I-4]
"""
import os, json, random, pickle, time, copy, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv, GCNConv, global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation
import pandas as pd

from config_single import (
    FEAT_DIM, EDGE_DIM, HOTSPOT_DIM, AUG_PKL_PATH,
    parse_id, USE_LOG10, FAT_INIT,
)

assert USE_LOG10 and FEAT_DIM == 3  # ★ V11: 不再 assert EDGE_DIM

# ============================================================ #
# HP (V11 TPE 最优, Trial 57, MRE=0.1554)
# ============================================================ #
HP = {
    'HIDDEN_DIM': 64, 'HEADS': 4, 'DROPOUT': 0.6,  # ★ V11 TPE
    'LR': 4e-4, 'WEIGHT_DECAY': 1e-4,
    'EPOCHS': 500, 'BATCH_SIZE': 4,  # ★ V11 TPE
    'PATIENCE': 50, 'EMA_DECAY': 0.995, 'ES_MIN_DELTA': 5e-4,
    'BASQUIN_M': 3.0, 'LOG_DS_CENTER': 2.0, 'LOG_DS_SCALE': 1.0,
    'AUG_LOW': (2, 0.03), 'AUG_MID': (3, 0.05), 'AUG_HIGH': (6, 0.08),
}
ABLATION_SEEDS = [11, 22, 33]
N_SPLITS = 10
LAMBDA_PB = 1.0  # P-B 软约束强度

JT_NAMES = ['DJ', 'TX', 'UL']
A_PER_JT = np.array([6.301 + HP['BASQUIN_M'] * np.log10(FAT_INIT[k])
                      for k in JT_NAMES], dtype=np.float32)
A_INIT = float(A_PER_JT.mean())
M_INIT = HP['BASQUIN_M']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================ #
# 增强策略配置
# ============================================================ #
AUG_CONFIGS = {
    'baseline': {'enabled': True, 'description': 'Stratified (V11 default)'},
    'none':     {'enabled': False, 'description': 'No augmentation'},
    'uniform':  {'enabled': True, 'uniform': True, 'n_aug': 3, 'noise': 0.05,
                 'description': 'Uniform 3x noise=0.05'},
}


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


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


def augment(data, noise_std, include_edge_attr=False):
    d = copy.deepcopy(data)
    d['x'] = d['x'].clone() + torch.randn_like(d['x']) * noise_std
    if include_edge_attr and d.get('edge_attr') is not None and d['edge_attr'].numel() > 0:
        d['edge_attr'] = d['edge_attr'].clone() + torch.randn_like(d['edge_attr']) * noise_std
    return d


def preprocess(sp, include_edge_attr=False, feat_slice=None):
    x = sp['x']
    if feat_slice is not None:
        x = x[:, feat_slice]
    if include_edge_attr:
        ea = sp.get('edge_attr', None)
        g = Data(x=x, edge_index=sp['edge_index'],
                 **({'edge_attr': ea} if ea is not None else {}))
    else:
        g = Data(x=x, edge_index=sp['edge_index'])
    sp_id = sp.get('ID', '')
    jt, k_hours = parse_id(sp_id)
    return {'graph': g, 'y': sp['y'], 'ID': sp_id,
            'joint_type': jt, 'corrosion_hours': int(k_hours),
            'hotspot_desc': sp.get('hotspot_desc', torch.zeros(HOTSPOT_DIM))}


def compute_life_bins(failed_data):
    y_all = np.array([d['y'].item() for d in failed_data])
    q33, q67 = np.percentile(y_all, [33, 67])
    return [(0.0, q33, *HP['AUG_LOW']), (q33, q67, *HP['AUG_MID']),
            (q67, 999.0, *HP['AUG_HIGH'])]


def get_aug_params(y_val, bins):
    for lo, hi, n, ns in bins:
        if lo <= y_val < hi: return n, ns
    raise ValueError


def do_augmentation(train_raw, aug_config, include_edge_attr=False):
    if not aug_config.get('enabled', True):
        return []
    train_aug = []
    if aug_config.get('uniform', False):
        n_aug, noise = aug_config['n_aug'], aug_config['noise']
        for d in train_raw:
            for _ in range(n_aug):
                train_aug.append(augment(d, noise, include_edge_attr))
    else:
        bins = compute_life_bins(train_raw)
        for d in train_raw:
            n_aug, noise = get_aug_params(d['y'].item(), bins)
            for _ in range(n_aug):
                train_aug.append(augment(d, noise, include_edge_attr))
    return train_aug


# ============================================================ #
# Basquin 工具 (多个模型共用)
# ============================================================ #
def make_sigma_head(in_dim, h):
    return nn.Sequential(
        nn.Linear(in_dim, h), nn.LeakyReLU(), nn.Dropout(HP['DROPOUT']),
        nn.Linear(h, h // 2), nn.LeakyReLU(), nn.Dropout(HP['DROPOUT']),
        nn.Linear(h // 2, 1))


def basquin_forward(sigma_head, A_g, m_g, zh):
    log_DS = HP['LOG_DS_CENTER'] + HP['LOG_DS_SCALE'] * torch.tanh(
        sigma_head(zh).squeeze(-1))
    preds = A_g - m_g * log_DS
    return preds, {'A_g': A_g.detach().clone(), 'm_g': m_g.detach().clone(),
                   'DS_per': torch.pow(10.0, log_DS).detach(), 'log_DS': log_DS.detach()}


# ============================================================ #
# GNN Backbone (P 系列共用, V11: edge_dim=None)
# ============================================================ #
class GNNBackbone(nn.Module):
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

    def forward(self, batch_data):
        graphs = [sp['graph'] for sp in batch_data]
        batch = Batch.from_data_list(graphs).to(device)
        x = self.drop(F.leaky_relu(self.bn1(self.conv1(batch.x, batch.edge_index))))
        x = self.drop(F.leaky_relu(self.bn2(self.conv2(x, batch.edge_index))))
        z = self.pool(x, batch.batch)
        hs = torch.stack([sp['hotspot_desc'] for sp in batch_data]).to(device)
        return torch.cat([z, hs], dim=1)


# ============================================================ #
# §5.1.1 P 系列模型
# ============================================================ #
class Model_PA1(nn.Module):
    VARIANT, DESCRIPTION = 'P-A1', 'PeNN hard-coded Basquin (V11 baseline)'
    def __init__(self):
        super().__init__()
        self.backbone = GNNBackbone()
        h = HP['HIDDEN_DIM']; in_dim = h + HOTSPOT_DIM
        self.sigma_head = make_sigma_head(in_dim, h)
        self.A_g = nn.Parameter(torch.tensor(A_INIT))
        self.m_g = nn.Parameter(torch.tensor(M_INIT))
    def forward(self, batch_data):
        return basquin_forward(self.sigma_head, self.A_g, self.m_g, self.backbone(batch_data))
    def compute_loss(self, preds, targets, info):
        return F.smooth_l1_loss(preds, targets)


class Model_PA2(nn.Module):
    VARIANT, DESCRIPTION = 'P-A2', 'Random init A/m (no IIW prior)'
    def __init__(self):
        super().__init__()
        self.backbone = GNNBackbone()
        h = HP['HIDDEN_DIM']; in_dim = h + HOTSPOT_DIM
        self.sigma_head = make_sigma_head(in_dim, h)
        self.A_g = nn.Parameter(torch.randn(1).squeeze())
        self.m_g = nn.Parameter(torch.rand(1).squeeze() + 0.5)
    def forward(self, batch_data):
        return basquin_forward(self.sigma_head, self.A_g, self.m_g, self.backbone(batch_data))
    def compute_loss(self, preds, targets, info):
        return F.smooth_l1_loss(preds, targets)


class Model_PB(nn.Module):
    VARIANT, DESCRIPTION = 'P-B', f'PiNN soft constraint (lambda={LAMBDA_PB})'
    def __init__(self):
        super().__init__()
        self.backbone = GNNBackbone()
        h = HP['HIDDEN_DIM']; in_dim = h + HOTSPOT_DIM
        self.pred_head = make_sigma_head(in_dim, h)
        self.sigma_head = make_sigma_head(in_dim, h)
        self.A_g = nn.Parameter(torch.tensor(A_INIT))
        self.m_g = nn.Parameter(torch.tensor(M_INIT))
    def forward(self, batch_data):
        zh = self.backbone(batch_data)
        preds = self.pred_head(zh).squeeze(-1)
        log_DS = HP['LOG_DS_CENTER'] + HP['LOG_DS_SCALE'] * torch.tanh(
            self.sigma_head(zh).squeeze(-1))
        basquin_ref = self.A_g - self.m_g * log_DS
        return preds, {'A_g': self.A_g.detach().clone(), 'm_g': self.m_g.detach().clone(),
                       'DS_per': torch.pow(10.0, log_DS).detach(), 'basquin_ref': basquin_ref}
    def compute_loss(self, preds, targets, info):
        return F.smooth_l1_loss(preds, targets) + LAMBDA_PB * F.mse_loss(preds, info['basquin_ref'])


class Model_PC(nn.Module):
    VARIANT, DESCRIPTION = 'P-C', 'Pure data-driven GNN+MLP (no physics)'
    def __init__(self):
        super().__init__()
        self.backbone = GNNBackbone()
        h = HP['HIDDEN_DIM']; in_dim = h + HOTSPOT_DIM
        self.pred_head = make_sigma_head(in_dim, h)
    def forward(self, batch_data):
        return self.pred_head(self.backbone(batch_data)).squeeze(-1), {}
    def compute_loss(self, preds, targets, info):
        return F.smooth_l1_loss(preds, targets)


# ============================================================ #
# §5.1.2 G 系列模型
# ============================================================ #
class Model_G1(nn.Module):
    VARIANT, DESCRIPTION = 'G-1', 'No GNN (hotspot descriptors only)'
    def __init__(self):
        super().__init__()
        h = HP['HIDDEN_DIM']
        self.hotspot_encoder = nn.Sequential(
            nn.Linear(HOTSPOT_DIM, h), nn.LeakyReLU(), nn.Dropout(HP['DROPOUT']),
            nn.Linear(h, h))
        self.sigma_head = make_sigma_head(h, h)
        self.A_g = nn.Parameter(torch.tensor(A_INIT))
        self.m_g = nn.Parameter(torch.tensor(M_INIT))
    def forward(self, batch_data):
        hs = torch.stack([sp['hotspot_desc'] for sp in batch_data]).to(device)
        return basquin_forward(self.sigma_head, self.A_g, self.m_g, self.hotspot_encoder(hs))
    def compute_loss(self, preds, targets, info):
        return F.smooth_l1_loss(preds, targets)


class Model_G2(nn.Module):
    VARIANT, DESCRIPTION = 'G-2', 'GCN backbone (no attention)'
    def __init__(self):
        super().__init__()
        h = HP['HIDDEN_DIM']; self.drop = nn.Dropout(HP['DROPOUT'])
        self.conv1 = GCNConv(FEAT_DIM, h); self.bn1 = nn.BatchNorm1d(h)
        self.conv2 = GCNConv(h, h);        self.bn2 = nn.BatchNorm1d(h)
        self.pool = AttentionalAggregation(
            nn.Sequential(nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1)))
        self.sigma_head = make_sigma_head(h + HOTSPOT_DIM, h)
        self.A_g = nn.Parameter(torch.tensor(A_INIT))
        self.m_g = nn.Parameter(torch.tensor(M_INIT))
    def forward(self, batch_data):
        batch = Batch.from_data_list([sp['graph'] for sp in batch_data]).to(device)
        x = self.drop(F.leaky_relu(self.bn1(self.conv1(batch.x, batch.edge_index))))
        x = self.drop(F.leaky_relu(self.bn2(self.conv2(x, batch.edge_index))))
        z = self.pool(x, batch.batch)
        hs = torch.stack([sp['hotspot_desc'] for sp in batch_data]).to(device)
        return basquin_forward(self.sigma_head, self.A_g, self.m_g, torch.cat([z, hs], 1))
    def compute_loss(self, preds, targets, info):
        return F.smooth_l1_loss(preds, targets)


class Model_G3(nn.Module):
    """★ 反向消融: V11 baseline 无边特征, 此变体加回 edge_dim=EDGE_DIM."""
    VARIANT, DESCRIPTION = 'G-3', 'Add edge features (reverse ablation)'
    def __init__(self):
        super().__init__()
        h = HP['HIDDEN_DIM']; self.drop = nn.Dropout(HP['DROPOUT'])
        self.conv1 = GATv2Conv(FEAT_DIM, h, heads=HP['HEADS'], edge_dim=EDGE_DIM)
        self.bn1 = nn.BatchNorm1d(h * HP['HEADS'])
        self.conv2 = GATv2Conv(h * HP['HEADS'], h, heads=1, edge_dim=EDGE_DIM)
        self.bn2 = nn.BatchNorm1d(h)
        self.pool = AttentionalAggregation(
            nn.Sequential(nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1)))
        self.sigma_head = make_sigma_head(h + HOTSPOT_DIM, h)
        self.A_g = nn.Parameter(torch.tensor(A_INIT))
        self.m_g = nn.Parameter(torch.tensor(M_INIT))
    def forward(self, batch_data):
        batch = Batch.from_data_list([sp['graph'] for sp in batch_data]).to(device)
        ea = batch.edge_attr
        x = self.drop(F.leaky_relu(self.bn1(self.conv1(batch.x, batch.edge_index, ea))))
        x = self.drop(F.leaky_relu(self.bn2(self.conv2(x, batch.edge_index, ea))))
        z = self.pool(x, batch.batch)
        hs = torch.stack([sp['hotspot_desc'] for sp in batch_data]).to(device)
        return basquin_forward(self.sigma_head, self.A_g, self.m_g, torch.cat([z, hs], 1))
    def compute_loss(self, preds, targets, info):
        return F.smooth_l1_loss(preds, targets)


class Model_G4(nn.Module):
    VARIANT, DESCRIPTION = 'G-4', 'Mean pooling (no attentional aggregation)'
    def __init__(self):
        super().__init__()
        h = HP['HIDDEN_DIM']; self.drop = nn.Dropout(HP['DROPOUT'])
        self.conv1 = GATv2Conv(FEAT_DIM, h, heads=HP['HEADS'], edge_dim=None)
        self.bn1 = nn.BatchNorm1d(h * HP['HEADS'])
        self.conv2 = GATv2Conv(h * HP['HEADS'], h, heads=1, edge_dim=None)
        self.bn2 = nn.BatchNorm1d(h)
        self.sigma_head = make_sigma_head(h + HOTSPOT_DIM, h)
        self.A_g = nn.Parameter(torch.tensor(A_INIT))
        self.m_g = nn.Parameter(torch.tensor(M_INIT))
    def forward(self, batch_data):
        batch = Batch.from_data_list([sp['graph'] for sp in batch_data]).to(device)
        x = self.drop(F.leaky_relu(self.bn1(self.conv1(batch.x, batch.edge_index))))
        x = self.drop(F.leaky_relu(self.bn2(self.conv2(x, batch.edge_index))))
        z = global_mean_pool(x, batch.batch)
        hs = torch.stack([sp['hotspot_desc'] for sp in batch_data]).to(device)
        return basquin_forward(self.sigma_head, self.A_g, self.m_g, torch.cat([z, hs], 1))
    def compute_loss(self, preds, targets, info):
        return F.smooth_l1_loss(preds, targets)


class Model_G5(nn.Module):
    VARIANT, DESCRIPTION = 'G-5', '1-layer GATv2 (1-hop)'
    def __init__(self):
        super().__init__()
        h = HP['HIDDEN_DIM']; self.drop = nn.Dropout(HP['DROPOUT'])
        self.conv1 = GATv2Conv(FEAT_DIM, h, heads=HP['HEADS'], edge_dim=None)
        self.bn1 = nn.BatchNorm1d(h * HP['HEADS'])
        self.proj = nn.Linear(h * HP['HEADS'], h); self.bn_proj = nn.BatchNorm1d(h)
        self.pool = AttentionalAggregation(
            nn.Sequential(nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1)))
        self.sigma_head = make_sigma_head(h + HOTSPOT_DIM, h)
        self.A_g = nn.Parameter(torch.tensor(A_INIT))
        self.m_g = nn.Parameter(torch.tensor(M_INIT))
    def forward(self, batch_data):
        batch = Batch.from_data_list([sp['graph'] for sp in batch_data]).to(device)
        x = self.drop(F.leaky_relu(self.bn1(self.conv1(batch.x, batch.edge_index))))
        x = self.drop(F.leaky_relu(self.bn_proj(self.proj(x))))
        z = self.pool(x, batch.batch)
        hs = torch.stack([sp['hotspot_desc'] for sp in batch_data]).to(device)
        return basquin_forward(self.sigma_head, self.A_g, self.m_g, torch.cat([z, hs], 1))
    def compute_loss(self, preds, targets, info):
        return F.smooth_l1_loss(preds, targets)


class Model_G6(nn.Module):
    VARIANT, DESCRIPTION = 'G-6', '3-layer GATv2 (3-hop)'
    def __init__(self):
        super().__init__()
        h = HP['HIDDEN_DIM']; self.drop = nn.Dropout(HP['DROPOUT'])
        self.conv1 = GATv2Conv(FEAT_DIM, h, heads=HP['HEADS'], edge_dim=None)
        self.bn1 = nn.BatchNorm1d(h * HP['HEADS'])
        self.conv2 = GATv2Conv(h * HP['HEADS'], h, heads=HP['HEADS'], edge_dim=None)
        self.bn2 = nn.BatchNorm1d(h * HP['HEADS'])
        self.conv3 = GATv2Conv(h * HP['HEADS'], h, heads=1, edge_dim=None)
        self.bn3 = nn.BatchNorm1d(h)
        self.pool = AttentionalAggregation(
            nn.Sequential(nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1)))
        self.sigma_head = make_sigma_head(h + HOTSPOT_DIM, h)
        self.A_g = nn.Parameter(torch.tensor(A_INIT))
        self.m_g = nn.Parameter(torch.tensor(M_INIT))
    def forward(self, batch_data):
        batch = Batch.from_data_list([sp['graph'] for sp in batch_data]).to(device)
        x = self.drop(F.leaky_relu(self.bn1(self.conv1(batch.x, batch.edge_index))))
        x = self.drop(F.leaky_relu(self.bn2(self.conv2(x, batch.edge_index))))
        x = self.drop(F.leaky_relu(self.bn3(self.conv3(x, batch.edge_index))))
        z = self.pool(x, batch.batch)
        hs = torch.stack([sp['hotspot_desc'] for sp in batch_data]).to(device)
        return basquin_forward(self.sigma_head, self.A_g, self.m_g, torch.cat([z, hs], 1))
    def compute_loss(self, preds, targets, info):
        return F.smooth_l1_loss(preds, targets)


# ============================================================ #
# §5.1.3 I 系列模型 (共用 Model_Standard, 通过配置区分)
# ============================================================ #
class Model_Standard(nn.Module):
    """V11 标准架构, 可配置 feat_dim 和 use_hotspot."""
    VARIANT, DESCRIPTION = 'I-std', 'Standard V11 (configurable)'
    def __init__(self, feat_dim=FEAT_DIM, use_hotspot=True):
        super().__init__()
        self.use_hotspot = use_hotspot
        h = HP['HIDDEN_DIM']; self.drop = nn.Dropout(HP['DROPOUT'])
        self.conv1 = GATv2Conv(feat_dim, h, heads=HP['HEADS'], edge_dim=None)
        self.bn1 = nn.BatchNorm1d(h * HP['HEADS'])
        self.conv2 = GATv2Conv(h * HP['HEADS'], h, heads=1, edge_dim=None)
        self.bn2 = nn.BatchNorm1d(h)
        self.pool = AttentionalAggregation(
            nn.Sequential(nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1)))
        in_dim = h + (HOTSPOT_DIM if use_hotspot else 0)
        self.sigma_head = make_sigma_head(in_dim, h)
        self.A_g = nn.Parameter(torch.tensor(A_INIT))
        self.m_g = nn.Parameter(torch.tensor(M_INIT))
    def forward(self, batch_data):
        batch = Batch.from_data_list([sp['graph'] for sp in batch_data]).to(device)
        x = self.drop(F.leaky_relu(self.bn1(self.conv1(batch.x, batch.edge_index))))
        x = self.drop(F.leaky_relu(self.bn2(self.conv2(x, batch.edge_index))))
        z = self.pool(x, batch.batch)
        if self.use_hotspot:
            hs = torch.stack([sp['hotspot_desc'] for sp in batch_data]).to(device)
            zh = torch.cat([z, hs], dim=1)
        else:
            zh = z
        return basquin_forward(self.sigma_head, self.A_g, self.m_g, zh)
    def compute_loss(self, preds, targets, info):
        return F.smooth_l1_loss(preds, targets)


# ============================================================ #
# 统一变体注册表
# ============================================================ #
VARIANT_REGISTRY = {
    # §5.1.1 物理信息注入
    'P-A1': {'description': Model_PA1.DESCRIPTION, 'model_fn': Model_PA1,
             'aug_config': AUG_CONFIGS['baseline']},
    'P-A2': {'description': Model_PA2.DESCRIPTION, 'model_fn': Model_PA2,
             'aug_config': AUG_CONFIGS['baseline']},
    'P-B':  {'description': Model_PB.DESCRIPTION,  'model_fn': Model_PB,
             'aug_config': AUG_CONFIGS['baseline']},
    'P-C':  {'description': Model_PC.DESCRIPTION,  'model_fn': Model_PC,
             'aug_config': AUG_CONFIGS['baseline']},
    # §5.1.2 图结构
    'G-1':  {'description': Model_G1.DESCRIPTION, 'model_fn': Model_G1,
             'aug_config': AUG_CONFIGS['baseline']},
    'G-2':  {'description': Model_G2.DESCRIPTION, 'model_fn': Model_G2,
             'aug_config': AUG_CONFIGS['baseline']},
    'G-3':  {'description': Model_G3.DESCRIPTION, 'model_fn': Model_G3,
             'aug_config': AUG_CONFIGS['baseline'],
             'include_edge_attr': True},  # ★ G-3 反向消融需要边特征
    'G-4':  {'description': Model_G4.DESCRIPTION, 'model_fn': Model_G4,
             'aug_config': AUG_CONFIGS['baseline']},
    'G-5':  {'description': Model_G5.DESCRIPTION, 'model_fn': Model_G5,
             'aug_config': AUG_CONFIGS['baseline']},
    'G-6':  {'description': Model_G6.DESCRIPTION, 'model_fn': Model_G6,
             'aug_config': AUG_CONFIGS['baseline']},
    # §5.1.3 输入与增强
    'I-1':  {'description': 'No hotspot descriptors (GNN output only)',
             'model_fn': lambda: Model_Standard(feat_dim=FEAT_DIM, use_hotspot=False),
             'aug_config': AUG_CONFIGS['baseline']},
    'I-2':  {'description': 'No augmentation (114 raw samples only)',
             'model_fn': lambda: Model_Standard(feat_dim=FEAT_DIM, use_hotspot=True),
             'aug_config': AUG_CONFIGS['none']},
    'I-3':  {'description': 'Uniform augmentation (3x, noise=0.05)',
             'model_fn': lambda: Model_Standard(feat_dim=FEAT_DIM, use_hotspot=True),
             'aug_config': AUG_CONFIGS['uniform']},
    'I-4':  {'description': 'Node features: sigma1 only (feat_dim=1)',
             'model_fn': lambda: Model_Standard(feat_dim=1, use_hotspot=True),
             'aug_config': AUG_CONFIGS['baseline'],
             'feat_slice': slice(0, 1)},
}

ALL_VARIANTS = 'P-A2,P-B,P-C,G-1,G-2,G-3,G-4,I-1,I-2,I-3,I-4'


# ============================================================ #
# 评估
# ============================================================ #
def evaluate(model, val_data, bs):
    model.eval()
    vp, vt, v_ids, per_sp = [], [], [], []
    A_snap, m_snap = np.nan, np.nan
    with torch.no_grad():
        for s in range(0, len(val_data), bs):
            batch = val_data[s:s + bs]
            preds, info = model(batch)
            vp.extend(preds.cpu().numpy())
            vt.extend([sp['y'].item() for sp in batch])
            v_ids.extend([sp['ID'] for sp in batch])
            A_val = float(info['A_g'].cpu()) if 'A_g' in info else np.nan
            DS_vals = info['DS_per'].cpu().numpy() if 'DS_per' in info else np.full(len(batch), np.nan)
            for i, sp in enumerate(batch):
                per_sp.append({'ID': sp['ID'], 'A': A_val,
                               'DS_MPa': float(DS_vals[i]) if np.isfinite(DS_vals[i]) else np.nan})
            if np.isnan(A_snap) and 'A_g' in info:
                A_snap = A_val
                m_snap = float(info['m_g'].cpu()) if 'm_g' in info else np.nan
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
    return ({'MRE': mre, 'logMAE': logmae, 'R2_logN': r2_log,
             'R2_N': r2_N, 'RMSE_N': rmse_N, 'P2X_cov': p2x},
            vp, vt, v_ids, per_sp, A_snap, m_snap)


# ============================================================ #
# 统一 train_fold
# ============================================================ #
def train_fold(variant_cfg, train_raw, val_raw, seed, fold_i):
    set_seed(seed)
    t0 = time.time()
    feat_slice = variant_cfg.get('feat_slice', None)
    use_edge = variant_cfg.get('include_edge_attr', False)
    aug_config = variant_cfg['aug_config']

    train_aug = do_augmentation(train_raw, aug_config, use_edge)
    train_data = [preprocess(d, use_edge, feat_slice) for d in train_raw + train_aug]
    val_data = [preprocess(d, use_edge, feat_slice) for d in val_raw]

    n_raw, n_aug = len(train_raw), len(train_aug)
    print(f'    训练: {n_raw}原始+{n_aug}增强={n_raw+n_aug}')

    model_fn = variant_cfg['model_fn']
    model = (model_fn() if callable(model_fn) and not isinstance(model_fn, type)
             else model_fn()).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=HP['LR'], weight_decay=HP['WEIGHT_DECAY'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=HP['EPOCHS'], eta_min=1e-5)
    es = EarlyStopping()
    ema = EMAModel(model, HP['EMA_DECAY'])
    bs = HP['BATCH_SIZE']
    best_ep, final_ep = 0, 0
    hist_rows = []

    for epoch in range(HP['EPOCHS']):
        model.train()
        idx = list(range(len(train_data))); random.shuffle(idx)
        ep_loss, nb = 0.0, 0
        for s in range(0, len(idx), bs):
            batch = [train_data[i] for i in idx[s:s + bs]]
            tgt = torch.tensor([sp['y'].item() for sp in batch], device=device)
            optimizer.zero_grad()
            preds, info = model(batch)
            loss = model.compute_loss(preds, tgt, info)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); ema.update(model)
            ep_loss += loss.item(); nb += 1
        scheduler.step()
        final_ep = epoch + 1
        avg_train = ep_loss / max(nb, 1)

        ema.apply(model)
        metrics, _, _, _, _, A_now, m_now = evaluate(model, val_data, bs)
        ema.restore(model)
        prev = es.best_score
        es(metrics['MRE'], ema)
        if es.best_score != prev:
            best_ep = final_ep

        hist_rows.append({
            'seed': seed, 'fold': fold_i, 'epoch': final_ep,
            'train_loss': avg_train, 'val_mre': metrics['MRE'],
            'val_logmae': metrics['logMAE'],
            'A_g': A_now if np.isfinite(A_now) else np.nan,
            'm_g': m_now if np.isfinite(m_now) else np.nan,
            'wait': es.counter,
        })

        if epoch % 50 == 0 or es.early_stop:
            print(f'    ep={final_ep:3d} | L={avg_train:.4f} | '
                  f'MRE={metrics["MRE"]:.4f} | '
                  f'A={A_now:.3f} m={m_now:.3f} | '
                  f'wait={es.counter}', flush=True)
        if es.early_stop:
            print(f'    早停 best_ep={best_ep}', flush=True)
            break

    if es.best_ema_state is not None:
        model.load_state_dict(es.best_ema_state); model.to(device)

    metrics, vp, vt, v_ids, per_sp, A_final, m_final = evaluate(model, val_data, bs)
    time_s = time.time() - t0

    fold_row = {
        **metrics,
        'A': A_final, 'm': m_final,
        'A_drift': (A_final - A_INIT) if np.isfinite(A_final) else np.nan,
        'm_drift': (m_final - M_INIT) if np.isfinite(m_final) else np.nan,
        'best_ep': best_ep, 'final_ep': final_ep,
        'wait_after_best': final_ep - best_ep,
        'time_s': time_s, 'n_params': n_params,
        'n_train_raw': n_raw, 'n_train_aug': n_aug,
    }
    id_to_meta = {sp['ID']: (sp['joint_type'], sp['corrosion_hours']) for sp in val_data}
    spec_rows = []
    for i in range(len(vp)):
        true_N = float(10.0 ** vt[i]); pred_N = float(10.0 ** vp[i])
        jt, k_h = id_to_meta.get(v_ids[i], ('UNKNOWN', -1))
        spec_rows.append({
            'seed': seed, 'fold': fold_i, 'ID': v_ids[i],
            'joint_type': jt, 'corrosion_hours': k_h,
            'true_N': true_N, 'pred_N': pred_N,
            'rel_error': abs(pred_N - true_N) / (true_N + 1e-8),
            'pred_log10N': float(vp[i]), 'true_log10N': float(vt[i]),
            'A': A_final,
            'DS_MPa': per_sp[i]['DS_MPa'] if per_sp[i]['DS_MPa'] is not None else np.nan,
        })
    return fold_row, spec_rows, hist_rows


# ============================================================ #
# 统一 run_variant
# ============================================================ #
def run_variant(variant_name, variant_cfg, failed, cv_splits, out_dir):
    var_dir = os.path.join(out_dir, variant_name)
    os.makedirs(var_dir, exist_ok=True)

    desc = variant_cfg['description']
    print(f'\n{"="*60}')
    print(f'  变体: {variant_name} — {desc}')
    print(f'  输出: {var_dir}')
    print(f'{"="*60}')

    all_fold, all_spec, all_hist = [], [], []
    for seed in ABLATION_SEEDS:
        print(f'\n  Seed {seed}:')
        for fold_i in range(N_SPLITS):
            key = f's{seed}_f{fold_i}'
            if key not in cv_splits: continue
            ti = cv_splits[key]['train_idx']
            vi = cv_splits[key]['val_idx']
            tf = [failed[i] for i in ti]; vf = [failed[i] for i in vi]
            print(f'  --- s={seed} f={fold_i} ---')
            fr, sr, hr = train_fold(variant_cfg, tf, vf, seed, fold_i)
            fr['seed'] = seed; fr['fold'] = fold_i; fr['variant'] = variant_name
            all_fold.append(fr); all_spec.extend(sr); all_hist.extend(hr)
            print(f'    >> MRE={fr["MRE"]:.4f} R²={fr["R2_logN"]:.3f} '
                  f'A={fr["A"]:.3f} m={fr["m"]:.3f} | {fr["time_s"]:.0f}s')
            pd.DataFrame(all_fold).to_csv(os.path.join(var_dir, 'results.csv'), index=False)
            pd.DataFrame(all_spec).to_csv(os.path.join(var_dir, 'specimen_preds.csv'), index=False)

    df_fold = pd.DataFrame(all_fold)
    df_sp = pd.DataFrame(all_spec)
    pd.DataFrame(all_hist).to_csv(os.path.join(var_dir, 'training_history.csv'), index=False)

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
        'variant': variant_name, 'description': desc,
        'seeds': ABLATION_SEEDS, 'n_folds': N_SPLITS,
        'n_total_folds': len(df_fold),
        'HP': {k: (list(v) if isinstance(v, tuple) else v) for k, v in HP.items()},
        'A_INIT': A_INIT, 'M_INIT': M_INIT,
        'fold_MRE_mean': float(df_fold['MRE'].mean()),
        'fold_MRE_std':  float(df_fold['MRE'].std()),
        'fold_logMAE_mean': float(df_fold['logMAE'].mean()),
        'fold_R2_logN_mean': float(df_fold['R2_logN'].mean()),
        'fold_R2_N_mean': float(df_fold['R2_N'].mean()),
        'fold_P2X_mean': float(df_fold['P2X_cov'].mean()),
        'fold_A_mean': float(df_fold['A'].mean()) if df_fold['A'].notna().any() else None,
        'fold_A_std':  float(df_fold['A'].std()) if df_fold['A'].notna().any() else None,
        'fold_m_mean': float(df_fold['m'].mean()) if df_fold['m'].notna().any() else None,
        'fold_m_std':  float(df_fold['m'].std()) if df_fold['m'].notna().any() else None,
        'fold_best_ep_median': int(df_fold['best_ep'].median()),
        'ensemble_MRE': ens_mre, 'ensemble_R2_logN': ens_r2,
        'ensemble_R2_N': ens_r2_N, 'ensemble_P2X': ens_p2x,
        'n_params': int(df_fold['n_params'].iloc[0]),
        'avg_n_train_raw': int(df_fold['n_train_raw'].mean()),
        'avg_n_train_aug': int(df_fold['n_train_aug'].mean()),
    }
    with open(os.path.join(var_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f'\n  {variant_name} 汇总:')
    print(f'    fold MRE:  {meta["fold_MRE_mean"]:.4f} ± {meta["fold_MRE_std"]:.4f}')
    print(f'    ensemble:  MRE={ens_mre:.4f}  R²_logN={ens_r2:.4f}  P_2X={ens_p2x:.4f}')
    if meta['fold_A_mean'] is not None:
        print(f'    A={meta["fold_A_mean"]:.3f}±{meta["fold_A_std"]:.3f}  '
              f'm={meta["fold_m_mean"]:.3f}±{meta["fold_m_std"]:.3f}')
    print(f'    params={meta["n_params"]}  raw={meta["avg_n_train_raw"]}  aug={meta["avg_n_train_aug"]}')
    return meta


# ============================================================ #
# 主入口
# ============================================================ #
def main():
    ap = argparse.ArgumentParser(description='§5.1 消融实验 (P + G + I 合并)')
    ap.add_argument('--cv_splits', default='./v11_final/cv_splits.json')
    ap.add_argument('--out_dir', default='./ablation_5_1')
    ap.add_argument('--variants', default=ALL_VARIANTS,
                    help=f'逗号分隔, 可选: {ALL_VARIANTS},G-5,G-6')
    args = ap.parse_args()

    print(f'Device: {device}')
    print(f'HP: H={HP["HIDDEN_DIM"]} heads={HP["HEADS"]} D={HP["DROPOUT"]} '
          f'LR={HP["LR"]} WD={HP["WEIGHT_DECAY"]} BS={HP["BATCH_SIZE"]}')

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

    variant_names = [v.strip() for v in args.variants.split(',')]
    print(f'变体: {variant_names}\n')

    all_meta = []
    for vn in variant_names:
        if vn not in VARIANT_REGISTRY:
            print(f'⚠️ 未知变体 {vn}, 跳过'); continue
        meta = run_variant(vn, VARIANT_REGISTRY[vn], failed, cv_splits, args.out_dir)
        all_meta.append(meta)

    # 汇总对比表
    if len(all_meta) > 1:
        summary_rows = []
        for m in all_meta:
            summary_rows.append({
                'variant': m['variant'], 'description': m['description'],
                'n_params': m['n_params'],
                'n_train_raw': m['avg_n_train_raw'], 'n_train_aug': m['avg_n_train_aug'],
                'fold_MRE_mean': m['fold_MRE_mean'], 'fold_MRE_std': m['fold_MRE_std'],
                'fold_R2_logN': m['fold_R2_logN_mean'],
                'fold_R2_N': m['fold_R2_N_mean'],
                'fold_P2X': m['fold_P2X_mean'],
                'ensemble_MRE': m['ensemble_MRE'],
                'ensemble_R2_logN': m['ensemble_R2_logN'],
                'ensemble_R2_N': m['ensemble_R2_N'],
                'ensemble_P2X': m['ensemble_P2X'],
                'A_mean': m.get('fold_A_mean'), 'm_mean': m.get('fold_m_mean'),
                'best_ep_median': m['fold_best_ep_median'],
            })
        df_sum = pd.DataFrame(summary_rows)
        sum_path = os.path.join(args.out_dir, 'ablation_5.1_summary.csv')
        df_sum.to_csv(sum_path, index=False)
        print(f'\n{"="*75}')
        print(f'  §5.1 消融汇总 → {sum_path}')
        print(f'{"="*75}')
        print(f'  {"Var":<6} {"MRE±std":>14} {"R²_logN":>8} {"P_2X":>6} '
              f'{"ens_MRE":>8} {"A":>8} {"m":>6} {"params":>8}')
        print('-' * 75)
        for _, r in df_sum.iterrows():
            A_str = f'{r["A_mean"]:.3f}' if pd.notna(r['A_mean']) else '  N/A'
            m_str = f'{r["m_mean"]:.3f}' if pd.notna(r['m_mean']) else ' N/A'
            print(f'  {r["variant"]:<6} '
                  f'{r["fold_MRE_mean"]:.4f}±{r["fold_MRE_std"]:.4f} '
                  f'{r["fold_R2_logN"]:8.4f} {r["fold_P2X"]:6.3f} '
                  f'{r["ensemble_MRE"]:8.4f} {A_str:>8} {m_str:>6} '
                  f'{r["n_params"]:8d}')
        print(f'{"="*75}')

    print(f'\n完成! 输出: {args.out_dir}/')
    for vn in variant_names:
        if vn in VARIANT_REGISTRY:
            print(f'  {vn}/  — results.csv, specimen_preds.csv, specimen_ensemble.csv, '
                  f'training_history.csv, meta.json')
    if len(all_meta) > 1:
        print(f'  ablation_5.1_summary.csv — 全部变体对比汇总表')


if __name__ == '__main__':
    main()
