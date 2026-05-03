# -*- coding: utf-8 -*-
"""
共享配置: 数据/路径/物理常数
训练超参数已迁出到 3_train_v9.py 内的 HP dict.
"""
import re

# ==================== 路径 ==================== #
LIFE_TABLE_PATH = './KL_N.xlsx'
RAW_DATA_DIR    = './toe_raw_csv'
TOP_N           = 500
DATA_DIR        = f'./toe_sdv_tables/sigma1_top{TOP_N}'
GRAPH_DIR       = './single_level_graphs'
RESULT_DIR      = './single_level_results_no_corr_a7_a7_V4_clean_a7_V4_no_hotspot_a7_V4_uniform_aug_a7_V4_noDelta_a7_V4_noDelta_a7_v10_final'

# ==================== 目标空间 ==================== #
USE_LOG10 = True
AUG_PKL_PATH = './single_level_aug_log10.pkl' if USE_LOG10 else './single_level_aug.pkl'

# ==================== Basquin 物理常数 ==================== #
FAT_INIT = {'DJ': 80, 'TX': 80, 'UL': 80}
BASQUIN_M = 3.0

# ==================== 主表 ==================== #
COL_ID   = 'ID'
COL_TOE  = 'KL'
COL_LIFE = 'N'

def parse_id(specimen_id):
    m = re.match(r'^([A-Z]+)(\d+)-\d+$', specimen_id)
    return (m.group(1), int(m.group(2))) if m else ('UNKNOWN', 0)

# ==================== 节点/边特征 ==================== #
RAW_STRESS_COLS = ['S11_peak', 'S22_peak', 'S33_peak',
                   'S12_peak', 'S13_peak', 'S23_peak']
COORD_COLS = ['X', 'Y', 'Z']

DERIVED_FEATURE_NAMES = ['sigma1', 'tau_max', 'sigma_vm']
FEAT_DIM = len(DERIVED_FEATURE_NAMES)   # 3
STRESS_PREPROCESS = 'log10'

K_NEIGHBORS = 8
EDGE_DIM = 3   # [dx, dy, dz] 有向差值

# ==================== 热点场描述符 ==================== #
HOTSPOT_DIM  = 6
HOTSPOT_KEYS = ['F_concentration', 'F_volume', 'F_gradient',
                'F_radius', 'F_n_hotspots', 'F_separation']
