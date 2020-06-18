from functools import partial
from itertools import product
from pathlib import Path
import multiprocessing
from pprint import pprint

import numpy as np
import pandas as pd

from reducto.codec import get_video_size
from reducto.data_loader import load_evaluation, load_diff_vector, load_diff_result, load_json, load_inference, \
    dump_json
from reducto.evaluator import MetricComposer
from reducto.hashbuilder import HashBuilder, ThreshMap
from reducto.utils import flatten, show_stats

_target_acc = 0.90
_tinyyolo_acc = 0.60
_datasets = [
    {
        'dataset': 'auburn',
        'subsets': ['fri005', 'sat001am', 'sat005pm'],
        'glimpse_subsets': [
            'fri000', 'fri001', 'fri003', 'fri007', 'fri009', 'fri011',
            'fri013', 'fri015', 'fri019', 'sat000pm', 'sat001pm', 'sat002pm',
            'sat006pm'
        ],
        'queries': [{
            'metric': 'mAP-2', 'differ': 'edge',
            'distance': 0.3, 'safe': 0.025, 'target_acc': _target_acc,  # reducto
            'send_thresh': 0.9601, 'tinyyolo_acc': _tinyyolo_acc,  # focus
            'glimpse_best': 0.0006700584274113686,  # glimpse
            'glimpse_offline': 0.0006700584274113686,
            'glimpse_all': 0.0006700584274113686,
            'glimpse_25th_best': 0.0006700584274113686,
            'glimpse_25th_offline': 0.0013401168548227373,
            'glimpse_25th_all': 0.0006700584274113686,
        }],
        'properties': {'fps': 30},
    },
    {
        'dataset': 'jacksonhole',
        'subsets': ['raw000', 'raw019', 'raw038'],
        'glimpse_subsets': [
            'raw001', 'raw002', 'raw003', 'raw004', 'raw005', 'raw006',
            'raw007', 'raw008', 'raw009', 'raw010', 'raw011', 'raw012',
            'raw013', 'raw014', 'raw015', 'raw016', 'raw017', 'raw018',
        ],
        'queries': [{
            'metric': 'mAP-2', 'differ': 'edge',
            'distance': 1.9, 'safe': 0.035, 'target_acc': _target_acc,
            'send_thresh': 0.1236, 'tinyyolo_acc': _tinyyolo_acc,
            'glimpse_best': 0.015458182859056516,
            'glimpse_offline': 0.047064646026234565,
            'glimpse_all': 0.07701487531565657,
            'glimpse_25th_best': 0.016562338777560554,
            'glimpse_25th_offline': 0.047064646026234565,
            'glimpse_25th_all': 0.04278604184203143,
        }],
        'properties': {'fps': 30},
    },
    {
        'dataset': 'lagrange',
        'subsets': ['raw000', 'raw005', 'raw022'],
        'glimpse_subsets': [
            'raw001', 'raw002', 'raw003', 'raw004', 'raw006', 'raw007',
            'raw009', 'raw010', 'raw011', 'raw012', 'raw018', 'raw019',
            'raw020', 'raw021', 'raw023',
        ],
        'queries': [{
            'metric': 'mAP-2', 'differ': 'edge',
            'distance': 1.9, 'safe': 0.045, 'target_acc': _target_acc,
            'send_thresh': 0.1284, 'tinyyolo_acc': _tinyyolo_acc,
            'glimpse_best': 0.007982309759177138,
            'glimpse_offline': 0.005041458795269771,
            'glimpse_all': 0.006301823494087214,
            'glimpse_25th_best': 0.006721945060359695,
            'glimpse_25th_offline': 0.007142066626632176,
            'glimpse_25th_all': 0.006301823494087214,
        }],
        'properties': {'fps': 30},
    },
    {
        'dataset': 'southampton',
        'subsets': ['raw001', 'sat005', 'sat009pm'],
        'glimpse_subsets': [
            'raw000', 'raw008', 'raw013', 'raw019', 'raw031', 'raw038',
            'raw042', 'raw043', 'raw049', 'sat000', 'sat000pm', 'sat001pm',
        ],
        'queries': [{
            'metric': 'mAP-2', 'differ': 'edge',
            'distance': 1.5, 'safe': 0.02, 'target_acc': _target_acc,
            'send_thresh': 0.1241, 'tinyyolo_acc': _tinyyolo_acc,
            'glimpse_best': 0.014329868626743628,
            'glimpse_offline': 0.01891542658730159,
            'glimpse_all': 0.015476258116883117,
            'glimpse_25th_best': 0.008024726430976431,
            'glimpse_25th_offline': 0.01662264760702261,
            'glimpse_25th_all': 0.013756673881673884,
        }],
        'properties': {'fps': 30},
    }
]

# network = {'bandwidth': 12, 'rtt': 100}
# network = {'bandwidth': 24, 'rtt': 20}
# network = {'bandwidth': 60, 'rtt': 5}

full_datasets = [
    {
        'dataset': 'auburn',
        'subsets': [
            'fri000', 'fri001', 'fri005',
            # 'fri003', 'fri007', 'fri009', 'fri011',

            # 'fri013', 'fri015', 'fri019', 'sat001am',
            # 'sat000pm', 'sat001pm', 'sat002pm', 'sat005pm', 'sat006pm',
        ],
        'small_subsets': ['fri005', 'sat001am', 'sat005pm'],
        'glimpse_queries_90': [
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.05, 'target_acc': 0.800},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.10, 'target_acc': 0.805},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.15, 'target_acc': 0.810},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.20, 'target_acc': 0.815},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.25, 'target_acc': 0.820},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.30, 'target_acc': 0.825},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.35, 'target_acc': 0.830},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.40, 'target_acc': 0.835},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.45, 'target_acc': 0.840},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.50, 'target_acc': 0.845},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.55, 'target_acc': 0.850},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.60, 'target_acc': 0.855},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.65, 'target_acc': 0.860},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.70, 'target_acc': 0.865},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.75, 'target_acc': 0.900},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.80, 'target_acc': 0.900},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.85, 'target_acc': 0.900},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.90, 'target_acc': 0.900},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.95, 'target_acc': 0.900},
        ],
        'optimal_queries_90': [{'target_acc': 0.90}],
        'optimal_queries_80': [{'target_acc': 0.80}],
        'optimal_queries_70': [{'target_acc': 0.70}],
        'reducto_queries_90': [
            {'metric': 'mAP-2', 'differ': 'edge', 'distance': 0.3, 'safe': 0.025, 'target_acc': 0.90},
            {'metric': 'mAP-0', 'differ': 'edge', 'distance': 2.0, 'safe': 0.000, 'target_acc': 0.90},
            {'metric': 'counting-2', 'differ': 'area', 'distance': 4.0, 'safe': -0.500, 'target_acc': 0.90},
            {'metric': 'counting-0', 'differ': 'area', 'distance': 4.0, 'safe': -0.025, 'target_acc': 0.90},
            {'metric': 'tagging-2', 'differ': 'area', 'distance': 4.0, 'safe': -0.350, 'target_acc': 0.90},
            {'metric': 'tagging-0', 'differ': 'area', 'distance': 4.0, 'safe': -0.350, 'target_acc': 0.90},
        ],
        'reducto_queries_80': [
            {'metric': 'mAP-2', 'differ': 'edge', 'distance': 2.0, 'safe': 0.073, 'target_acc': 0.80},
            {'metric': 'mAP-0', 'differ': 'edge', 'distance': 2.0, 'safe': 0.000, 'target_acc': 0.80},
            {'metric': 'counting-2', 'differ': 'area', 'distance': 4.0, 'safe': -0.350, 'target_acc': 0.80},
            {'metric': 'counting-0', 'differ': 'area', 'distance': 4.0, 'safe': -0.350, 'target_acc': 0.80},
            {'metric': 'tagging-2', 'differ': 'area', 'distance': 4.0, 'safe': -0.350, 'target_acc': 0.80},
            {'metric': 'tagging-0', 'differ': 'area', 'distance': 4.0, 'safe': -0.350, 'target_acc': 0.80},
        ],
        'reducto_queries_70': [
            {'metric': 'mAP-2', 'differ': 'edge', 'distance': 1.2, 'safe': 0.080, 'target_acc': 0.70},
            {'metric': 'mAP-0', 'differ': 'edge', 'distance': 2.0, 'safe': 0.000, 'target_acc': 0.70},
            {'metric': 'counting-2', 'differ': 'area', 'distance': 4.0, 'safe': -0.350, 'target_acc': 0.70},
            {'metric': 'counting-0', 'differ': 'area', 'distance': 4.0, 'safe': -0.350, 'target_acc': 0.70},
            {'metric': 'tagging-2', 'differ': 'area', 'distance': 4.0, 'safe': -0.350, 'target_acc': 0.70},
            {'metric': 'tagging-0', 'differ': 'area', 'distance': 4.0, 'safe': -0.350, 'target_acc': 0.70},
        ],
        'properties': {'fps': 30},
    },
    {
        'dataset': 'jacksonhole',
        'subsets': [
            'raw000', 'raw001', 'raw002',
            # 'raw003', 'raw004', 'raw005',
            # 'raw006', 'raw007', 'raw008', 'raw009', 'raw010', 'raw011',

            # 'raw012', 'raw013', 'raw014', 'raw015', 'raw016', 'raw017',
            # 'raw018', 'raw019',
            # 'raw038',
        ],
        'small_subsets': ['raw000', 'raw019', 'raw038'],
        'glimpse_queries_90': [
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.05, 'target_acc': 0.800},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.10, 'target_acc': 0.805},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.15, 'target_acc': 0.810},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.20, 'target_acc': 0.815},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.25, 'target_acc': 0.820},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.30, 'target_acc': 0.825},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.35, 'target_acc': 0.830},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.40, 'target_acc': 0.835},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.45, 'target_acc': 0.840},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.50, 'target_acc': 0.845},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.55, 'target_acc': 0.850},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.60, 'target_acc': 0.855},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.65, 'target_acc': 0.860},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.70, 'target_acc': 0.865},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.75, 'target_acc': 0.900},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.80, 'target_acc': 0.900},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.85, 'target_acc': 0.900},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.90, 'target_acc': 0.900},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.95, 'target_acc': 0.900},
        ],
        'optimal_queries_90': [{'target_acc': 0.90}],
        'optimal_queries_80': [{'target_acc': 0.80}],
        'optimal_queries_70': [{'target_acc': 0.70}],
        'reducto_queries_90': [
            {'metric': 'mAP-2', 'differ': 'edge', 'distance': 1.9, 'safe': 0.035, 'target_acc': 0.90},
            {'metric': 'mAP-0', 'differ': 'edge', 'distance': 1.6, 'safe': -0.020, 'target_acc': 0.90},
            {'metric': 'counting-2', 'differ': 'area', 'distance': 4.0, 'safe': -0.050, 'target_acc': 0.90},
            {'metric': 'counting-0', 'differ': 'area', 'distance': 4.0, 'safe': 0.010, 'target_acc': 0.90},
            {'metric': 'tagging-2', 'differ': 'area', 'distance': 4.0, 'safe': -0.500, 'target_acc': 0.90},
            {'metric': 'tagging-0', 'differ': 'area', 'distance': 4.0, 'safe': 0.000, 'target_acc': 0.90},
        ],
        'reducto_queries_80': [
            {'metric': 'mAP-2', 'differ': 'edge', 'distance': 1.7, 'safe': 0.08, 'target_acc': 0.80},
            {'metric': 'mAP-0', 'differ': 'edge', 'distance': 1.6, 'safe': -0.025, 'target_acc': 0.80},
            {'metric': 'counting-2', 'differ': 'area', 'distance': 4.0, 'safe': -0.050, 'target_acc': 0.80},
            {'metric': 'counting-0', 'differ': 'area', 'distance': 4.0, 'safe': -0.050, 'target_acc': 0.80},
            {'metric': 'tagging-2', 'differ': 'area', 'distance': 4.0, 'safe': -0.500, 'target_acc': 0.80},
            {'metric': 'tagging-0', 'differ': 'area', 'distance': 4.0, 'safe': 0.000, 'target_acc': 0.80},
        ],
        'reducto_queries_70': [
            {'metric': 'mAP-2', 'differ': 'edge', 'distance': 2.0, 'safe': -0.150, 'target_acc': 0.70},
            {'metric': 'mAP-0', 'differ': 'edge', 'distance': 1.6, 'safe': -0.040, 'target_acc': 0.70},
            {'metric': 'counting-2', 'differ': 'area', 'distance': 4.0, 'safe': -0.050, 'target_acc': 0.70},
            {'metric': 'counting-0', 'differ': 'area', 'distance': 4.0, 'safe': -0.050, 'target_acc': 0.70},
            {'metric': 'tagging-2', 'differ': 'area', 'distance': 4.0, 'safe': -0.500, 'target_acc': 0.70},
            {'metric': 'tagging-0', 'differ': 'area', 'distance': 4.0, 'safe': 0.000, 'target_acc': 0.70},
        ],
        'properties': {'fps': 30},
    },
    {
        'dataset': 'lagrange',
        'subsets': [
            'raw000',
            'raw001', 'raw002', 'raw003',
            # 'raw004', 'raw005', 'raw006', 'raw007',

            # 'raw009', 'raw010', 'raw011', 'raw012', 'raw018', 'raw019',
            # 'raw020', 'raw021', 'raw022', 'raw023',
        ],
        'small_subsets': ['raw000', 'raw005', 'raw022'],
        'glimpse_queries_90': [
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.05, 'target_acc': 0.800},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.10, 'target_acc': 0.805},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.15, 'target_acc': 0.810},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.20, 'target_acc': 0.815},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.25, 'target_acc': 0.820},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.30, 'target_acc': 0.825},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.35, 'target_acc': 0.830},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.40, 'target_acc': 0.835},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.45, 'target_acc': 0.840},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.50, 'target_acc': 0.845},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.55, 'target_acc': 0.850},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.60, 'target_acc': 0.855},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.65, 'target_acc': 0.860},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.70, 'target_acc': 0.865},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.75, 'target_acc': 0.900},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.80, 'target_acc': 0.900},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.85, 'target_acc': 0.900},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.90, 'target_acc': 0.900},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.95, 'target_acc': 0.900},
        ],
        'optimal_queries_90': [{'target_acc': 0.90}],
        'optimal_queries_80': [{'target_acc': 0.80}],
        'optimal_queries_70': [{'target_acc': 0.70}],
        'reducto_queries_90': [
            {'metric': 'mAP-2', 'differ': 'edge', 'distance': 1.9, 'safe': 0.045, 'target_acc': 0.90},
            {'metric': 'mAP-0', 'differ': 'edge', 'distance': 1.6, 'safe': 0.000, 'target_acc': 0.90},
            {'metric': 'counting-2', 'differ': 'area', 'distance': 1.3, 'safe': 0.070, 'target_acc': 0.90},
            {'metric': 'counting-0', 'differ': 'area', 'distance': 1.5, 'safe': -0.060, 'target_acc': 0.90},
            {'metric': 'tagging-2', 'differ': 'area', 'distance': 1.5, 'safe': -0.000, 'target_acc': 0.90},
            {'metric': 'tagging-0', 'differ': 'area', 'distance': 1.5, 'safe': -0.000, 'target_acc': 0.90},
        ],
        'reducto_queries_80': [
            {'metric': 'mAP-2', 'differ': 'edge', 'distance': 1.7, 'safe': 0.109, 'target_acc': 0.80},
            {'metric': 'mAP-0', 'differ': 'edge', 'distance': 1.7, 'safe': 0.000, 'target_acc': 0.80},
            {'metric': 'counting-2', 'differ': 'area', 'distance': 1.0, 'safe': 0.142, 'target_acc': 0.80},
            {'metric': 'counting-0', 'differ': 'area', 'distance': 1.5, 'safe': 0.000, 'target_acc': 0.80},
            {'metric': 'tagging-2', 'differ': 'area', 'distance': ..., 'safe': ..., 'target_acc': 0.80},
            {'metric': 'tagging-0', 'differ': 'area', 'distance': ..., 'safe': ..., 'target_acc': 0.80},
        ],
        'reducto_queries_70': [
            {'metric': 'mAP-2', 'differ': 'edge', 'distance': 0.7, 'safe': 0.095, 'target_acc': 0.70},
            {'metric': 'mAP-0', 'differ': 'edge', 'distance': 1.7, 'safe': 0.000, 'target_acc': 0.70},
            {'metric': 'counting-2', 'differ': 'area', 'distance': 4.0, 'safe': 0.000, 'target_acc': 0.70},
            {'metric': 'counting-0', 'differ': 'area', 'distance': 1.5, 'safe': 0.000, 'target_acc': 0.70},
            {'metric': 'tagging-2', 'differ': 'area', 'distance': ..., 'safe': ..., 'target_acc': 0.70},
            {'metric': 'tagging-0', 'differ': 'area', 'distance': ..., 'safe': ..., 'target_acc': 0.70},
        ],
        'properties': {'fps': 30},
    },
    {
        'dataset': 'southampton',
        'subsets': [
            'raw000', 'raw008', 'raw013',

            # 'raw019', 'raw031', 'raw038',
            # 'raw042', 'raw043', 'raw049',
            # 'sat000', 'sat000pm', 'sat001pm',
            # 'raw001', 'sat005', 'sat009pm',
        ],
        'small_subsets': ['raw000', 'sat005', 'sat009pm'],
        'glimpse_queries_90': [
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.05, 'target_acc': 0.800},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.10, 'target_acc': 0.805},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.15, 'target_acc': 0.810},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.20, 'target_acc': 0.815},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.25, 'target_acc': 0.820},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.30, 'target_acc': 0.825},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.35, 'target_acc': 0.830},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.40, 'target_acc': 0.835},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.45, 'target_acc': 0.840},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.50, 'target_acc': 0.845},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.55, 'target_acc': 0.850},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.60, 'target_acc': 0.855},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.65, 'target_acc': 0.860},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.70, 'target_acc': 0.865},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.75, 'target_acc': 0.900},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.80, 'target_acc': 0.900},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.85, 'target_acc': 0.900},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.90, 'target_acc': 0.900},
            {'metric': 'mAP-2', 'differ': 'edge', 'split': 0.95, 'target_acc': 0.900},
        ],
        'optimal_queries_90': [{'target_acc': 0.90}],
        'optimal_queries_80': [{'target_acc': 0.80}],
        'optimal_queries_70': [{'target_acc': 0.70}],
        'reducto_queries_90': [
            {'metric': 'mAP-2', 'differ': 'edge', 'distance': 1.5, 'safe': 0.010, 'target_acc': 0.90},
            {'metric': 'mAP-0', 'differ': 'edge', 'distance': 2.0, 'safe': 0.000, 'target_acc': 0.90},
            {'metric': 'counting-2', 'differ': 'area', 'distance': 1.3, 'safe': -0.100, 'target_acc': 0.90},
            {'metric': 'counting-0', 'differ': 'area', 'distance': 1.3, 'safe': -0.100, 'target_acc': 0.90},
            {'metric': 'tagging-2', 'differ': 'area', 'distance': 1.3, 'safe': -0.100, 'target_acc': 0.90},
            {'metric': 'tagging-0', 'differ': 'area', 'distance': 1.3, 'safe': -0.100, 'target_acc': 0.90},
        ],
        'reducto_queries_80': [
            {'metric': 'mAP-2', 'differ': 'edge', 'distance': 1.7, 'safe': 0.055, 'target_acc': 0.80},
            {'metric': 'mAP-0', 'differ': 'edge', 'distance': 2.0, 'safe': 0.000, 'target_acc': 0.80},
            {'metric': 'counting-2', 'differ': 'area', 'distance': ..., 'safe': ..., 'target_acc': 0.80},
            {'metric': 'counting-0', 'differ': 'area', 'distance': ..., 'safe': ..., 'target_acc': 0.80},
            {'metric': 'tagging-2', 'differ': 'area', 'distance': ..., 'safe': ..., 'target_acc': 0.80},
            {'metric': 'tagging-0', 'differ': 'area', 'distance': ..., 'safe': ..., 'target_acc': 0.80},
        ],
        'reducto_queries_70': [
            {'metric': 'mAP-2', 'differ': 'edge', 'distance': 1.7, 'safe': 0.04, 'target_acc': 0.70},
            {'metric': 'mAP-0', 'differ': 'edge', 'distance': 2.0, 'safe': 0.000, 'target_acc': 0.70},
            {'metric': 'counting-2', 'differ': 'area', 'distance': ..., 'safe': ..., 'target_acc': 0.70},
            {'metric': 'counting-0', 'differ': 'area', 'distance': ..., 'safe': ..., 'target_acc': 0.70},
            {'metric': 'tagging-2', 'differ': 'area', 'distance': ..., 'safe': ..., 'target_acc': 0.70},
            {'metric': 'tagging-0', 'differ': 'area', 'distance': ..., 'safe': ..., 'target_acc': 0.70},
        ],
        'properties': {'fps': 30},
    }
]


class Simulator:

    def __init__(self, datasets, **kwargs):
        self.datasets = datasets
        self.actual_size = kwargs.get('actual_size', 2113284)
        self.video_root = Path(kwargs.get('video_root', '/mnt/ssd2t/dataset'))
        self.root = Path(kwargs.get('result_root', 'data'))
        self.fps = kwargs.get('fps', 30)
        self.segment_duration = kwargs.get('segment_duration', 5.0)
        self.gpu_time = -1
        self.send_all = False
        self.name = 'none'
        self.network_logname = '!!!'

    def simulate(self, query_key='queries', network=None, network_name=None, video_scale=1,
                 verbose=False, dataset_names=None, metric=None, subsets='subsets'):
        evaluations = {'fractions': [], 'accuracies': [], 'selected_frames': []}
        latencies = {'latencies': [], 'lat_cam': [], 'lat_net': [], 'lat_inf': []}
        other = {'true_sizes': [], 'sizes': []}

        for video in self.datasets:
            if dataset_names is not None and video['dataset'] not in dataset_names:
                continue
            print(video['dataset'])
            for query in video[query_key]:
                if metric is not None and query['metric'] != metric:
                    continue
                evaluation_summary = self.eval(video['dataset'], video[subsets], query)
                for item in evaluation_summary:
                    evaluations['fractions'].append(item['fraction'])
                    evaluations['accuracies'].append(item['evaluation'])
                    evaluations['selected_frames'].append(item['selected_frames'])

                if verbose:
                    new_evaluations = {'fractions': [], 'accuracies': []}
                    for item in evaluation_summary:
                        new_evaluations['fractions'].append(item['fraction'])
                        new_evaluations['accuracies'].append(item['evaluation'])
                    # print(f'{query["split"]:.2f},'
                    #       f'{np.mean(new_evaluations["fractions"]):.4f},'
                    #       f'{1-np.mean(new_evaluations["fractions"]):.4f},'
                    #       f'{np.mean(new_evaluations["accuracies"]):.4f}')

                    # cuts = [.40]
                    # df = pd.DataFrame(new_evaluations)
                    # frac_sent = df.quantile(cuts)['fractions'].to_list()[0]
                    # frac_filtered = 1 - frac_sent
                    # acc = df.quantile(cuts)['accuracies'].to_list()[0]

                    # print(f'{query["split"]:.2f},'
                    #       f'{frac_sent:.4f},'
                    #       f'{frac_filtered:.4f},'
                    #       f'{acc:.4f}')
                    print(video['dataset'])
                    show_stats(new_evaluations, ['fractions'])
                    # print('---')

                if not network:
                    continue
                network_summary = self.frame_latency(
                    evaluation_summary, network['bandwidth'], network['rtt'], network_name, scale=video_scale)
                latencies['latencies'].append(network_summary['latencies'])
                latencies['lat_cam'].append(network_summary['lat_cam'])
                latencies['lat_net'].append(network_summary['lat_net'])
                latencies['lat_inf'].append(network_summary['lat_inf'])
                other['sizes'].append(network_summary['sizes'])
                other['true_sizes'].append(network_summary['true_sizes'])

        print(self.name)
        print('-' * 41)
        evaluations['fractions'] = [1 - f for f in evaluations['fractions']]
        show_stats(evaluations, ['fractions', 'accuracies'])
        if not network:
            print()
            return
        latencies['latencies'] = flatten(latencies['latencies'])
        latencies['lat_cam'] = flatten(latencies['lat_cam'])
        latencies['lat_net'] = flatten(latencies['lat_net'])
        latencies['lat_inf'] = flatten(latencies['lat_inf'])
        # show_stats(latencies, ['latencies'])
        show_stats(latencies, ['latencies', 'lat_cam', 'lat_net', 'lat_inf'])

        other['sizes'] = flatten(other['sizes'])
        other['true_sizes'] = flatten(other['true_sizes'])
        size_mean = np.mean(other['sizes'])
        truesize_mean = np.mean(other['true_sizes'])
        print(f'      size: ({1 - size_mean / truesize_mean :.4f})')
        print()

    def eval(self, dataset, subsets, query):
        raise NotImplementedError()

    def eval_path(self, seg):
        return self.root / 'evaluation' / seg[0] / seg[1] / f'{seg[2]}.json'

    def diff_path(self, seg):
        return self.root / 'diff' / seg[0] / seg[1] / f'{seg[2]}.json'

    def infer_path(self, seg):
        return self.root / 'inference' / seg[0] / seg[1] / f'{seg[2]}.json'

    @staticmethod
    def get_segments(dataset, subsets, video_list_path=None):
        video_list_path = video_list_path or 'config/video_list.json'
        video_list = load_json(video_list_path)[dataset]
        segments = [(dataset, i[0], i[1].split('.')[0])
                    for i in flatten([list(product([s], video_list[s])) for s in subsets])]
        return segments

    def load_result(self, dataset, subsets, differ, metric):
        segments = self.get_segments(dataset, subsets)
        evals = [load_evaluation(self.eval_path(seg), differ, metric) for seg in segments]
        diff_vectors = [load_diff_vector(self.diff_path(seg), differ) for seg in segments]
        diff_results = [load_diff_result(self.diff_path(seg), differ) for seg in segments]
        assert len(evals) == len(diff_vectors) == len(diff_results)
        return evals, diff_vectors, diff_results

    def load_inference(self, dataset, subsets):
        segments = self.get_segments(dataset, subsets)
        inference = [load_inference(self.infer_path(seg)) for seg in segments]
        return inference

    def get_segment_size(self, dataset, subset, segment, selected_frames=None, log_name=None, scale=1):
        selected_frames = selected_frames or []
        selected_frames = selected_frames if len(selected_frames) > 0 else None
        video_path = self.video_root / dataset / subset / f'{segment}.mp4'
        size = get_video_size(video_path, selected_frames, log_name, scale=scale)
        return size

    def frame_latency(self, summary, bandwidth, rtt, network_name=None, divided_by=4, scale=1):
        report = {'sizes': [], 'true_sizes': [], 'latencies': [], 'lat_cam': [], 'lat_net': [], 'lat_inf': []}

        bandwidth_Bps = bandwidth / 8 * 1024 * 1024
        rtt_latency = rtt / 2 / 1000
        num_segment_frames = int(self.fps * self.segment_duration) // 5

        for seg in summary:
            selected_frames = seg['selected_frames']
            num_sent_frames = len(selected_frames)
            if self.send_all:
                selected_frames = []
                num_sent_frames = num_segment_frames
            size = self.get_segment_size(
                seg['dataset'], seg['subset'], seg['segment'],
                selected_frames, network_name, scale=scale)
            original_size = self.get_segment_size(seg['dataset'], seg['subset'], seg['segment'], log_name='true')
            report['sizes'].append(size)
            report['true_sizes'].append(original_size)
            for fid in range(num_sent_frames):
                sent_id = fid if self.send_all else selected_frames[fid]
                cam_latency = (1 / self.fps) * (num_segment_frames - sent_id // 5)
                # print(f'camera latency: fid={fid:03d}, lat={cam_latency:.4f}, s/d={sent_id // 5}')
                net_latency = (size / bandwidth_Bps + rtt_latency) / float(divided_by)
                inf_latency = self.gpu_time * (fid // 5 + 1)
                latency = cam_latency + net_latency + inf_latency
                report['latencies'].append(latency)
                report['lat_cam'].append(cam_latency)
                report['lat_net'].append(net_latency)
                report['lat_inf'].append(inf_latency)
        return report


class Optimal(Simulator):

    def __init__(self, datasets, typ, classes, **kwargs):
        super().__init__(datasets, **kwargs)
        self.gpu_time = 1 / 40
        classes_str = ':'.join([str(c) for c in classes])
        self.network_logname = f'optimal_{typ}_{classes_str}'
        self.type = typ
        self.classes = classes
        self.name = f'optimal {typ} ({classes})'
        if self.type == 'coco':
            self.evaluator = MetricComposer.from_json([{'type': 'coco', 'class': self.classes}])
        else:
            self.evaluator = None

    def eval(self, dataset, subsets, query):
        segments = self.get_segments(dataset, subsets)
        target_acc = query['target_acc']
        inferences = self.load_inference(dataset, subsets)

        output_path = f'data/optimal-{target_acc:.1f}/{dataset}_{self.network_logname}.json'
        if Path(output_path).exists():
            loaded_summary = load_json(output_path)
            summary = [item for item in loaded_summary if item['dataset'] == dataset and item['subset'] in subsets]
        else:
            with multiprocessing.Pool() as pool:
                result = pool.map(partial(self.select_frames, target_acc=target_acc), inferences)
            summary = [
                {
                    'dataset': segments[index][0],
                    'subset': segments[index][1],
                    'segment': segments[index][2],
                    'fraction': len(res['selected_frames']) / len(inferences[index].keys()),
                    'evaluation': sum(res['scores']) / len(res['scores']),
                    'selected_frames': res['selected_frames'],
                }
                for res, index in zip(result, range(len(segments)))
            ]
            dump_json(summary, output_path, mkdir=True)
        return summary

    def select_frames(self, inference, target_acc):
        frame_ids = list(inference.keys())
        summary = [
            {
                'fid': int(fid),
                'count': Optimal.count_objects(inference[fid], self.classes),
                'inference': inference[fid],
            }
            for fid in frame_ids
        ]
        selected_frames = [summary[0]['fid']]
        scores = [1.0]
        for fid in range(1, len(summary)):
            last_selected_fid = selected_frames[-1]
            if self.type == 'counting':
                score = Optimal.get_counting_score(summary[fid]['count'], summary[last_selected_fid]['count'])
            elif self.type == 'tagging':
                score = Optimal.get_tagging_score(summary[fid]['count'], summary[last_selected_fid]['count'])
            elif self.type == 'coco':
                score = self.get_detection_score(summary[fid]['inference'], summary[last_selected_fid]['inference'])
            new_scores = scores + [score]
            if sum(new_scores) / len(new_scores) >= target_acc:
                # if score >= target_acc:
                scores.append(score)
            else:
                selected_frames.append(fid)
                scores.append(1.0)
        return {
            'selected_frames': selected_frames,
            'scores': scores,
        }

    @staticmethod
    def count_objects(frame_inference, classes):
        count = len([c for c in frame_inference['detection_classes'] if c in classes])
        return count

    @staticmethod
    def get_counting_score(count1, count2):
        if count1 == count2:
            return 1.0
        return (max(count1, count2) - abs(count1 - count2)) / max(count1, count2)

    @staticmethod
    def get_tagging_score(count1, count2):
        if count1 == 0 or count2 == 0:
            return 0.0
        return 1.0

    def get_detection_score(self, inference1, inference2):
        result = self.evaluator.evaluate_single_frame(inference1, inference2)
        return result[f'mAP-{":".join(str(i) for i in self.classes)}']


class Reducto(Simulator):

    def __init__(self, datasets, **kwargs):
        super().__init__(datasets, **kwargs)
        self.inference_fps = 40
        self.profiling_fps = 40
        self.camera_diff_fps = 30
        self.inference_time = 1 / self.inference_fps
        self.profiling_time = 1 / self.profiling_fps
        self.camera_diff_time = 1 / self.camera_diff_fps
        self.len_bootstrapping = 5
        self.gpu_time = 1 / 40
        self.network_logname = 'reducto'
        self.name = 'reducto'

    def eval(self, dataset, subsets, query):
        segments = self.get_segments(dataset, subsets)
        metric = query['metric']
        differ = query['differ']
        target_acc = query['target_acc']
        dist_thresh = query['distance']
        safe_zone = query['safe']
        evals, diff_vectors, diff_results = self.load_result(dataset, subsets, differ, metric)
        length = len(evals)

        profiled_indexes = []
        summary = {}
        for index in range(length):
            # starting
            summary[index] = {
                'start': index * self.segment_duration,
                'taken': (index + 1) * self.segment_duration,
            }
            # bootstrapping
            if len(profiled_indexes) < self.len_bootstrapping:
                profiled_indexes.append(index)
                distance = -1
                fraction = 1.0
                evaluation = 1.0
                summary[index]['sent'] = summary[index]['taken']
                summary[index]['inf_done'] = summary[index]['sent'] + self.inference_time
                summary[index]['prof_done'] = summary[index]['inf_done'] + self.profiling_time
            # dynamic phase
            else:
                profiled_available = [
                    p_index for p_index in profiled_indexes
                    if summary[p_index]['prof_done'] < summary[index]['taken']
                ]
                # if there is not enough profiled segments, still sends everything
                if len(profiled_available) < self.len_bootstrapping:
                    profiled_indexes.append(index)
                    distance = -1
                    fraction = 1.0
                    evaluation = 1.0
                    summary[index]['sent'] = summary[index]['taken']
                    summary[index]['inf_done'] = summary[index]['sent'] + self.inference_time
                    summary[index]['prof_done'] = summary[index]['inf_done'] + self.profiling_time
                # good to start doing dynamic adoption
                else:
                    threshmap_init_dict = HashBuilder().generate_threshmap(
                        [evals[i] for i in profiled_indexes],
                        [diff_vectors[i] for i in profiled_indexes],
                        target_acc=target_acc, safe_zone=safe_zone)
                    thresh_map = ThreshMap(threshmap_init_dict[differ])
                    thresh, distance = thresh_map.get_thresh(diff_vectors[index][differ])
                    distance = np.sum(distance)
                    if distance > dist_thresh:
                        profiled_indexes.append(index)
                        fraction = 1.0
                        evaluation = 1.0
                        summary[index]['sent'] = summary[index]['taken'] + self.camera_diff_time
                        summary[index]['inf_done'] = summary[index]['sent'] + self.inference_time
                        summary[index]['prof_done'] = summary[index]['inf_done'] + self.profiling_time
                    else:
                        fraction = diff_results[index][differ][thresh]['fraction']
                        evaluation = evals[index][differ][thresh][metric]
                        summary[index]['sent'] = summary[index]['taken'] + self.camera_diff_time
                        summary[index]['inf_done'] = summary[index]['sent'] + self.inference_time * fraction
                        summary[index]['prof_done'] = -1

            summary[index]['dataset'] = segments[index][0]
            summary[index]['subset'] = segments[index][1]
            summary[index]['segment'] = segments[index][2]
            summary[index]['profiling?'] = int(index in profiled_indexes)
            summary[index]['diff_vector'] = diff_vectors[index][differ]
            summary[index]['distance'] = distance
            summary[index]['fraction'] = fraction
            summary[index]['evaluation'] = evaluation
            if fraction == 1.0:
                summary[index]['selected_frames'] = []
            elif fraction < 1.0:
                selected_frames = diff_results[index][differ][thresh]['selected_frames']
                summary[index]['selected_frames'] = selected_frames

        summary_list = [summary[v] for v in range(self.len_bootstrapping, length)]
        return summary_list


class ReductoOptimal(Simulator):

    def __init__(self, datasets, **kwargs):
        super().__init__(datasets, **kwargs)
        self.gpu_time = 1 / 40
        self.network_logname = 'reducto_optimal'
        self.name = 'reducto optimal'

    def eval(self, dataset, subsets, query):
        segments = self.get_segments(dataset, subsets)
        metric = query['metric']
        differ = query['differ']
        target_acc = query['target_acc']
        evals, diff_vectors, diff_results = self.load_result(dataset, subsets, differ, metric)

        summary = []
        for index in range(len(evals)):
            good_threshes = [th for th, acc_dict in evals[index][differ].items()
                             if acc_dict[metric] > target_acc]
            good_fracs = [(th, diff_results[index][differ][th]['fraction'],
                           evals[index][differ][th][metric]) for th in good_threshes]
            good_fracs_sorted = sorted(good_fracs, key=lambda th_acc: th_acc[1])
            if len(good_fracs_sorted) == 0:
                optimal_fraction = 1.0
                optimal_evaluation = 1.0
            else:
                optimal_fraction = good_fracs_sorted[0][1]
                optimal_evaluation = good_fracs_sorted[0][2]
            summary.append({
                'fraction': optimal_fraction,
                'evaluation': optimal_evaluation,
                'dataset': segments[index][0],
                'subset': segments[index][1],
                'segment': segments[index][2],
                'selected_frames': [],
            })
        return summary


class Glimpse(Simulator):

    def __init__(self, datasets, thresh_key, **kwargs):
        super().__init__(datasets, **kwargs)
        self.thresh_key = thresh_key
        self.gpu_time = 1 / 40
        self.network_logname = thresh_key
        self.name = f'glimpse ({thresh_key})'

    def eval(self, dataset, subsets, query):
        segments = self.get_segments(dataset, subsets)
        metric = query['metric']
        differ = query['differ']
        thresh = query[self.thresh_key]
        evals, diff_vectors, diff_results = self.load_result(dataset, subsets, differ, metric)

        summary = []
        for index in range(len(evals)):
            summary.append({
                'dataset': segments[index][0],
                'subset': segments[index][1],
                'segment': segments[index][2],
                'fraction': diff_results[index][differ][thresh]['fraction'],
                'evaluation': evals[index][differ][thresh][metric],
                'diff_vector': diff_vectors[index][differ],
                'selected_frames': diff_results[index][differ][thresh]['selected_frames'],
            })
        return summary


class GlimpseOptimizer(Simulator):

    def __init__(self, datasets, **kwargs):
        super().__init__(datasets, **kwargs)
        self.gpu_time = 1 / 40
        self.network_logname = 'glimpse_optimizer'
        self.name = 'glimpse optimizer'

    def eval(self, dataset, subsets, query):
        segments = self.get_segments(dataset, subsets)
        metric = query['metric']
        differ = query['differ']
        split = query['split']
        target_acc = query['target_acc']
        threshes = self.load_threshes(dataset)[differ]
        evals, diff_vectors, diff_results = self.load_result(dataset, subsets, differ, metric)

        split_point = int(len(segments) * split)
        training_segs, testing_segs = segments[:split_point], segments[split_point:]
        training_evals = evals[:split_point]
        training_diffs = diff_results[:split_point]
        thresh = self.find_best_thresh(training_evals, training_diffs, target_acc, threshes, metric, differ)

        summary = []
        for index in range(split_point, len(evals)):
            summary.append({
                'dataset': segments[index][0],
                'subset': segments[index][1],
                'segment': segments[index][2],
                'fraction': diff_results[index][differ][thresh]['fraction'],
                'evaluation': evals[index][differ][thresh][metric],
                'diff_vector': diff_vectors[index][differ],
                'selected_frames': diff_results[index][differ][thresh]['selected_frames'],
            })
        return summary

    @staticmethod
    def load_threshes(dataset):
        thresh_path = Path('config') / 'threshes' / f'{dataset}.json'
        threshes = load_json(thresh_path)
        return threshes

    @staticmethod
    def find_best_thresh(training_evals, training_diffs, target_acc, threshes, metric, differ):
        assert len(training_evals) == len(training_diffs)
        best_thresh = threshes[0]
        for thresh in threshes:
            acc = np.mean([evl[differ][thresh][metric] for evl in training_evals])
            if acc <= target_acc:
                return best_thresh
            else:
                best_thresh = thresh
        return best_thresh


class Focus(Simulator):

    def __init__(self, datasets, **kwargs):
        super().__init__(datasets, **kwargs)
        self.result_root = Path('data') / 'focus'
        self.inference_root = Path('data') / 'inference'
        self.gpu_time = 1 / 40
        self.network_logname = 'focus'
        self.name = 'focus'

    def eval(self, dataset, subsets, query):
        inference_path = self.inference_root / dataset
        metric = query['metric']
        send_thresh = query['send_thresh']
        tinyyolo_acc = query['tinyyolo_acc']
        evaluator = self.metric2evaluator(metric)

        summary = []
        for subset in subsets:
            j = load_json(self.result_root / dataset / f'{subset}_car.json')
            for segment_id in sorted(j.keys()):
                if segment_id == '':
                    continue
                segment_name = f'segment{segment_id}' if not segment_id.startswith('segment') else segment_id
                frame_ids = sorted(int(fid) for fid in j[segment_id])

                selected_frames = []
                evaluations = []
                for fid in frame_ids:
                    inference = j[segment_id][str(fid)]
                    scores = inference['scores']
                    num_detections = len(inference['scores'])

                    if num_detections == 0 or any(float(s) < send_thresh for s in scores):
                        selected_frames.append(fid + 1)
                        evaluations.append(1.0)
                    elif tinyyolo_acc > 0.0:
                        evaluations.append(tinyyolo_acc)
                    else:
                        detection = {
                            'num_detections': num_detections,
                            'detection_scores': inference['scores'],
                            # [[xmin, ymin, xmax, ymax]]
                            'detection_boxes': [[float(coord) for coord in box] for box in inference['boxes']],
                            'detection_classes': [2] * num_detections,
                        }
                        ground_truth_path = inference_path / subset / f'{segment_name}.json'
                        ground_truth = load_json(ground_truth_path)[str(fid + 1)]
                        ground_truth_filtered = {
                            'num_detections': 0,
                            'detection_scores': [],
                            'detection_boxes': [],
                            'detection_classes': [],
                        }
                        for i in range(ground_truth['num_detections']):
                            if ground_truth['detection_classes'][i] != 2:
                                continue
                            ground_truth_filtered['num_detections'] += 1
                            ground_truth_filtered['detection_scores'].append(ground_truth['detection_scores'][i])
                            ground_truth_filtered['detection_boxes'].append(ground_truth['detection_boxes'][i])
                            ground_truth_filtered['detection_classes'].append(ground_truth['detection_classes'][i])
                        feval = evaluator.evaluate_single_frame(
                            ground_truth=ground_truth_filtered,
                            comparision=detection)['mAP-2']
                        evaluations.append(feval)

                summary.append({
                    'dataset': dataset,
                    'subset': subset,
                    'segment': segment_name,
                    'fraction': len(selected_frames) / len(frame_ids),
                    'evaluation': sum(evaluations) / len(evaluations),
                    'selected_frames': selected_frames,
                })
        return summary

    @staticmethod
    def metric2evaluator(metric):
        typ, cls = metric.split('-')
        if typ == 'mAP':
            typ = 'coco'
        evaluator = MetricComposer.from_json([{'type': typ, 'class': [cls]}])
        return evaluator


class Cloudseg(Simulator):

    def __init__(self, datasets, scale, **kwargs):
        super().__init__(datasets, **kwargs)
        self.result_root = Path('data') / 'cloudseg'
        self.scale = scale
        self.gpu_time = 1 / 30
        self.send_all = True
        self.network_logname = f'x{scale}_orig'
        self.name = f'cloudseg {scale}x'

    def eval(self, dataset, subsets, query):
        metric = query['metric']

        if metric == 'mAP-2':
            evals_path = self.result_root / f'{dataset}x{self.scale}.json'
            data = load_json(evals_path)
            summary = []
            for item in data:
                # evaluation didn't evaluate raw001, so we skip it
                if item['subset'] not in subsets:
                    continue
                summary.append({
                    'dataset': dataset,
                    'subset': item['subset'],
                    'segment': item['segment'].split('.')[0],
                    'evaluation': item['mAP-2'],
                    'fraction': 1.0,
                    'selected_frames': [],
                    'scale': self.scale,
                })
            return summary
        if metric == 'mAP-0':
            raise ValueError()

        segments = []
        for subset in subsets:
            p = self.video_root / dataset / subset
            segs = [(dataset, subset, v.stem) for v in sorted(p.iterdir()) if v.match('segment???.mp4')]
            segments += segs

        evaluator = self.metric2evaluator(metric)
        summary = []
        for seg in segments:
            subset = seg[1]
            segment = seg[2]
            cloudseg_inference_path = Path('data') / 'inference' / dataset / f'{subset}x{self.scale}' / f'{segment}.json'
            cloudseg_inference = load_json(cloudseg_inference_path)
            cloudseg_inference = self.fix_inference(cloudseg_inference)
            ground_truth_inference_path = Path('data') / 'inference' / dataset / subset / f'{segment}.json'
            ground_truth_inference = load_json(ground_truth_inference_path)
            ground_truth_inference = self.fix_inference(ground_truth_inference)

            selected_frames = [int(k) for k in ground_truth_inference.keys()]
            accuracy = evaluator.evaluate_with_diff(cloudseg_inference, selected_frames, ground_truth_inference)[metric]
            fraction = 1.0
            segment_summary = {
                'dataset': dataset,
                'subset': subset,
                'segment': segment,
                'evaluation': accuracy,
                'fraction': fraction,
                'selected_frames': selected_frames,
                'scale': self.scale,
            }
            # print(f'{dataset}/{subset}/{segment}: frac={fraction:.4f}, acc={accuracy:.4f}, selected={len(selected_frames)}')
            summary.append(segment_summary)
        return summary

    @staticmethod
    def metric2evaluator(metric):
        if metric == 'mAP-0':
            j = [{'type': 'coco', 'class': [0]}]
        if metric == 'mAP-2':
            j = [{'type': 'coco', 'class': [2]}]
        if metric == 'counting-0':
            j = [{'type': 'counting', 'class': [0]}]
        if metric == 'counting-2':
            j = [{'type': 'counting', 'class': [2]}]
        if metric == 'tagging-0':
            j = [{'type': 'tagging', 'class': [0]}]
        if metric == 'tagging-2':
            j = [{'type': 'tagging', 'class': [2]}]
        return MetricComposer.from_json(j)

    @staticmethod
    def fix_inference(inference):
        new_inference = {
            int(fid): inference[fid]
            for fid in inference.keys()
        }
        return new_inference


class CloudsegReducto(Simulator):

    def __init__(self, datasets, scale, **kwargs):
        super().__init__(datasets, **kwargs)
        self.inference_fps = 40
        self.profiling_fps = 40
        self.camera_diff_fps = 30
        self.inference_time = 1 / self.inference_fps
        self.profiling_time = 1 / self.profiling_fps
        self.camera_diff_time = 1 / self.camera_diff_fps
        self.len_bootstrapping = 5
        self.gpu_time = 1 / 40
        self.scale = scale
        self.network_logname = f'reducto_cloudseg_{scale}x'
        self.name = f'reducto cloudseg {scale}x'

    def eval(self, dataset, subsets, query):
        reducto_summary = self.reducto_select_frames(dataset, subsets, query)
        summary = []
        metric = query['metric']
        evaluator = self.metric2evaluator(metric)
        for item in reducto_summary:
            subset = item['subset']
            segment = item['segment']
            fraction = item['fraction']
            evaluation = item['evaluation']
            selected_frames = item['selected_frames']

            cloudseg_inference_path = Path('data') / 'inference' / dataset / f'{subset}x{self.scale}' / f'{segment}.json'
            cloudseg_inference = load_json(cloudseg_inference_path)
            cloudseg_inference = self.fix_inference(cloudseg_inference)
            ground_truth_inference_path = Path('data') / 'inference' / dataset / subset / f'{segment}.json'
            ground_truth_inference = load_json(ground_truth_inference_path)
            ground_truth_inference = self.fix_inference(ground_truth_inference)
            if fraction == 1.0 or len(selected_frames) == 0:
                selected_frames = [int(k) for k in ground_truth_inference.keys()]
            accuracy = evaluator.evaluate_with_diff(cloudseg_inference, selected_frames, ground_truth_inference)[metric]

            segment_summary = {
                'dataset': dataset,
                'subset': subset,
                'segment': segment,
                'evaluation': accuracy,
                'fraction': fraction,
                'selected_frames': selected_frames,
                'scale': self.scale,
            }
            # print(f'{dataset}/{subset}/{segment}: frac={fraction:.4f}, acc={accuracy:.4f}, eval={evaluation:.4f}, selected={len(selected_frames)}')
            summary.append(segment_summary)
        return summary

    def reducto_select_frames(self, dataset, subsets, query):
        segments = self.get_segments(dataset, subsets)
        metric = query['metric']
        differ = query['differ']
        target_acc = query['target_acc']
        dist_thresh = query['distance']
        safe_zone = query['safe']
        evals, diff_vectors, diff_results = self.load_result(dataset, subsets, differ, metric)
        length = len(evals)

        profiled_indexes = []
        summary = {}
        for index in range(length):
            # starting
            summary[index] = {
                'start': index * self.segment_duration,
                'taken': (index + 1) * self.segment_duration,
            }
            # bootstrapping
            if len(profiled_indexes) < self.len_bootstrapping:
                profiled_indexes.append(index)
                distance = -1
                fraction = 1.0
                evaluation = 1.0
                summary[index]['sent'] = summary[index]['taken']
                summary[index]['inf_done'] = summary[index]['sent'] + self.inference_time
                summary[index]['prof_done'] = summary[index]['inf_done'] + self.profiling_time
            # dynamic phase
            else:
                profiled_available = [
                    p_index for p_index in profiled_indexes
                    if summary[p_index]['prof_done'] < summary[index]['taken']
                ]
                # if there is not enough profiled segments, still sends everything
                if len(profiled_available) < self.len_bootstrapping:
                    profiled_indexes.append(index)
                    distance = -1
                    fraction = 1.0
                    evaluation = 1.0
                    summary[index]['sent'] = summary[index]['taken']
                    summary[index]['inf_done'] = summary[index]['sent'] + self.inference_time
                    summary[index]['prof_done'] = summary[index]['inf_done'] + self.profiling_time
                # good to start doing dynamic adoption
                else:
                    threshmap_init_dict = HashBuilder().generate_threshmap(
                        [evals[i] for i in profiled_indexes],
                        [diff_vectors[i] for i in profiled_indexes],
                        target_acc=target_acc, safe_zone=safe_zone)
                    thresh_map = ThreshMap(threshmap_init_dict[differ])
                    thresh, distance = thresh_map.get_thresh(diff_vectors[index][differ])
                    distance = np.sum(distance)
                    if distance > dist_thresh:
                        profiled_indexes.append(index)
                        fraction = 1.0
                        evaluation = 1.0
                        summary[index]['sent'] = summary[index]['taken'] + self.camera_diff_time
                        summary[index]['inf_done'] = summary[index]['sent'] + self.inference_time
                        summary[index]['prof_done'] = summary[index]['inf_done'] + self.profiling_time
                    else:
                        fraction = diff_results[index][differ][thresh]['fraction']
                        evaluation = evals[index][differ][thresh][metric]
                        summary[index]['sent'] = summary[index]['taken'] + self.camera_diff_time
                        summary[index]['inf_done'] = summary[index]['sent'] + self.inference_time * fraction
                        summary[index]['prof_done'] = -1

            summary[index]['dataset'] = segments[index][0]
            summary[index]['subset'] = segments[index][1]
            summary[index]['segment'] = segments[index][2]
            summary[index]['profiling?'] = int(index in profiled_indexes)
            summary[index]['diff_vector'] = diff_vectors[index][differ]
            summary[index]['distance'] = distance
            summary[index]['fraction'] = fraction
            summary[index]['evaluation'] = evaluation
            if fraction == 1.0:
                summary[index]['selected_frames'] = []
            elif fraction < 1.0:
                selected_frames = diff_results[index][differ][thresh]['selected_frames']
                summary[index]['selected_frames'] = selected_frames

        summary_list = [summary[v] for v in range(length)]
        return summary_list

    @staticmethod
    def load_cloudseg_evals(dataset, scale):
        evals_path = Path('data') / 'cloudseg' / f'{dataset}x{scale}.json'
        evals = load_json(evals_path)
        return evals

    @staticmethod
    def fix_inference(inference):
        new_inference = {
            int(fid): inference[fid]
            for fid in inference.keys()
        }
        return new_inference

    @staticmethod
    def metric2evaluator(metric):
        if metric == 'mAP-0':
            j = [{'type': 'coco', 'class': [0]}]
        if metric == 'mAP-2':
            j = [{'type': 'coco', 'class': [2]}]
        if metric == 'counting-0':
            j = [{'type': 'counting', 'class': [0]}]
        if metric == 'counting-2':
            j = [{'type': 'counting', 'class': [2]}]
        if metric == 'tagging-0':
            j = [{'type': 'tagging', 'class': [0]}]
        if metric == 'tagging-2':
            j = [{'type': 'tagging', 'class': [2]}]
        return MetricComposer.from_json(j)


class FilterForward(Simulator):

    def __init__(self, datasets, result_dir, **kwargs):
        super().__init__(datasets, **kwargs)
        self.result_root = Path('data') / 'ff' / result_dir
        self.inference_root = Path('data') / 'inference'
        self.gpu_time = 1 / 40
        self.network_logname = 'ff'
        self.name = 'filterforward'

    def eval(self, dataset, subsets, query):
        metric = query['metric']
        evaluator = self.metric2evaluator(metric)

        segments = self.get_segments(dataset, subsets)
        summary = []
        for seg in segments:
            subset = seg[1]
            segment = seg[2]
            ## QUICK FIX DO NOT USE IT FOR OTHER TESTS
            # result = load_json(self.result_root / dataset / subset / f'{segment}.json')
            result = load_json(self.result_root / subset / f'{segment}.json')
            selected_frames = result['selected_frames']
            fraction = result['fraction']
            ground_truth_inference_path = self.inference_root / dataset / subset / f'{segment}.json'
            ground_truth_inference = load_json(ground_truth_inference_path)
            ground_truth_inference = self.fix_inference(ground_truth_inference)
            accuracy = evaluator.evaluate_with_diff(ground_truth_inference, selected_frames)[metric]
            segment_summary = {
                'dataset': dataset,
                'subset': subset,
                'segment': segment,
                'evaluation': accuracy,
                'fraction': fraction,
                'selected_frames': selected_frames,
            }
            summary.append(segment_summary)
        return summary

    @staticmethod
    def metric2evaluator(metric):
        if metric == 'mAP-0':
            j = [{'type': 'coco', 'class': [0]}]
        if metric == 'mAP-2':
            j = [{'type': 'coco', 'class': [2]}]
        if metric == 'counting-0':
            j = [{'type': 'counting', 'class': [0]}]
        if metric == 'counting-2':
            j = [{'type': 'counting', 'class': [2]}]
        if metric == 'tagging-0':
            j = [{'type': 'tagging', 'class': [0]}]
        if metric == 'tagging-2':
            j = [{'type': 'tagging', 'class': [2]}]
        return MetricComposer.from_json(j)

    @staticmethod
    def fix_inference(inference):
        new_inference = {
            int(fid): inference[fid]
            for fid in inference.keys()
        }
        return new_inference


class Simple(Simulator):

    def __init__(self, datasets, **kwargs):
        super().__init__(datasets, **kwargs)
        self.gpu_time = 1 / 40
        self.send_all = True
        self.network_logname = 'true'
        self.name = 'simple'

    def eval(self, dataset, subsets, query):
        segments = self.get_segments(dataset, subsets)
        summary = []
        for seg in segments:
            summary.append({
                'dataset': seg[0],
                'subset': seg[1],
                'segment': seg[2],
                'evaluation': 1.0,
                'fraction': 1.0,
                'selected_frames': [],
            })
        return summary


if __name__ == '__main__':
    # Reducto overall evaluations
    reducto_verbose = True

    # network = {'bandwidth': 24, 'rtt': 20}
    # Reducto(full_datasets).simulate(query_key='reducto_queries_90', metric='mAP-0',
    #                                 network=network, network_name='reducto_mAP0')
    # Reducto(full_datasets).simulate(query_key='reducto_queries_90', metric='tagging-0',
    #                                 network=network, network_name='reducto_tagging0')
    # Simple(full_datasets).simulate(network=network, network_name='true')
    # Reducto(full_datasets).simulate(query_key='reducto_queries_80', metric='mAP-2', verbose=reducto_verbose)
    # Reducto(full_datasets).simulate(query_key='reducto_queries_70', metric='mAP-2', verbose=reducto_verbose)

    # Reducto(full_datasets).simulate(query_key='reducto_queries_90', metric='mAP-0', dataset_names=['jacksonhole'], verbose=reducto_verbose)
    # Reducto(full_datasets).simulate(query_key='reducto_queries_80', metric='mAP-0', dataset_names=['jacksonhole'], verbose=reducto_verbose)
    # Reducto(full_datasets).simulate(query_key='reducto_queries_70', metric='mAP-0', dataset_names=['jacksonhole'], verbose=reducto_verbose)

    # Reducto(full_datasets).simulate(query_key='reducto_queries_90', metric='counting-2', verbose=reducto_verbose)
    # Reducto(full_datasets).simulate(query_key='reducto_queries_90', metric='counting-0', verbose=reducto_verbose)
    # Reducto(full_datasets).simulate(query_key='reducto_queries_90', metric='tagging-2', verbose=reducto_verbose)
    # Reducto(full_datasets).simulate(query_key='reducto_queries_90', metric='tagging-0', verbose=reducto_verbose)

    # Reducto + Cloudseg comparision evaluations
    # reducto_cloudseg_dataset = [
    #     {
    #         'dataset': 'auburn',
    #         'subsets': ['fri005', 'sat001am', 'sat005pm'],
    #         'queries': [
    #             {'metric': 'mAP-2',      'differ': 'edge', 'distance': 1.0, 'safe': 0.01, 'target_acc': 0.90},
    #             {'metric': 'counting-2', 'differ': 'area', 'distance': 6.0, 'safe': 0.05, 'target_acc': 0.90},
    #             {'metric': 'tagging-2',  'differ': 'area', 'distance': 6.0, 'safe': 0.00, 'target_acc': 0.90},
    #         ],
    #         'properties': {'fps': 30},
    #     },
    #     {
    #         'dataset': 'jacksonhole',
    #         'subsets': ['raw000', 'raw019', 'raw038'],
    #         'queries': [
    #             {'metric': 'mAP-2',      'differ': 'edge', 'distance': 1.4, 'safe': 0.05, 'target_acc': 0.90},
    #             {'metric': 'counting-2', 'differ': 'area', 'distance': 6.0, 'safe': 0.05, 'target_acc': 0.90},
    #             {'metric': 'tagging-2',  'differ': 'area', 'distance': 6.0, 'safe': 0.00, 'target_acc': 0.90},
    #         ],
    #         'properties': {'fps': 30},
    #     },
    #     {
    #         'dataset': 'lagrange',
    #         'subsets': ['raw000', 'raw005', 'raw022'],
    #         'queries': [
    #             {'metric': 'mAP-2',      'differ': 'edge', 'distance': 1.4, 'safe': 0.03, 'target_acc': 0.90},
    #             {'metric': 'counting-2', 'differ': 'area', 'distance': 6.0, 'safe': 0.05, 'target_acc': 0.90},
    #             {'metric': 'tagging-2',  'differ': 'area', 'distance': 6.0, 'safe': 0.00, 'target_acc': 0.90},
    #         ],
    #         'properties': {'fps': 30},
    #     },
    #     {
    #         'dataset': 'southampton',
    #         'subsets': ['raw000', 'sat005', 'sat009pm'],
    #         'queries': [
    #             {'metric': 'mAP-2',      'differ': 'edge', 'distance': 1.0, 'safe': 0.00, 'target_acc': 0.90},
    #             {'metric': 'counting-2', 'differ': 'area', 'distance': 6.0, 'safe': 0.05, 'target_acc': 0.90},
    #             {'metric': 'tagging-2',  'differ': 'area', 'distance': 6.0, 'safe': 0.00, 'target_acc': 0.90},
    #         ],
    #         'properties': {'fps': 30},
    #     },
    # ]

    # Reducto(reducto_cloudseg_dataset).simulate(metric='mAP-2', network=network, network_name='reducto_small_subsets_map2')
    # Cloudseg(reducto_cloudseg_dataset, scale=2).simulate(metric='mAP-2', network=network, network_name='x2_orig')
    # Cloudseg(reducto_cloudseg_dataset, scale=4).simulate(metric='mAP-2', network=network, network_name='x4_orig')
    # CloudsegReducto(reducto_cloudseg_dataset, scale=2).simulate(metric='mAP-2', network=network, network_name='reducto_x2_map2', video_scale=2)
    # CloudsegReducto(reducto_cloudseg_dataset, scale=4).simulate(metric='mAP-2', network=network, network_name='reducto_x4_map2', video_scale=4)
    # Reducto(reducto_cloudseg_dataset).simulate(metric='counting-2', network=network, network_name='reducto_small_subsets_counting2')
    # Cloudseg(reducto_cloudseg_dataset, scale=2).simulate(metric='counting-2', network=network, network_name='x2_orig')
    # Cloudseg(reducto_cloudseg_dataset, scale=4).simulate(metric='counting-2', network=network, network_name='x4_orig')
    # CloudsegReducto(reducto_cloudseg_dataset, scale=2).simulate(metric='counting-2', network=network, network_name='reducto_x2_counting2', video_scale=2)
    # CloudsegReducto(reducto_cloudseg_dataset, scale=4).simulate(metric='counting-2', network=network, network_name='reducto_x4_counting2', video_scale=4)
    # Reducto(reducto_cloudseg_dataset).simulate(network=network, network_name='reducto_small_subsets')
    # Cloudseg(reducto_cloudseg_dataset, scale=2).simulate(network=network, network_name='x2_orig')
    # Cloudseg(reducto_cloudseg_dataset, scale=4).simulate(network=network, network_name='x4_orig')
    # CloudsegReducto(reducto_cloudseg_dataset, scale=2).simulate(network=network, network_name='reducto_x2_tagging2', video_scale=2)
    # CloudsegReducto(reducto_cloudseg_dataset, scale=4).simulate(network=network, network_name='reducto_x4_tagging2', video_scale=4)

    # FilterForward
    ff_dataset = [
        # {
        #     'dataset': 'southampton',
        #     'subsets': ['raw001', 'sat005', 'sat009pm'],
        #     'queries': [
        #         {'metric': 'mAP-2',      'differ': 'edge', 'distance': 1.0, 'safe': 0.00, 'target_acc': 0.90},
        #         {'metric': 'counting-2', 'differ': 'area', 'distance': 6.0, 'safe': 0.05, 'target_acc': 0.90},
        #         {'metric': 'tagging-2',  'differ': 'area', 'distance': 6.0, 'safe': 0.00, 'target_acc': 0.90},
        #     ],
        #     'properties': {'fps': 30},
        # },
        {
            'dataset': 'jacksonhole',
            'subsets': ['raw000', 'raw019', 'raw038'],
            'queries': [
                {'metric': 'mAP-2',      'differ': 'edge', 'distance': 1.0, 'safe': 0.00, 'target_acc': 0.90},
                {'metric': 'counting-2', 'differ': 'area', 'distance': 6.0, 'safe': 0.05, 'target_acc': 0.90},
                {'metric': 'tagging-2',  'differ': 'area', 'distance': 6.0, 'safe': 0.00, 'target_acc': 0.90},
            ],
            'properties': {'fps': 30},
        }
    ]
    # FilterForward(ff_dataset, result_dir='0203_0.465').simulate(metric='mAP-2')
    # FilterForward(ff_dataset, result_dir='0203_0.46').simulate(metric='mAP-2')
    # FilterForward(ff_dataset, result_dir='0203_0.47').simulate(metric='mAP-2')
    # FilterForward(ff_dataset, result_dir='0202_0.44').simulate(metric='mAP-2')

    # FilterForward(ff_dataset, result_dir='0202_0.4').simulate(metric='mAP-2')
    # FilterForward(ff_dataset, result_dir='0202_0.41').simulate(metric='mAP-2')
    # FilterForward(ff_dataset, result_dir='0202_0.42').simulate(metric='mAP-2')
    # FilterForward(ff_dataset, result_dir='0202_0.43').simulate(metric='mAP-2')

    # network = {'bandwidth': 60, 'rtt': 5}
    # FilterForward(ff_dataset, result_dir='0202_0.44').simulate(metric='mAP-2', network=network, network_name='ff_0202_0.43_map2')

    # FilterForward(ff_dataset).simulate(metric='counting-2', network=network, network_name='ff_counting2')
    # FilterForward(ff_dataset).simulate(metric='tagging-2', network=network, network_name='ff_tagging2')
    # FilterForward(ff_dataset).simulate(metric='mAP-2', network=network, network_name='ff_4718275_map2')
    # FilterForward(ff_dataset).simulate(metric='counting-2', network=network, network_name='ff_4718275_counting2')
    # FilterForward(ff_dataset).simulate(metric='tagging-2', network=network, network_name='ff_4718275_tagging2')

    # focus_dataset = [
    #     {
    #         'dataset': 'auburn',
    #         'subsets': ['fri005', 'sat001am', 'sat005pm'],
    #         'queries': [{'metric': 'mAP-2', 'differ': 'edge', 'distance': 1.7, 'safe': 0.01,
    #                      'send_thresh': 0.9601, 'tinyyolo_acc': 0.6, 'target_acc': 0.90}],
    #         'properties': {'fps': 30},
    #     },
    #     {
    #         'dataset': 'jacksonhole',
    #         'subsets': ['raw000', 'raw019', 'raw038'],
    #         'queries': [{'metric': 'mAP-2', 'differ': 'edge', 'distance': 1.5, 'safe': 0.030,
    #                      'send_thresh': 0.1236, 'tinyyolo_acc': 0.6, 'target_acc': 0.90}],
    #         'properties': {'fps': 30},
    #     },
    #     {
    #         'dataset': 'lagrange',
    #         'subsets': ['raw000', 'raw005', 'raw022'],
    #         'queries': [{'metric': 'mAP-2', 'differ': 'edge', 'distance': 1.5, 'safe': 0.025,
    #                      'send_thresh': 0.1284, 'tinyyolo_acc': 0.6, 'target_acc': 0.90}],
    #         'properties': {'fps': 30},
    #     },
    #     {
    #         'dataset': 'southampton',
    #         'subsets': ['raw001', 'sat005', 'sat009pm'],
    #         'queries': [{'metric': 'mAP-2', 'differ': 'edge', 'distance': 1.5, 'safe': 0.020,
    #                      'send_thresh': 0.1241, 'tinyyolo_acc': 0.6, 'target_acc': 0.90}],
    #         'properties': {'fps': 30},
    #     },
    # ]
    # network = {'bandwidth': 60, 'rtt': 5}
    # Focus(focus_dataset).simulate(metric='mAP-2', network=network, network_name='focus')
    # Simple(focus_dataset).simulate(network=network, network_name='true')
    # Reducto(focus_dataset).simulate(network=network, network_name='reducto_final_small_map2')
    # Optimal(focus_dataset, typ='coco', classes=[2]).simulate(
    #     network=network, network_name='true_optimal_map2_small')

    # GlimpseOptimizer(full_datasets).simulate(query_key='glimpse_queries_90', verbose=True)

    # _network = {'bandwidth': 60, 'rtt': 5}
    # Reducto(_datasets).simulate()
    # Glimpse(_datasets, thresh_key='glimpse_best').simulate(_network)
    # Focus(_datasets).simulate(_network)
    # Cloudseg(_datasets, scale=2).simulate(_network)
    # Cloudseg(_datasets, scale=4).simulate(_network)
    # ReductoOptimal(_datasets).simulate(_network)

    _target_acc = 0.70
    counting_dataset = [
        {
            'dataset': 'auburn',
            'subsets': [
                'fri005', 'sat001am', 'sat005pm',
                'fri000', 'fri001', 'fri003', 'fri007', 'fri009', 'fri011',
                'fri013', 'fri015', 'fri019', 'sat000pm', 'sat001pm', 'sat002pm',
                'sat006pm'
            ],
            'queries': [{'target_acc': _target_acc}],
            'properties': {'fps': 30},
        },
        {
            'dataset': 'jacksonhole',
            'subsets': [
                'raw000', 'raw019', 'raw038',
                'raw001', 'raw002', 'raw003', 'raw004', 'raw005', 'raw006',
                'raw007', 'raw008', 'raw009', 'raw010', 'raw011', 'raw012',
                'raw013', 'raw014', 'raw015', 'raw016', 'raw017', 'raw018',
            ],
            'queries': [{'target_acc': _target_acc}],
            'properties': {'fps': 30},
        },
        {
            'dataset': 'lagrange',
            'subsets': [
                'raw000', 'raw005', 'raw022',
                'raw001', 'raw002', 'raw003', 'raw004', 'raw006', 'raw007',
                'raw009', 'raw010', 'raw011', 'raw012', 'raw018', 'raw019',
                'raw020', 'raw021', 'raw023',
            ],
            'queries': [{'target_acc': _target_acc}],
            'properties': {'fps': 30},
        },
        {
            'dataset': 'southampton',
            'subsets': [
                'raw001', 'sat005', 'sat009pm',
                'raw000', 'raw008', 'raw013', 'raw019', 'raw031', 'raw038',
                'raw042', 'raw043', 'raw049', 'sat000', 'sat000pm', 'sat001pm',
            ],
            'queries': [{'target_acc': _target_acc}],
            'properties': {'fps': 30},
        },
        {
            'dataset': 'clintonthomas',
            'subsets': [
                'sun03001', 'sun11005', 'tue19000'
            ],
            'queries': [{'target_acc': _target_acc}],
            'properties': {'fps': 30},
        }
    ]
    # Optimal(counting_dataset, typ='coco', classes=[0]).simulate(dataset_names=['jacksonhole'], verbose=True)
    # Optimal(counting_dataset, typ='coco', classes=[2]).simulate()
    # Optimal(counting_dataset, typ='tagging', classes=[0]).simulate()
    # Optimal(counting_dataset, typ='tagging', classes=[2]).simulate()
    # Optimal(counting_dataset, typ='counting', classes=[0]).simulate()
    # Optimal(counting_dataset, typ='counting', classes=[2]).simulate()

    # Optimal(counting_dataset, typ='counting', classes=[0]).simulate(dataset_names=['jacksonhole'])
    # Optimal(counting_dataset, typ='counting', classes=[0]).simulate(dataset_names=['lagrange'])
    # Optimal(counting_dataset, typ='counting', classes=[0]).simulate(dataset_names=['southampton'])
    # Optimal(counting_dataset, typ='counting', classes=[0]).simulate(dataset_names=['auburn'])
    # Optimal(counting_dataset, typ='counting', classes=[0]).simulate(dataset_names=['clintonthomas'])

    Optimal(counting_dataset, typ='counting', classes=[2]).simulate(dataset_names=['jacksonhole'])
    Optimal(counting_dataset, typ='counting', classes=[2]).simulate(dataset_names=['lagrange'])
    Optimal(counting_dataset, typ='counting', classes=[2]).simulate(dataset_names=['southampton'])
    Optimal(counting_dataset, typ='counting', classes=[2]).simulate(dataset_names=['auburn'])
    Optimal(counting_dataset, typ='counting', classes=[2]).simulate(dataset_names=['clintonthomas'])

    # Optimal(counting_dataset, typ='coco', classes=[0]).simulate(dataset_names=['jacksonhole'])
    # Optimal(counting_dataset, typ='coco', classes=[0]).simulate(dataset_names=['lagrange'])
    # Optimal(counting_dataset, typ='coco', classes=[0]).simulate(dataset_names=['southampton'])
    # Optimal(counting_dataset, typ='coco', classes=[0]).simulate(dataset_names=['auburn'])
    # Optimal(counting_dataset, typ='coco', classes=[0]).simulate(dataset_names=['clintonthomas'])
    #
    Optimal(counting_dataset, typ='coco', classes=[2]).simulate(dataset_names=['jacksonhole'])
    Optimal(counting_dataset, typ='coco', classes=[2]).simulate(dataset_names=['lagrange'])
    Optimal(counting_dataset, typ='coco', classes=[2]).simulate(dataset_names=['southampton'])
    Optimal(counting_dataset, typ='coco', classes=[2]).simulate(dataset_names=['auburn'])
    Optimal(counting_dataset, typ='coco', classes=[2]).simulate(dataset_names=['clintonthomas'])
    #
    # Optimal(counting_dataset, typ='tagging', classes=[0]).simulate(dataset_names=['jacksonhole'])
    # Optimal(counting_dataset, typ='tagging', classes=[0]).simulate(dataset_names=['lagrange'])
    # Optimal(counting_dataset, typ='tagging', classes=[0]).simulate(dataset_names=['southampton'])
    # Optimal(counting_dataset, typ='tagging', classes=[0]).simulate(dataset_names=['auburn'])
    # Optimal(counting_dataset, typ='tagging', classes=[0]).simulate(dataset_names=['clintonthomas'])
    #
    # Optimal(counting_dataset, typ='tagging', classes=[2]).simulate(dataset_names=['jacksonhole'])
    # Optimal(counting_dataset, typ='tagging', classes=[2]).simulate(dataset_names=['lagrange'])
    # Optimal(counting_dataset, typ='tagging', classes=[2]).simulate(dataset_names=['southampton'])
    # Optimal(counting_dataset, typ='tagging', classes=[2]).simulate(dataset_names=['auburn'])
    # Optimal(counting_dataset, typ='tagging', classes=[2]).simulate(dataset_names=['clintonthomas'])
