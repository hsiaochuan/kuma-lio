#!/usr/bin/env python3
"""Convert TartanGround pose files (x y z qx qy qz qw) to TUM format by
prepending a timestamp column (start 1000.0, step 0.1)."""

import os

import numpy as np

ROOT = '/home/hsiaochuan/Downloads/tartanground'
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gt')
T0 = 1000.0
TIME_STEP = 0.1

pose_txt = [
    'Downtown/Data_anymal/P2000/pose_lcam_front.txt',
    'ModernCityDowntown/Data_anymal/P2000/pose_lcam_front.txt',
    'ModularNeighborhood/Data_anymal/P2000/pose_lcam_front.txt',
    'NordicHarbor/Data_anymal/P2000/pose_lcam_front.txt',
    'OldTownFall/Data_anymal/P2000/pose_lcam_front.txt',
    'OldTownSummer/Data_anymal/P2000/pose_lcam_front.txt',

    'AbandonedCable/Data_diff/P1000/pose_lcam_front.txt',
    'CarWelding/Data_diff/P1000/pose_lcam_front.txt',
    'GothicIsland/Data_diff/P1000/pose_lcam_front.txt',
    'JapaneseAlley/Data_diff/P1000/pose_lcam_front.txt',
    'Supermarket/Data_diff/P1000/pose_lcam_front.txt',
]

os.makedirs(OUT_DIR, exist_ok=True)
for rel in pose_txt:
    pose = np.loadtxt(os.path.join(ROOT, rel))
    time = T0 + TIME_STEP * np.arange(len(pose))
    env, version, traj, _ = rel.split('/')
    out = os.path.join(OUT_DIR, f'{env}_{traj}_gt.txt')
    np.savetxt(out, np.column_stack([time, pose]), fmt='%.6f')
    print(f'{rel} -> {out} ({len(pose)} poses)')
