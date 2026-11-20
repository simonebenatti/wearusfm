import json, sys
import torch
from torch import nn
r = json.load(open('rec/LONG-FIT-G49046-pose.json'))
for cell in ('post_ridge', 'pre_ridge'):
    fu = r['cells'][cell]['fit']['selection']['fold_users']
    u = sorted(fu[0] + fu[1])
    print(cell, 'fold sizes', len(fu[0]), len(fu[1]), 'distinti', len(set(u)), 'alternati ordinati:', fu[0] == u[::2] and fu[1] == u[1::2])
val_users = sorted(r['cells']['post_ridge']['analysis']['groups']['all']['centered']['per_user'])
print('utenti val', len(val_users), 'intersezione con i 40 dei fold:', len(set(val_users) & set(u)))
v = json.load(open('rec/LONG-G49046-pose-val.json')); t = json.load(open('rec/LONG-G49046-pose-train.json'))
for n, x in (('train', t), ('val', v)):
    cg = x['checkpoint_guard']
    print(n, 'checkpoint', cg['checkpoint_sha256'][:12], cg['checkpoint_step'], cg['which'], 'stats', cg['checkpoint_pose_stats_sha256'][:12],
          'cache', x['artifacts']['cache']['bytes'], x['artifacts']['cache']['sha256'][:12], x['details']['n_windows'], x['details']['n_users'], x['details']['n_recordings'])
# parametri delle teste con le specifiche di PROT:33-47 (codice mio)
D, J = 384, 20
heads = {
 'L0': nn.Sequential(nn.LayerNorm(D), nn.Linear(D, J)),
 'MLP': nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 96), nn.Linear(96, 96), nn.Linear(96, 96), nn.Linear(96, 64), nn.Linear(64, 32), nn.Linear(32, J)),
 'L13': nn.Sequential(nn.LayerNorm(D), nn.Conv1d(D, J, 13, padding=6)),
 'TCN': nn.Sequential(nn.LayerNorm(D), nn.Conv1d(D, 64, 1), nn.Conv1d(64, 64, 3, padding=1), nn.Conv1d(64, 64, 3, padding=2, dilation=2),
                      nn.Conv1d(64, 64, 3, padding=3, dilation=3), nn.Conv1d(64, 32, 1), nn.Conv1d(32, J, 1))}
for k, m in heads.items():
    tot = sum(p.numel() for p in m.parameters()); w = sum(p.numel() for n_, p in m.named_parameters() if p.ndim >= 2)
    print(k, 'parametri', tot, 'pesi di matrici/kernel', w)
# soglia del candidato: 2% sotto ciascun riferimento storico
E_r, E_h = 0.7859020004638595, 0.7786289843513977
print('E_TCN massimo per beneficio >= 2% contro ridge', E_r * 0.98, 'contro testa', E_h * 0.98, '; rispetto alla ridge il secondo vale', 100 * (1 - E_h * 0.98 / E_r), '%')
