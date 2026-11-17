"""Reviewer arithmetic (dimension 6): parameters, ratios, level-B cost, budget table, fit counts."""
from fractions import Fraction as F
# --- parameter counts (weights + biases; LayerNorm affine = 2*d)
def lin(i, o, k=1): return i*o*k + o
LN = 2*384
L0 = LN + lin(384, 20)
MLP_w = [384, 96, 96, 96, 64, 32, 20]
MLP = LN + sum(lin(a, b) for a, b in zip(MLP_w, MLP_w[1:]))
MLP_weights = sum(a*b for a, b in zip(MLP_w, MLP_w[1:]))
L13 = LN + lin(384, 20, 13)
TCN = LN + lin(384, 64) + 3*lin(64, 64, 3) + lin(64, 32) + lin(32, 20)
TCN_weights = 384*64 + 3*64*64*3 + 64*32 + 32*20
print('L0', L0, 'MLP', MLP, 'L13', L13, 'TCN', TCN)
print('MLP weights', MLP_weights, 'TCN weights', TCN_weights, 'bias diff MLP-TCN', MLP - TCN)
# --- level B statements
P30 = 28362013
print('TCN / 30M = %.4f%%' % (100*TCN/P30), '| TCN / PoseHead = %.3f' % (TCN/L0), '| PoseHead / 30M = %.4f%%' % (100*L0/P30))
print('2 x 16,94 =', 2*F('16.94'), '=', float(2*F('16.94')), '| 2 x 16,944444 = %.6f' % (2*16.944444))
cont = [7955.654906749725, 7145.292625427246, 7536.233420848846]  # summary.elapsed_s, world_size 4 (receipts 30000, 39237, 49046)
gpuh_cont = sum(cont)*4/3600
print('continuation 19618->49046 (elapsed_s x 4 GPU): %.4f GPUh' % gpuh_cont)
one49 = 16.944444 + gpuh_cont
print('one T100-G training 0->49046 ~ %.4f GPUh ; two ~ %.4f GPUh' % (one49, 2*one49))
print('proportional: 16,944444 x 49046/19618 = %.4f ; x2 = %.4f' % (16.944444*49046/19618, 2*16.944444*49046/19618))
print('local hours (billing 8/GPUh): B@19618 %.1f ; B@49046 ~ %.1f' % (33.88*8, 2*one49*8))
# --- budget table of paragraph 9
cpu_only = {'checkout 2CPUx15min': F(2)*F(15,60), 'bench CPU 8x15min': F(8)*F(15,60), 'refs/metrics 8x1h': F(8)*1,
            'collect 2x15min': F(2)*F(15,60), 'reserve': F(1)}
gpu = {'bench GPU 15min': F(15,60), 'fit 3h': F(3), 'inference 30min': F(30,60), 'reserve': F(1,4)}
print('CPU-only coreh:', {k: float(v) for k, v in cpu_only.items()}, 'sum', float(sum(cpu_only.values())))
print('GPUh:', {k: float(v) for k, v in gpu.items()}, 'sum', float(sum(gpu.values())))
tot = sum(cpu_only.values()) + 8*sum(gpu.values())
print('total allocated coreh = 12 + 8*4 =', float(tot), '| local hours at billing 8 per GPU-job hour + 1 per CPU core-hour =', float(8*sum(gpu.values()) + sum(cpu_only.values())))
# --- fits
print('CV fits 5 heads x 2 LR x 2 folds =', 5*2*2, '| refits 5 x 3 seeds =', 5*3, '| total', 5*2*2 + 5*3)
# --- receptive field
print('RF = 1 + 2*(1+2+3) =', 1 + 2*(1+2+3), 'patches; centers 6..153 =', 153-6+1, '; +/-', 6*25, 'ms')
# --- prediction storage
b = 18548*160*20*4
print('val predictions FP32 per head (160 patches):', b, 'bytes; x15 =', 15*b, '= %.4f GiB' % (15*b/2**30))
print('cache total bytes train+val:', 474412856 + 4593977338, '= %.4f GiB' % ((474412856 + 4593977338)/2**30))
print('all five inputs bytes:', 474412856 + 4593977338 + 59367 + 418440754 + 234048726, '= %.4f GiB' % ((474412856 + 4593977338 + 59367 + 418440754 + 234048726)/2**30))
