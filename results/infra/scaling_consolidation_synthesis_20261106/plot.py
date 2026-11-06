"""Render published aggregate results; no fitting or new statistical inference."""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/wearusfm-synthesis-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

D = json.loads((HERE / 'data.json').read_text())
BLUE, TEAL, ORANGE, GRAY = '#3265a8', '#008577', '#cc6e24', '#626b76'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.labelcolor': '#263342', 'text.color': '#263342',
                     'axes.titleweight': 'bold', 'figure.facecolor': 'white'})


def interval(ax, x, ci, y, color, marker='o'):
    assert ci[0] <= x <= ci[1]
    ax.errorbar(x, y, xerr=[[x-ci[0]], [ci[1]-x]], fmt=marker,
                color=color, markersize=7, capsize=4, elinewidth=2, zorder=3)


def save(fig, name):
    fig.savefig(HERE / (name + '.png'), dpi=180, bbox_inches='tight')
    fig.savefig(HERE / (name + '.svg'), bbox_inches='tight')
    plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={'width_ratios': [1.35, 1]})
fig.subplots_adjust(top=.79, bottom=.22, left=.12, right=.97, wspace=.42)
fig.suptitle('1 · Scaling: il vantaggio emerge nella classificazione', x=.035, ha='left', y=.98, fontsize=19, weight='bold')
fig.text(.035, .915, '30M e 300M · tokenizer 100 ms + GELU · 5.000 passi · encoder congelati', fontsize=12)
ax = axes[0]
keys = ['epn612', 'ninapro_db2', 'ninapro_db3', 'ninapro_db6', 'uci_emg', 'p1', 'p2']
labels = ['EPN612 (primario)', 'NinaPro DB2', 'NinaPro DB3', 'NinaPro DB6', 'UCI', 'P1 (tre NinaPro)', 'P2 (EPN + UCI)']
for y, key in zip(range(6, -1, -1), keys):
    row = D['scaling']['classification'][key]['contrasts']['natural_300m_minus_30m']
    interval(ax, row['delta_pp'], row['ci95_pp'], y, TEAL if key=='epn612' else BLUE)
    ax.text(3.75, y, f"{row['delta_pp']:+.2f}", va='center', ha='left', fontsize=10)
ax.axvline(0, color=GRAY, lw=1, ls='--')
ax.axhline(1.5, color='#dfe4ea', lw=1)
ax.set(yticks=range(6,-1,-1), yticklabels=labels, xlim=(-3,4.35), ylim=(-.5,6.6),
       xlabel='300M − 30M · punti di balanced accuracy')
ax.set_title('Classificazione: a destra è meglio', fontsize=12, pad=15)
ax.grid(axis='x', alpha=.15)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='y', length=0)

ax = axes[1]
p = D['scaling']['pose']; diff = p['delta_E_dyn']
interval(ax, diff, p['ci95'], 0, ORANGE)
ax.axvline(0, color=GRAY, lw=1, ls='--')
ax.axvline(-.02*p['control_E_dyn'], color=TEAL, lw=1.5, ls=':')
ax.set(xlim=(-.022,.008), ylim=(-.8,.8), yticks=[], xlabel='300M − 30M · errore dinamico normalizzato')
ax.set_title('Dinamica della posa: a sinistra è meglio', fontsize=12, pad=15)
ax.text(.03,.93, f"Errore 30M:  {p['control_E_dyn']:.5f}\nErrore 300M: {p['candidate_E_dyn']:.5f}", transform=ax.transAxes, fontsize=12, va='top', bbox={'facecolor':'white','edgecolor':'none','pad':2})
ax.text(-.018, -.40, 'Riferimento pratico\n−2% di errore', ha='center', color=TEAL, fontsize=10, bbox={'facecolor':'white','edgecolor':'none','pad':2})
ax.text(.03,.09, f"Δ = {diff:+.5f}\nIC95% [{p['ci95'][0]:+.5f}; {p['ci95'][1]:+.5f}]", transform=ax.transAxes, fontsize=11, bbox={'facecolor':'white','edgecolor':'none','pad':2})
ax.set_xticks([-.02,-.01,0])
ax.grid(axis='x', alpha=.15)
ax.spines['left'].set_visible(False)
fig.text(.035,.10, 'Punti e barre: differenze appaiate e IC95% per soggetto. Un seme; popolazioni di sviluppo.', fontsize=10, color=GRAY)
fig.text(.035,.055, 'P1/P2 aggregano dataset già mostrati, non sono repliche indipendenti. Posa: 15 soggetti, lettore ridge sui latenti mediati.', fontsize=10, color=GRAY)
save(fig, 'scaling_5000')

fig, axes = plt.subplots(1, 2, figsize=(13,6.2), gridspec_kw={'width_ratios':[1.25,1]})
fig.subplots_adjust(top=.79, bottom=.24, left=.075, right=.97, wspace=.43)
fig.suptitle('2 · EPN: il vantaggio del pretraining rimane, ma si riduce', x=.035, ha='left', y=.98, fontsize=18, weight='bold')
fig.text(.035,.915,'Stesso 30M preaddestrato a 19.618 passi · confronto fra ricette downstream a 5 e 15 epoche',fontsize=12)
ax=axes[0]; keys=['frozen','finetune','scratch']; x=np.arange(3); width=.35
for epoch, shift, color in [('5',-.19,BLUE),('15',.19,TEAL)]:
    cells=D['epn'][epoch]['cells']; values=[cells[k]['balanced_accuracy_percent'] for k in keys]
    lo=[v-cells[k]['ci95_percent'][0] for v,k in zip(values,keys)]
    hi=[cells[k]['ci95_percent'][1]-v for v,k in zip(values,keys)]
    ax.bar(x+shift, values, width, color=color, label=f'{epoch} epoche', yerr=[lo,hi], capsize=3, error_kw={'elinewidth':1})
    for xx,v in zip(x+shift,values):ax.text(xx,v+3.1,f'{v:.2f}',ha='center',fontsize=10)
ax.set(xticks=x, xticklabels=['Congelato\nLN + lineare','Fine-tuning\nencoder + testa','Da zero\nstessa architettura'], ylim=(0,107), ylabel='Balanced accuracy (%)')
ax.set_title('Prestazioni sul compito EPN612',fontsize=12,pad=15)
ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
ax.legend(loc='upper left',frameon=False)
ax=axes[1]
for epoch,y,color in [('5',1,BLUE),('15',0,TEAL)]:
    row=D['epn'][epoch]['contrasts']['finetune_minus_scratch']
    interval(ax,row['delta_pp'],row['ci95_pp'],y,color)
    ax.text(row['delta_pp'],y+.22,f"{row['delta_pp']:+.2f} pp",ha='center',color=color,weight='bold')
ax.axvline(0,color=GRAY,lw=1,ls='--')
ax.set(yticks=[1,0],yticklabels=['5 epoche','15 epoche'],xlim=(-.1,2.12),ylim=(-.5,1.55),xlabel='Fine-tuning − da zero · punti percentuali')
ax.set_title('Quanto aggiunge il pretraining?',fontsize=12,pad=15)
ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0);ax.grid(axis='x',alpha=.15)
gap=D['epn']['recipe_comparison']['transfer_gap15_minus_gap5']
fig.text(.035,.12,f"Il vantaggio si restringe di {abs(gap['delta_pp']):.2f} pp; IC95% della differenza [{gap['ci95_pp'][0]:+.2f}; {gap['ci95_pp'][1]:+.2f}].",fontsize=11)
fig.text(.035,.075,'314 soggetti di sviluppo · IC95% per soggetto · un seme. Le due ricette cambiano anche l’orizzonte dello schedule.',fontsize=10,color=GRAY)
fig.text(.035,.03,'Il congelato qui usa una nuova testa: il riferimento storico con logistica raggiunge 82,58% (ricetta diversa, confronto descrittivo).',fontsize=10,color=GRAY)
save(fig,'epn_transfer_5_15')
print('Rendered scaling_5000 and epn_transfer_5_15, PNG + SVG.')
