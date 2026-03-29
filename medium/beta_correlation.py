import numpy as np
from scipy import stats

data = [
    ('Co-Physics',  0.93, 2.09),
    ('Am-Photo',    0.83, 1.75),
    ('Co-CS',       0.81, 2.08),
    ('Cora',        0.81, 1.19),
    ('PubMed',      0.80, 0.31),
    ('Am-Comp',     0.78, 2.12),
    ('CiteSeer',    0.74, 2.38),
    ('Deezer',      0.53, -0.57),
    ('Film',        0.31, -0.99),
    ('Chameleon',   0.24, 3.08),
    ('Squirrel',    0.21, 3.53),
]

hs = [d[1] for d in data]
betas = [d[2] for d in data]

r_all, p_all = stats.pearsonr(hs, betas)
print("ALL 11: Pearson r = %.4f, p = %.4f" % (r_all, p_all))

hs9 = [d[1] for d in data if d[0] not in ('Chameleon', 'Squirrel')]
betas9 = [d[2] for d in data if d[0] not in ('Chameleon', 'Squirrel')]
r9, p9 = stats.pearsonr(hs9, betas9)
print("9 datasets (excl Cham/Sqrl): Pearson r = %.4f, p = %.4f" % (r9, p9))

rho_all, sp_all = stats.spearmanr(hs, betas)
print("ALL 11: Spearman rho = %.4f, p = %.4f" % (rho_all, sp_all))

rho9, sp9 = stats.spearmanr(hs9, betas9)
print("9 datasets: Spearman rho = %.4f, p = %.4f" % (rho9, sp9))

print()
for name, h, beta in data:
    pred = "pos" if h > 0.5 else "neg"
    actual = "pos" if beta > 0 else "neg"
    match = "Y" if pred == actual else "N"
    print("  %-15s h=%.2f b=%+.2f  pred=%s actual=%s %s" % (name, h, beta, pred, actual, match))
