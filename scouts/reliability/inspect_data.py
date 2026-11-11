import sys, numpy as np
sys.path.insert(0,'/home/claude/HEPA')
from exp_shared_eps import load
from hepa.utils import get_context
for n in "FD001,FD003,MBA,PSM,ETTm1".split(","):
    b,norm=load(n)
    print(n, 'norm',norm,'ctx',get_context(n),'horizons',b['horizons'], 'keys',[k for k in b])
    for split in ['ft_train','ft_val','ft_test']:
        es=b[split]
        L=[len(e['test']) for e in es]
        pos=[int(np.asarray(e['labels']).sum()) for e in es]
        # number of event onsets
        on=[int((np.diff(np.asarray(e['labels']).astype(int))==1).sum()+int(e['labels'][0]==1)) for e in es]
        print('  ',split,'n_ent',len(es),'len min/med/max',min(L),int(np.median(L)),max(L),'sum',sum(L),'pos_frac',round(sum(pos)/sum(L),4),'onsets',sum(on), 'ekeys',list(es[0].keys()))
