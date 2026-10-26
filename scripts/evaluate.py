"""Rescore stored predictions and generate all plot input tables."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from src.stats19.core import score
def evaluate(out):
    choices=json.loads((out/'frozen_choices.json').read_text());rows=[];prs=[];cals=[]
    for split in ['validation','test']:
        file=out/f'{split}_predictions.csv.gz'
        if not file.exists():continue
        pred=pd.read_csv(file,float_precision='round_trip')
        for model,t in choices['thresholds'].items():
            y,p=pred.y.to_numpy(),pred[model].to_numpy()
            for policy,cutoff in [('validation_f1' if model!='majority' else 'majority',t),('default_0.5',.5)]:
                rows.append(dict(split=split,model=model,policy=policy,threshold=cutoff,**score(y,p,cutoff)))
            precision,recall,thresholds=precision_recall_curve(y,p)
            # Exact step endpoints, not an interpolated display approximation.
            for i in range(len(precision)):prs.append(dict(split=split,model=model,precision=precision[i],recall=recall[i]))
            bins=np.minimum((p*10).astype(int),9)
            for b in range(10):
                mask=bins==b;n=int(mask.sum())
                cals.append(dict(split=split,model=model,bin=b,n=n,mean_probability=p[mask].mean() if n else np.nan,severe_fraction=y[mask].mean() if n else np.nan))
    metrics=pd.DataFrame(rows);metrics.to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(prs).to_csv(out/'pr_curves.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(cals).to_csv(out/'calibration.csv',index=False)
    increments=[]
    for split in metrics.split.unique():
        m=metrics[(metrics.split==split)&metrics.policy.isin(['validation_f1','majority'])].set_index('model')
        for compare,base in [('logistic_B','logistic_A'),('random_forest_B','logistic_B'),('hist_gradient_B','logistic_B')]:
            for metric in ['precision','recall','f1','average_precision','pr_auc','roc_auc','brier']:
                increments.append(dict(split=split,comparison=compare+' minus '+base,metric=metric,delta=m.loc[compare,metric]-m.loc[base,metric]))
    pd.DataFrame(increments).to_csv(out/'increments.csv',index=False)
    print(metrics.to_string(index=False))
    return metrics
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();evaluate(ROOT/'results'/a.run_id)
