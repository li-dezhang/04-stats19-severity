"""Independent checks of saved predictions, tables, figures and optional repeat."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,hashlib,json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_curve,auc
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',default='final-20261021');p.add_argument('--compare');a=p.parse_args()
    r=ROOT/'results'/a.run_id;m=pd.read_csv(r/'metrics.csv',float_precision='round_trip');checks=[]
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    available=all((ROOT/f['path']).exists() for f in manifest['files'])
    if available:
        for f in manifest['files']:assert sha(ROOT/f['path'])==f['sha256']
        checks.append('raw source SHA-256 unchanged')
    else:checks.append('raw files absent: checksum check skipped; stored-prediction verification does not require downloads')
    for split in m.split.unique():
        pred=pd.read_csv(r/f'{split}_predictions.csv.gz',float_precision='round_trip');y=pred.y.to_numpy().astype(bool)
        assert len(pred)==(101525 if split=='test' else len(pred))
        for _,row in m[m.split==split].iterrows():
            probabilities=pred[row.model].to_numpy();positive=probabilities>=row.threshold
            tp=int(np.sum(y&positive));fp=int(np.sum(~y&positive));fn=int(np.sum(y&~positive));tn=int(np.sum(~y&~positive))
            assert [tp,fp,fn,tn]==[row.tp,row.fp,row.fn,row.tn]
            assert tp+fp+fn+tn==len(y)==row.n
            precision=tp/(tp+fp) if tp+fp else np.nan;recall=tp/(tp+fn)
            pr,re,_=precision_recall_curve(y,probabilities)
            values=[precision,recall,2*tp/(2*tp+fp+fn),fp/(fp+tn),average_precision_score(y,probabilities),auc(re,pr),roc_auc_score(y,probabilities),np.mean((probabilities-y)**2)]
            np.testing.assert_allclose(values,row[['precision','recall','f1','fpr','average_precision','pr_auc','roc_auc','brier']].astype(float),rtol=0,atol=1e-14,equal_nan=True)
        cal=pd.read_csv(r/'calibration.csv')
        for model in pred.columns.drop('y'):
            assert cal[(cal.split==split)&cal.model.eq(model)].n.sum()==len(pred)
    checks.append('all saved confusion counts and metrics independently recomputed')
    inputs=ROOT/'figures'/a.run_id/'inputs.json'
    if inputs.exists():
        for name,digest in json.loads(inputs.read_text()).items():assert sha(ROOT/name)==digest,name
        checks.append('figure input CSV hashes match current tables')
    if a.compare:
        other=ROOT/'results'/a.compare
        for name in ['metrics.csv','test_predictions.csv.gz','validation_predictions.csv.gz','permutation_importance.csv','logistic_coefficients.csv']:
            pd.testing.assert_frame_equal(pd.read_csv(r/name,float_precision='round_trip'),pd.read_csv(other/name,float_precision='round_trip'),check_exact=True)
        checks.append('independent full retraining: predictions, metrics, importance and coefficients exactly equal')
    result={'run_id':a.run_id,'checks':checks,'status':'PASS','raw_hash_check':available,'metrics_sha256':sha(r/'metrics.csv')}
    (r/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
