"""Fixed models; only 2024 validation selects thresholds. No test-set tuning."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,datetime,hashlib,json,platform,time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from threadpoolctl import threadpool_limits
from src.stats19.core import SEED,A,B,features,preprocessing,threshold,score
def models():
    return {
      'majority':('B',None),
      'logistic_A':('A',LogisticRegression(C=1,max_iter=2000,solver='lbfgs')),
      'logistic_B':('B',LogisticRegression(C=1,max_iter=2000,solver='lbfgs')),
      'random_forest_B':('B',RandomForestClassifier(n_estimators=150,max_depth=12,min_samples_leaf=20,max_features='sqrt',n_jobs=1,random_state=SEED)),
      'hist_gradient_B':('B',HistGradientBoostingClassifier(learning_rate=.05,max_iter=150,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=1,early_stopping=False,random_state=SEED))
    }
def load(year):
    return pd.read_csv(ROOT/f'data/processed/clean_{year}.csv.gz',dtype={c:str for c in B if c!='speed_limit'})
def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['pilot','final'],required=True);p.add_argument('--run-id',required=True);a=p.parse_args()
    out=ROOT/'results'/a.run_id;out.mkdir(parents=True,exist_ok=False)
    begin=datetime.datetime.now(datetime.timezone.utc).isoformat();d=load(2024);split=np.load(ROOT/'data/processed/split.npz')
    train=d.iloc[split['train']].reset_index(drop=True);val=d.iloc[split['validation']].reset_index(drop=True)
    if a.mode=='pilot':
        ix,_=train_test_split(np.arange(len(train)),train_size=15000,stratify=train.y,random_state=SEED)
        small=train.iloc[ix];tr,va=train_test_split(np.arange(len(small)),test_size=.2,stratify=small.y,random_state=SEED)
        train=small.iloc[tr].reset_index(drop=True);val=small.iloc[va].reset_index(drop=True)
    registry=models();fitted={};thresholds={};validation={'y':val.y.to_numpy()};logs=[];curves=[]
    for name,(group,estimator) in registry.items():
        start=time.perf_counter();X=features(train,group);V=features(val,group)
        if estimator is None:
            model=None;prob=np.full(len(val),train.y.mean());t=.5
        else:
            model=Pipeline([('preprocess',preprocessing(list(X.columns))),('model',estimator)])
            with threadpool_limits(limits=2):model.fit(X,train.y);prob=model.predict_proba(V)[:,1]
            t=threshold(val.y,prob)
        fitted[name]=model;validation[name]=prob;thresholds[name]=t
        logs.append(dict(model=name,feature_set=group,train_n=len(train),validation_n=len(val),train_severe=int(train.y.sum()),validation_severe=int(val.y.sum()),fit_predict_seconds=time.perf_counter()-start,threshold=t,parameters=estimator.get_params() if estimator is not None else {'class':'constant training prevalence','p':float(train.y.mean())}))
        print(json.dumps(logs[-1]),flush=True)
        if model is not None:
            for tgrid in np.arange(.05,1,.05):curves.append(dict(model=name,threshold=tgrid,**score(val.y,prob,tgrid)))
    # Write and timestamp frozen choices BEFORE loading any 2025 model input.
    (out/'frozen_choices.json').write_text(json.dumps({'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'mode':a.mode,'thresholds':thresholds,'models':logs,'test_used_for_selection':False},indent=2)+'\n')
    pd.DataFrame(validation).to_csv(out/'validation_predictions.csv.gz',index=False,float_format='%.17g',compression={'method':'gzip','mtime':0})
    pd.DataFrame(curves).to_csv(out/'validation_thresholds.csv',index=False)
    if a.mode=='final':
        test=load(2025);predictions={'y':test.y.to_numpy()}
        for name,(group,_) in registry.items():
            model=fitted[name]
            with threadpool_limits(limits=2):predictions[name]=model.predict_proba(features(test,group))[:,1] if model is not None else np.full(len(test),train.y.mean())
        pd.DataFrame(predictions).to_csv(out/'test_predictions.csv.gz',index=False,float_format='%.17g',compression={'method':'gzip','mtime':0})
        # Interpretation on a fixed subset of 2024 validation; no 2025 feature selection.
        ix,_=train_test_split(np.arange(len(val)),train_size=5000,stratify=val.y,random_state=SEED)
        importance=[]
        for name in ['logistic_B','random_forest_B','hist_gradient_B']:
            with threadpool_limits(limits=2):imp=permutation_importance(fitted[name],features(val.iloc[ix],'B'),val.iloc[ix].y,scoring='average_precision',n_repeats=3,random_state=SEED,n_jobs=1)
            for j,col in enumerate(B):
                for repeat,value in enumerate(imp.importances[j]):importance.append(dict(model=name,feature=col,repeat=repeat,ap_drop=value,n=5000))
        pd.DataFrame(importance).to_csv(out/'permutation_importance.csv',index=False)
        lr=fitted['logistic_B'];names=lr['preprocess'].get_feature_names_out()
        pd.DataFrame({'encoded_feature':names,'coefficient':lr['model'].coef_[0]}).to_csv(out/'logistic_coefficients.csv',index=False)
        (out/'logistic_intercept.json').write_text(json.dumps({'intercept':float(lr['model'].intercept_[0]),'coding':'all category levels retained with L2 penalty; numeric speed standardized; coefficients are conditional associations, not causal effects'},indent=2)+'\n')
    source_hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for folder in ['src','scripts'] for f in (ROOT/folder).rglob('*.py')}
    (out/'run.json').write_text(json.dumps({'start_utc':begin,'end_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'platform':platform.platform(),'python':platform.python_version(),'seed':SEED,'mode':a.mode,'source_sha256':source_hashes,'data_manifest_sha256':hashlib.sha256((ROOT/'data/manifest.json').read_bytes()).hexdigest(),'split_sha256':hashlib.sha256((ROOT/'data/processed/split.csv.gz').read_bytes()).hexdigest(),'test_predictions_order':'sorted collision_index in clean_2025; IDs deliberately omitted from public predictions'},indent=2)+'\n')
if __name__=='__main__':main()
