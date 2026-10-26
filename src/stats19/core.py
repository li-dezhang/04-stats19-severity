"""Dictionary-based cleaning and model inputs."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix,precision_recall_curve,average_precision_score,roc_auc_score,auc,brier_score_loss
SEED=20261021
A=['speed_limit','road_type','junction_detail','first_road_class']
B=A+['light_conditions','weather_conditions','road_surface_conditions','day_of_week','hour']
VALID={'road_type':{1,2,3,6,7,9,12},'junction_detail':{0,13,16,17,18,19,99},'first_road_class':{1,2,3,4,5,6},'light_conditions':{1,4,5,6,7},'weather_conditions':set(range(1,10)),'road_surface_conditions':{1,2,3,4,5,6,7,9},'day_of_week':set(range(1,8))}
UNKNOWN={'road_type':9,'junction_detail':99,'weather_conditions':9,'road_surface_conditions':9}
def target(s):
    if not s.isin([1,2,3]).all():raise ValueError('Invalid target')
    return s.isin([1,2]).astype(np.int8)
def clean(df,year):
    if not df.collision_year.eq(year).all():raise ValueError('Unexpected year')
    if df.collision_index.isna().any() or df.collision_index.duplicated().any():raise ValueError('Duplicate/missing ID')
    valid_target=df.collision_severity.isin([1,2,3]);out=df.loc[valid_target].copy().sort_values('collision_index').reset_index(drop=True);rows=[]
    for col,codes in VALID.items():
        s=pd.to_numeric(out[col],errors='coerce');missing=s.isna()|s.eq(-1);unknown=s.eq(UNKNOWN.get(col,-999));invalid=~missing&~s.isin(codes)
        v=s.astype('Int64').astype(str);v[missing|invalid]='__MISSING__';v[unknown]='__UNKNOWN__';out[col]=v
        rows.append(dict(year=year,feature=col,rows=len(out),missing=int(missing.sum()),unknown=int(unknown.sum()),invalid=int(invalid.sum()),partial_unknown=int(s.eq(7).sum()) if col=='light_conditions' else 0))
    s=pd.to_numeric(out.speed_limit,errors='coerce');missing=s.isna()|s.eq(-1);unknown=s.eq(99);invalid=~missing&~unknown&~s.isin([20,30,40,50,60,70]);out['speed_limit']=s.mask(missing|unknown|invalid,np.nan)
    rows.append(dict(year=year,feature='speed_limit',rows=len(out),missing=int(missing.sum()),unknown=int(unknown.sum()),invalid=int(invalid.sum()),partial_unknown=0))
    time=out.time.astype('string');valid=time.str.fullmatch(r'(?:[01][0-9]|2[0-3]):[0-5][0-9]',na=False);out['hour']=time.str[:2].where(valid,'__MISSING__')
    rows.append(dict(year=year,feature='hour',rows=len(out),missing=int(time.isna().sum()),unknown=0,invalid=int((~valid&time.notna()).sum()),partial_unknown=0))
    out['y']=target(out.collision_severity)
    return out,rows,dict(year=year,raw_rows=len(df),eligible_rows=len(out),invalid_targets=int((~valid_target).sum()),duplicate_ids=0,severe=int(out.y.sum()),slight=int((out.y==0).sum()),fatal=int(out.collision_severity.eq(1).sum()),serious=int(out.collision_severity.eq(2).sum()),severe_fraction=float(out.y.mean()))
def features(df,which):return df[{'A':A,'B':B}[which]].copy()
def split_2024(df,seed=SEED):
    if not df.collision_year.eq(2024).all():raise ValueError('Only 2024 may be split')
    tr,va=train_test_split(np.arange(len(df)),test_size=.2,stratify=df.y,random_state=seed)
    return np.sort(tr),np.sort(va)
def preprocessing(cols):
    cats=[c for c in cols if c!='speed_limit']
    return ColumnTransformer([('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float64),cats),('numeric',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),['speed_limit'])],remainder='drop')
def score(y,p,t):
    y,p=np.asarray(y),np.asarray(p)
    if len(y)!=len(p) or not np.isfinite(p).all() or (p<0).any() or (p>1).any():raise ValueError('Invalid probabilities')
    tn,fp,fn,tp=confusion_matrix(y,p>=t,labels=[0,1]).ravel().tolist();pr,re,_=precision_recall_curve(y,p)
    return dict(n=len(y),tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else np.nan,recall=tp/(tp+fn) if tp+fn else np.nan,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan,fpr=fp/(fp+tn) if fp+tn else np.nan,average_precision=average_precision_score(y,p),pr_auc=auc(re,pr),roc_auc=roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan,brier=brier_score_loss(y,p),accuracy=(tp+tn)/len(y))
def threshold(y,p):
    y,p=np.asarray(y),np.asarray(p);order=np.argsort(-p,kind='stable');sy,sp=y[order],p[order]
    end=np.r_[np.flatnonzero(sp[:-1]!=sp[1:]),len(sp)-1];tp=np.r_[0,np.cumsum(sy)[end]];fp=np.r_[0,end+1-np.cumsum(sy)[end]];ts=np.r_[np.nextafter(sp[0],np.inf),sp[end]]
    f1=2*tp/(int(y.sum())+tp+fp);best=np.flatnonzero(f1==f1.max());i=min(best,key=lambda k:(fp[k],-ts[k]));return float(ts[i])
