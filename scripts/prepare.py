"""Verify dictionary, both annual tables and temporal split."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import hashlib,json
import numpy as np
import pandas as pd
import openpyxl
from src.stats19.core import clean,split_2024,VALID,B
def main():
    out=ROOT/'results/data';out.mkdir(parents=True,exist_ok=True);local=ROOT/'data/processed';local.mkdir(exist_ok=True)
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    for f in manifest['files']:assert hashlib.sha256((ROOT/f['path']).read_bytes()).hexdigest()==f['sha256']
    w=openpyxl.load_workbook(ROOT/'data/raw/data_guide_2025.xlsx',read_only=True,data_only=True)
    dictionary=pd.DataFrame(list(w['2024_code_list'].values)[1:],columns=['table','field','code','label','note'])
    chosen=dictionary[(dictionary.table=='collision')&dictionary.field.isin(B+['collision_severity','time','date','collision_index','collision_year'])]
    chosen.to_csv(out/'dictionary_extract.csv',index=False)
    for col,codes in VALID.items():
        declared=set(pd.to_numeric(chosen.loc[chosen.field==col,'code'],errors='coerce').dropna().astype(int))
        assert codes<=declared,(col,codes-declared)
    severity=chosen[chosen.field=='collision_severity'].set_index('code').label.to_dict();assert severity=={1:'Fatal',2:'Serious',3:'Slight'}
    frames={};summaries=[];quality=[];schema=[];distributions=[];ids=[]
    for year in [2024,2025]:
        raw=pd.read_csv(ROOT/f'data/raw/collisions_{year}.csv',low_memory=False)
        for col in raw:schema.append(dict(year=year,field=col,dtype=str(raw[col].dtype),nulls=int(raw[col].isna().sum())))
        d,q,s=clean(raw,year)
        dates=pd.to_datetime(d.date,format='%d/%m/%Y',errors='coerce')
        assert dates.notna().all() and dates.dt.year.eq(year).all(),'Invalid date/year'
        expected=(dates.dt.dayofweek+1)%7+1
        assert expected.astype(str).eq(d.day_of_week).all(),'Date/day mismatch'
        frames[year]=d;summaries.append(s);quality+=q;ids.append(set(d.collision_index))
        d[['collision_index','collision_year','collision_severity','y']+B].to_csv(local/f'clean_{year}.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        for col in ['collision_severity','speed_limit','light_conditions','weather_conditions']:
            for value,n in d[col].value_counts(dropna=False).items():distributions.append(dict(year=year,feature=col,value=('__MISSING__' if pd.isna(value) else format(value,'g') if col=='speed_limit' else str(value)),count=int(n),fraction=float(n/len(d))))
    assert not ids[0]&ids[1],'Cross-year ID overlap'
    tr,va=split_2024(frames[2024]);np.savez_compressed(local/'split.npz',train=tr,validation=va)
    assignment=np.full(len(frames[2024]),'train',dtype='U10');assignment[va]='validation'
    pd.DataFrame({'collision_index':frames[2024].collision_index,'split':assignment}).to_csv(local/'split.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(summaries).to_csv(out/'sample_counts.csv',index=False);pd.DataFrame(quality).to_csv(out/'missingness.csv',index=False);pd.DataFrame(schema).to_csv(out/'schema.csv',index=False);pd.DataFrame(distributions).to_csv(out/'distributions.csv',index=False)
    check={'columns_equal':list(frames[2024].columns)==list(frames[2025].columns),'cross_year_id_overlap':0,'date_day_year_checks':'PASS','target_mapping':{'1':1,'2':1,'3':0},'feature_sets':{'A':B[:4],'B':B},'source_specification':'2024_code_list applied to both published years','dictionary_warning':'Conversion sheet mislabels road-surface-like values as weather; no legacy conversion applied','split_sha256':hashlib.sha256((local/'split.csv.gz').read_bytes()).hexdigest(),'train_n':len(tr),'validation_n':len(va)}
    (out/'quality.json').write_text(json.dumps(check,indent=2)+'\n');print(pd.DataFrame(summaries).to_string(index=False));print(pd.DataFrame(quality).to_string(index=False));print(json.dumps(check,indent=2))
if __name__=='__main__':main()
