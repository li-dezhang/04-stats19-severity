"""Generate report/README tables directly from final CSVs."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def markdown(df):
    def fmt(v):
        if pd.isna(v):return 'NA'
        if isinstance(v,float):return f'{v:.4f}'
        return str(v)
    return '\n'.join(['| '+' | '.join(df.columns)+' |','|'+'|'.join(['---']*len(df.columns))+'|']+['| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])
def tables():
    r=ROOT/'results/final-20261021';m=pd.read_csv(r/'metrics.csv');m=m[(m.split=='test')&m.policy.isin(['validation_f1','majority'])].copy()
    m['model']=m.model.map({'majority':'Majority','logistic_A':'Logistic A','logistic_B':'Logistic B','random_forest_B':'Random forest B','hist_gradient_B':'Gradient boosting B'})
    sample=pd.read_csv(ROOT/'results/data/sample_counts.csv')[['year','eligible_rows','fatal','serious','slight','severe_fraction']]
    inc=pd.read_csv(r/'increments.csv');inc=inc[inc.split=='test'].pivot(index='comparison',columns='metric',values='delta').reset_index()[['comparison','f1','average_precision','roc_auc','brier']]
    return {'SAMPLES':markdown(sample),'PERFORMANCE':markdown(m[['model','precision','recall','f1','average_precision','roc_auc','brier']]),'COUNTS':markdown(m[['model','threshold','tp','fp','fn','tn','fpr']]),'INCREMENTS':markdown(inc)}
def main():
    t=tables()
    (ROOT/'results/final-20261021/tables.md').write_text('\n\n'.join('## '+k+'\n\n'+v for k,v in t.items())+'\n')
    for path in [ROOT/'README.md',ROOT/'report/RESEARCH_REPORT.md']:
        s=path.read_text()
        for k,v in t.items():
            start=f'<!-- {k}:START -->';end=f'<!-- {k}:END -->'
            if start in s:s=s[:s.index(start)+len(start)]+'\n'+v+'\n'+s[s.index(end):]
        path.write_text(s)
if __name__=='__main__':main()
