import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.stats19.core import target,clean,features,A,B,split_2024,preprocessing,score,threshold
from scripts.plot import plot_rows
class StudyTests(unittest.TestCase):
    def frame(self,n=20):
        d=pd.DataFrame({c:[1]*n for c in B if c!='hour'})
        d['collision_year']=2024;d['collision_index']=[f'x{i:03}' for i in range(n)]
        d['collision_severity']=[1,2,3,3]*int(n/4);d['time']='13:05';d['speed_limit']=30;d['junction_detail']=13
        return d
    def test_mapping(self):
        self.assertEqual(target(pd.Series([1,2,3])).tolist(),[1,1,0])
        with self.assertRaises(ValueError):target(pd.Series([4]))
    def test_year(self):
        with self.assertRaises(ValueError):clean(self.frame(),2025)
        d,_,_=clean(self.frame(),2024);d['collision_year']=2025
        with self.assertRaises(ValueError):split_2024(d)
    def test_allowlist(self):
        d,_,_=clean(self.frame(),2024);d['number_of_casualties']=999;d['collision_adjusted_severity_serious']=1
        self.assertEqual(list(features(d,'A')),A);self.assertEqual(list(features(d,'B')),B)
        self.assertFalse(set(B)&{'y','collision_severity','collision_index','collision_year','number_of_casualties'})
    def test_missing_categories(self):
        d=self.frame();d.loc[0,'road_type']=-1;d.loc[1,'road_type']=9;d.loc[2,'road_type']=777;d.loc[3,'time']='25:01'
        c,q,_=clean(d,2024)
        self.assertEqual(c.road_type.iloc[:3].tolist(),['__MISSING__','__UNKNOWN__','__MISSING__'])
        self.assertEqual(c.hour.iloc[3],'__MISSING__')
    def test_duplicate(self):
        d=self.frame();d.loc[1,'collision_index']=d.loc[0,'collision_index']
        with self.assertRaises(ValueError):clean(d,2024)
    def test_train_only_transform(self):
        d,_,_=clean(self.frame(),2024);d.loc[0,'speed_limit']=np.nan;p=preprocessing(B);p.fit(features(d,'B'))
        median=p.named_transformers_['numeric']['impute'].statistics_[0];self.assertEqual(median,30)
        test=features(d,'B');test['road_type']='unseen';test['speed_limit']=70
        self.assertTrue(np.isfinite(p.transform(test)).all());self.assertEqual(p.named_transformers_['numeric']['impute'].statistics_[0],30)
    def test_metrics(self):
        m=score([1,1,0,0],[.9,.3,.8,.1],.5)
        for k in ['tp','fp','fn','tn']:self.assertEqual(m[k],1)
        self.assertEqual(m['precision'],.5);self.assertEqual(m['recall'],.5);self.assertEqual(m['f1'],.5)
        self.assertAlmostEqual(m['brier'],(.01+.49+.64+.01)/4)
    def test_ap_not_trapezoid(self):
        m=score([1,0,1,0],[.4]*4,.5)
        self.assertEqual(m['average_precision'],.5);self.assertEqual(m['pr_auc'],.75);self.assertTrue(np.isnan(m['precision']))
    def test_threshold_ties(self):
        y=np.array([1,0,1,0]);p=np.array([.8,.8,.2,.1]);t=threshold(y,p)
        self.assertEqual(t,.2);self.assertEqual(score(y,p,t)['tp'],2)
    def test_split_reproducible(self):
        d,_,_=clean(self.frame(100),2024);a,b=split_2024(d);c,e=split_2024(d)
        np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,e);self.assertFalse(set(a)&set(b));self.assertEqual(len(a),80)
    def test_distribution_categories(self):
        p=ROOT/'results/data/distributions.csv'
        if not p.exists():self.skipTest('prepared data not available')
        d=pd.read_csv(p,dtype={'value':str})
        speed=d[d.feature=='speed_limit']
        self.assertEqual(set(speed.value)-{'__MISSING__'},{'20','30','40','50','60','70'})
        self.assertEqual(speed.groupby('year')['count'].sum().to_dict(),{2024:100927,2025:101525})
    def test_figure_csv_contract(self):
        p=ROOT/'results/final-20261021/metrics.csv'
        if not p.exists():self.skipTest('final run not yet available')
        rows=plot_rows(p,'test');self.assertEqual(len(rows),5)
        original=pd.read_csv(p);original=original[(original.split=='test')&original.policy.isin(['validation_f1','majority'])]
        pd.testing.assert_frame_equal(rows,original)
if __name__=='__main__':unittest.main()
