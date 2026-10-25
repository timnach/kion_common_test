"""Recompute recall from actual candidate lists and full original truth."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from kion_recsys.retrieval_benchmark import evaluate
from kion_recsys.metrics import paired_bootstrap


def main(root):
    read=lambda p:json.loads(Path(p).read_text())
    if read(root/'complete.json')['status']!='complete':raise ValueError('Incomplete run')
    config=read(root/'config.json');decisions=read(root/'decisions_before_evaluation.json')['folds'];checks=0
    for date,decision in decisions.items():
        truth={int(u):set(v) for u,v in sorted(read(Path(decision['original_run'])/'test_truth.json').items(),key=lambda pair:int(pair[0]))}
        record=read(root/date/'results.json');per_users={}
        for name,expected in record['metrics'].items():
            f=pd.read_csv(root/date/('candidates_'+name+'.csv.gz'))
            assert not f.duplicated(['user_id','item_id']).any()
            assert set(f.user_id)==set(truth)
            assert f.groupby('user_id').size().eq(config['candidate_k']).all()
            assert f.groupby('user_id')['rank'].apply(lambda r:set(r)==set(range(1,config['candidate_k']+1))).all()
            actual,per=evaluate(f,truth)
            for metric in ['candidate_recall','micro_recall','fallback_fraction','mean_candidates']:
                np.testing.assert_allclose(actual[metric],expected[metric],atol=1e-12)
            saved=pd.read_csv(root/date/('per_user_'+name+'.csv'))
            np.testing.assert_allclose(saved.sort_values('user_id').candidate_recall,per.sort_values('user_id').candidate_recall,atol=1e-12)
            # Seeded resampling depends on row order. Retain the experiment's
            # saved order, while independently recomputing every user's recall.
            per_users[name]=per.set_index('user_id').loc[saved.user_id].reset_index();checks+=1
        for name,expected in record['bootstrap_vs_legacy_knn'].items():
            actual=paired_bootstrap(per_users['legacy+knn'],per_users[name],metric='candidate_recall',n_boot=config['bootstrap'],seed=config['seed'])
            for key in ['delta','ci_low','ci_high']:np.testing.assert_allclose(actual[key],expected[key],atol=1e-12)
            checks+=1
        val=decision['validation_metrics'];unions=[name for name in val if '+' in name]
        assert decision['chosen_union']==max(unions,key=lambda n:val[n]['candidate_recall'])
        for name in ['als','ease']:
            tuning=[r for r in decision['tuning'] if r['source'].startswith(name+'_')]
            assert decision[name+'_tag']==max(tuning,key=lambda r:r['candidate_recall'])['source']
            checks+=1
    print(f'Passed {checks} independent artifact checks')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path,nargs='?',default=Path('runs/retrieval_benchmark'));main(p.parse_args().root)
