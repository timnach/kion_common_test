"""Recompute decay study metrics and verify chronology, factor isolation and caps."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRanker
from kion_recsys.decay_ablation import validate,final_specs
from kion_recsys.metrics import evaluate_ranking,paired_bootstrap


def verify(root,allow_partial=False):
    read=lambda p:json.loads(Path(p).read_text());config=read(root/'config.json');validate(config)
    if not allow_partial:assert read(root/'complete.json')['no_test_evaluation']
    assert read(root/'protocol_before_results.json')['config']==config
    choice=read(root/'h_choice_before_validation.json');h=choice['half_life'];assert choice['no_diagnostic_validation_used']
    hgrid=pd.read_csv(root/'h_grid_metrics.csv');means=hgrid.groupby('half_life')['ndcg@10'].mean()
    assert float(means.idxmax())==h
    specs=final_specs(config,h);stored=read(root/'factor_specs_before_validation.json');assert stored['specs']==specs
    for name,spec in specs.items():
        if spec['kind']=='decay':
            assert spec['half_life']==h
            assert sum(float(spec.get(k,0))>0 for k in ['alpha','beta','adaptive_power'])<=1
    checks=0
    calibration_truth={int(u):set(v) for u,v in read(root/'calibration_truth.json').items()}
    with np.load(root/'calibration_candidates.npz',allow_pickle=False) as z:
        calibration_pairs=set(zip(z['user_id'],z['item_id']))
    for seed in config['seeds']:
        directory=root/'calibration'/f'seed{seed}';cohorts=read(directory/'cohorts.json')
        assert all(len(c['users'])==config['train_users'] and pd.Timestamp(c['cutoff'])+pd.Timedelta(days=7)<=pd.Timestamp(config['calibration_cutoff']) for c in cohorts)
        for x in config['half_lives']:
            model=CatBoostRanker();model.load_model(str(directory/('h_'+str(x)+'.cbm')))
            assert model.tree_count_==config['ranker']['iterations'];assert model.get_all_params()['random_seed']==seed
            assert len(model.feature_names_)==len(read(directory/('h_'+str(x)+'_fit.json'))['columns'])
            with np.load(directory/('h_'+str(x)+'_top10.npz'),allow_pickle=False) as z:
                pairs=pd.DataFrame({'user_id':z['user_id'],'item_id':z['item_id'],'label':0});scores=z['score'].copy()
            assert set(zip(pairs.user_id,pairs.item_id))<=calibration_pairs
            met,_=evaluate_ranking(pairs,scores,calibration_truth,k=config['k'])
            row=hgrid.loc[(hgrid.half_life==x)&(hgrid.seed==seed)].iloc[0]
            for key in ['ndcg@10','recall@10','map@10']:np.testing.assert_allclose(met[key],row[key],atol=1e-12)
            checks+=1
    for fold in config['validation_cutoffs']:
        folder=root/fold
        if allow_partial and not (folder/'candidates.npz').exists():continue
        truth={int(u):set(v) for u,v in read(folder/'truth.json').items()}
        with np.load(folder/'candidates.npz',allow_pickle=False) as x:
            full=pd.DataFrame({'user_id':x['user_id'],'item_id':x['item_id'],'label':x['label']})
        assert full.groupby('user_id').size().eq(config['candidate_k']).all()
        eligible=set(zip(full.user_id,full.item_id));_,cp=evaluate_ranking(full,np.zeros(len(full)),truth,k=config['k'])
        candidate_recall=cp.set_index('user_id').candidate_recall
        for seed in config['seeds']:
            directory=folder/f'seed{seed}'
            if allow_partial and not (directory/'complete.json').exists():continue
            record=read(directory/'complete.json');cohorts=read(directory/'cohorts.json')
            assert all(len(c['users'])==config['train_users'] and pd.Timestamp(c['cutoff'])+pd.Timedelta(days=7)<=pd.Timestamp(fold) for c in cohorts)
            source=Path(config['source_study'])/fold/f"n{config['train_users']}_seed{seed}"
            assert cohorts==read(source/'cohorts.json');pers={}
            for method in ['manual',*specs]:
                with np.load(directory/(method+'_top10.npz'),allow_pickle=False) as x:
                    top=pd.DataFrame({'user_id':x['user_id'],'item_id':x['item_id'],'label':0});scores=x['score'].copy()
                assert set(zip(top.user_id,top.item_id))<=eligible
                assert top.groupby('user_id').size().eq(config['k']).all()
                _,per=evaluate_ranking(top,scores,truth,k=config['k']);saved=pd.read_csv(directory/(method+'_per_user.csv'))
                ordered=per.set_index('user_id').loc[saved.user_id]
                for key in ['ndcg','map','recall','hitrate']:
                    np.testing.assert_allclose(saved[key],ordered[key],atol=1e-12)
                    np.testing.assert_allclose(saved[key].mean(),record['metrics'][method][key+'@10'],atol=1e-12)
                np.testing.assert_allclose(saved.candidate_recall,candidate_recall.loc[saved.user_id],atol=1e-12)
                pers[method]=saved
                meta=read(directory/(method+'_fit.json'));model=CatBoostRanker();model.load_model(str(directory/(method+'.cbm')))
                assert model.feature_names_==meta['columns'];assert model.tree_count_==config['ranker']['iterations']
                assert model.get_all_params()['random_seed']==seed;assert model.get_all_params()['depth']==config['ranker']['depth']
                if method=='manual':
                    assert hashlib.sha256((source/'manual.cbm').read_bytes()).hexdigest()==meta['source_sha256']
                    assert hashlib.sha256((directory/'manual.cbm').read_bytes()).hexdigest()==meta['source_sha256']
                else:assert len(meta['columns'])==len(read(directory/'manual_fit.json')['columns'])+3
                checks+=1
            for name,expected in record['bootstrap_vs_constant'].items():
                if name=='constant_minus_uniform':a,b='uniform','constant'
                elif name=='constant_minus_manual':a,b='manual','constant'
                else:a,b='constant',name
                actual=paired_bootstrap(pers[a],pers[b],metric='ndcg',n_boot=config['bootstrap'],seed=config['pool_seed'])
                for key in ['delta','ci_low','ci_high']:np.testing.assert_allclose(actual[key],expected[key],atol=1e-12)
                checks+=1
    if not allow_partial:
        metrics=pd.read_csv(root/'metrics.csv');assert len(metrics)==len(config['validation_cutoffs'])*len(config['seeds'])*(1+len(specs))
    print(f'Passed {checks} independent decay artifact checks'+(' (partial run)' if allow_partial else ''));return checks


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path,nargs='?',default=Path('runs/decay_ablation'));p.add_argument('--partial',action='store_true');a=p.parse_args();verify(a.root,a.partial)
