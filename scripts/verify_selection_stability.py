"""Independent saved-prediction/metric/OOF/cohort checks; needs no raw dataset."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRanker
from kion_recsys.metrics import evaluate_ranking,paired_bootstrap
from kion_recsys.selection_stability import validate_protocol,baseline_bins


def verify(root,allow_partial=False):
    read=lambda p:json.loads(Path(p).read_text())
    config=read(root/'config.json');validate_protocol(config)
    if not allow_partial:assert read(root/'complete.json')['no_test_evaluation']
    assert read(root/'protocol_before_results.json')['config']==config
    checks=0;records={}
    for fold in config['validation_cutoffs']:
        folder=root/fold
        if allow_partial and not (folder/'candidates.npz').exists():continue
        truth={int(u):set(v) for u,v in read(folder/'truth.json').items()}
        with np.load(folder/'candidates.npz',allow_pickle=False) as x:
            full=pd.DataFrame({'user_id':x['user_id'],'item_id':x['item_id'],'label':x['label']})
        assert not full.duplicated(['user_id','item_id']).any()
        assert full.groupby('user_id').size().eq(config['candidate_k']).all()
        candidate_pairs=set(zip(full.user_id,full.item_id))
        _,candidate_per=evaluate_ranking(full,np.zeros(len(full)),truth,k=config['k'])
        candidate_recall=candidate_per.set_index('user_id').candidate_recall
        for count,seed in itertools.product(config['train_users'],config['seeds']):
            directory=folder/f'n{count}_seed{seed}'
            if allow_partial and not (directory/'complete.json').exists():continue
            selection=read(directory/'selection.json');complete=read(directory/'complete.json')
            cohorts=read(directory/'cohorts.json');records[(fold,count,seed)]=cohorts
            assert all(len(c['users'])==count for c in cohorts)
            for index,audit in enumerate(selection['oof_audit'],start=1):
                assert pd.Timestamp(audit['train_label_end'])<=pd.Timestamp(audit['target_cutoff'])
                assert all(pd.Timestamp(w)+pd.Timedelta(days=7)<=pd.Timestamp(audit['target_cutoff']) for w in audit['source_weeks'])
                model=CatBoostRanker();model.load_model(str(directory/f'oof_{index}.cbm'))
                assert model.tree_count_==config['ranker']['iterations']
                assert model.get_all_params()['random_seed']==seed
                assert model.get_all_params()['depth']==config['ranker']['depth']
                del model
                with np.load(directory/f'oof_{index}.npz',allow_pickle=False) as x:
                    assert str(x['train_label_end'])==audit['train_label_end']
                    assert str(x['cutoff'])==audit['target_cutoff']
                    assert set(map(int,x['user_id']))==set(cohorts[index]['users'])
                    assert np.isfinite(x['baseline_score']).all()
                    np.testing.assert_array_equal(baseline_bins(x['baseline_score'],x['user_id'],config['baseline_bins']),x['baseline_bin'])
                    assert len(x['user_id'])==audit['prediction_rows']
                checks+=1
            pers={}
            for method,features in selection['features'].items():
                assert len(features)<=config['selected_features'] and len(set(features))==len(features)
                with np.load(directory/(method+'_top10.npz'),allow_pickle=False) as x:
                    top=pd.DataFrame({'user_id':x['user_id'],'item_id':x['item_id'],'label':0})
                    scores=x['score'].copy()
                assert set(zip(top.user_id,top.item_id))<=candidate_pairs
                assert set(top.user_id)==set(truth)
                assert top.groupby('user_id').size().eq(config['k']).all()
                met,per=evaluate_ranking(top,scores,truth,k=config['k'])
                saved=pd.read_csv(directory/(method+'_per_user.csv'))
                aligned=per.set_index('user_id').loc[saved.user_id]
                for m in ['ndcg','recall','map','hitrate']:
                    np.testing.assert_allclose(saved[m],aligned[m],atol=1e-12)
                    np.testing.assert_allclose(saved[m].mean(),complete['metrics'][method][f"{m}@{config['k']}"],atol=1e-12)
                np.testing.assert_allclose(saved.candidate_recall,candidate_recall.loc[saved.user_id],atol=1e-12)
                pers[method]=saved
                columns=read(directory/(method+'_fit.json'))['columns']
                assert not any(c.startswith('oof_') or c in {'label','query_id','user_id'} for c in columns)
                model=CatBoostRanker();model.load_model(str(directory/(method+'.cbm')))
                assert model.feature_names_==columns
                assert model.tree_count_==config['ranker']['iterations']
                assert model.get_all_params()['random_seed']==seed
                assert model.get_all_params()['depth']==config['ranker']['depth']
                del model
                checks+=1
            for method in ['mi','cmim','cmim_oof']:
                actual=paired_bootstrap(pers['manual'],pers[method],metric='ndcg',n_boot=config['bootstrap'],seed=config['pool_seed'])
                for key in ['delta','ci_low','ci_high']:np.testing.assert_allclose(actual[key],complete['bootstrap'][method][key],atol=1e-12)
                checks+=1
            actual=paired_bootstrap(pers['cmim'],pers['cmim_oof'],metric='ndcg',n_boot=config['bootstrap'],seed=config['pool_seed'])
            for key in ['delta','ci_low','ci_high']:np.testing.assert_allclose(actual[key],complete['bootstrap']['cmim_oof_minus_cmim'][key],atol=1e-12)
            checks+=1
        sizes=sorted(config['train_users'])
        for seed in config['seeds']:
            for small,big in zip(sizes[:-1],sizes[1:]):
                if allow_partial and ((fold,small,seed) not in records or (fold,big,seed) not in records):continue
                assert all(set(a['users'])<=set(b['users']) for a,b in zip(records[(fold,small,seed)],records[(fold,big,seed)]))
                checks+=1
    print(f'Passed {checks} independent stability artifact checks'+(' (partial run)' if allow_partial else ''))
    return checks


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path,nargs='?',default=Path('runs/selection_stability'));p.add_argument('--partial',action='store_true');args=p.parse_args();verify(args.root,args.partial)
