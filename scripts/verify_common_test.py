"""Recompute common-test metrics from top-10 and complete truth/candidates."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRanker
from kion_recsys.metrics import paired_bootstrap
from kion_recsys.verification import independent_metrics


def verify(root):
    root=Path(root);read=lambda p:json.loads(Path(p).read_text());c=read(root/'config.json');checks=0
    assert read(root/'complete.json')['status']=='complete'
    assert max(c['train_users'])<=2000 and c['test_status']=='previously_examined_exploratory'
    assert pd.Timestamp(c['validation_cutoff'])+pd.Timedelta(days=7)<=pd.Timestamp(c['test_cutoff'])
    frozen=read(root/'ALL_DECISIONS_FROZEN.json');start=read(root/'TEST_STARTED.json')
    assert datetime.fromisoformat(frozen['frozen_at_utc'])<=datetime.fromisoformat(start['started_at_utc'])
    assert hashlib.sha256((root/'ALL_DECISIONS_FROZEN.json').read_bytes()).hexdigest()==start['frozen_decisions_sha256'];checks+=1
    truth={int(u):set(v) for u,v in read(root/'test'/'truth.json').items()}
    with np.load(root/'test'/'candidates.npz',allow_pickle=False) as z:
        candidates=pd.DataFrame({'user_id':z['user_id'],'item_id':z['item_id'],'label':z['label']})
    assert not candidates.duplicated(['user_id','item_id']).any()
    assert candidates.groupby('user_id').size().eq(c['candidate_k']).all()
    assert set(candidates.user_id)==set(truth) and len(truth)==c['validation_users']
    assert candidates.label.tolist()==[int(i in truth[u]) for u,i in zip(candidates.user_id,candidates.item_id)]
    pairs=set(zip(candidates.user_id,candidates.item_id));checks+=1
    candidate_recall={int(u):len(set(g.item_id)&truth[int(u)])/len(truth[int(u)]) for u,g in candidates.groupby('user_id')}
    independent=independent_metrics
    val_truth={int(u):set(v) for u,v in read(root/'validation'/'truth.json').items()}
    with np.load(root/'validation'/'candidates.npz',allow_pickle=False) as z:
        val_pairs=set(zip(z['user_id'],z['item_id']))
        assert set(z['user_id'])==set(val_truth)
    expected_arms={'manual','mi','cmim','cmim_oof','all_relational','random_search','guided_search','uniform','constant_decay','tuned_alpha','tuned_beta','activity_adaptive','multiscale_decay','guided_multiscale','cmim_decay'}
    all_rows=[]
    for seed in c['seeds']:
        vd=root/'validation'/f'seed{seed}';td=root/'test'/f'seed{seed}';decision=read(vd/'decisions.json')
        assert decision==frozen['decisions'][str(seed)] and not decision['test_used_for_selection']
        assert set(decision['arms'])==expected_arms
        assert len(decision['random_sets'])==len({tuple(s) for s in decision['random_sets']})==c['search_trials']
        assert len(decision['guided_sets'])==len({tuple(s) for s in decision['guided_sets']})==c['search_trials']
        selection=read(vd/'selection.json');cohorts=read(vd/'cohorts.json')
        for entry in cohorts:
            assert len(entry['users'])==c['train_users'][0]
            assert pd.Timestamp(entry['cutoff'])+pd.Timedelta(days=7)<=pd.Timestamp(c['validation_cutoff'])
        for audit in selection['oof_audit']:
            assert pd.Timestamp(audit['train_label_end'])<=pd.Timestamp(audit['target_cutoff'])
        tm=read(vd/'trial_metrics.json');hs=[f'h_{h}' for h in c['half_lives']]
        for trial,expected in tm.items():
            with np.load(vd/(trial+'_top10.npz'),allow_pickle=False) as z:
                assert set(zip(z['user_id'],z['item_id']))<=val_pairs
                assert set(z['user_id'])==set(val_truth)
                actual,_,_=independent(z['user_id'],z['item_id'],z['score'],val_truth,c['k'])
            for metric in ['ndcg@10','recall@10','map@10','hitrate@10']:
                np.testing.assert_allclose(actual[metric],expected[metric],atol=1e-12)
            checks+=1
        assert decision['best_h']==max(hs,key=lambda n:tm[n]['ndcg@10'])
        assert decision['arms']['random_search']['trial']==max([f'random_{i}' for i in range(c['search_trials'])],key=lambda n:tm[n]['ndcg@10'])
        assert decision['arms']['guided_search']['trial']==max(decision['guided_names'],key=lambda n:tm[n]['ndcg@10'])
        for family,factor,values in [('tuned_alpha','alpha',c['alpha_values']),('tuned_beta','beta',c['beta_values']),('activity_adaptive','adaptive',c['adaptive_powers'])]:
            names=[decision['best_h']]+[decision['best_h']+'_'+factor+'_'+str(v) for v in values]
            assert decision['arms'][family]['trial']==max(names,key=lambda n:tm[n]['ndcg@10'])
        winner=max(decision['arms'],key=lambda n:decision['arms'][n]['validation_metrics']['ndcg@10'])
        assert winner==decision['validation_winner'];checks+=1
        record=read(td/'complete.json');pers={}
        for name,arm in decision['arms'].items():
            model_path=vd/(arm['trial']+'.cbm')
            assert hashlib.sha256(model_path.read_bytes()).hexdigest()==arm['model_sha256']
            model=CatBoostRanker();model.load_model(str(model_path))
            assert model.feature_names_==arm['columns'] and model.tree_count_==c['ranker']['iterations']
            assert model.get_all_params()['random_seed']==seed
            assert model.get_all_params()['depth']==c['ranker']['depth']
            np.testing.assert_allclose(model.get_all_params()['learning_rate'],c['ranker']['learning_rate'],rtol=1e-6)
            assert not any(f.startswith('oof_') for f in arm['columns'])
            with np.load(td/(name+'_top10.npz'),allow_pickle=False) as z:
                uids=z['user_id'];items=z['item_id'];scores=z['score'];actual,per,_=independent(uids,items,scores,truth,c['k'])
            assert set(zip(uids,items))<=pairs and set(uids)==set(truth)
            assert pd.Series(uids).value_counts().eq(c['k']).all()
            saved=pd.read_csv(td/(name+'_per_user.csv'));per=per.loc[saved.user_id]
            for metric in ['ndcg','recall','map','hitrate']:
                np.testing.assert_allclose(saved[metric],per[metric],atol=1e-12)
                np.testing.assert_allclose(record['metrics'][name][metric+'@10'],actual[metric+'@10'],atol=1e-12)
            np.testing.assert_allclose(saved.candidate_recall,[candidate_recall[int(u)] for u in saved.user_id],atol=1e-12)
            np.testing.assert_allclose(record['metrics'][name]['candidate_recall'],np.mean(list(candidate_recall.values())),atol=1e-12)
            assert record['metrics'][name]['mean_candidates']==c['candidate_k']
            pers[name]=saved;all_rows.append((seed,name,actual['ndcg@10']));checks+=1
        for comparison,expected in record['bootstrap'].items():
            if comparison=='guided_minus_random':a,b='random_search','guided_search'
            elif comparison=='oof_minus_cmim':a,b='cmim','cmim_oof'
            elif comparison=='constant_minus_uniform':a,b='uniform','constant_decay'
            else:a,b='manual',comparison
            actual=paired_bootstrap(pers[a],pers[b],metric='ndcg',n_boot=c['bootstrap'],seed=c['pool_seed'])
            for k in ['delta','ci_low','ci_high']:np.testing.assert_allclose(actual[k],expected[k],atol=1e-12)
            checks+=1
    table=pd.read_csv(root/'metrics.csv');assert len(table)==len(all_rows)
    for seed,name,ndcg in all_rows:np.testing.assert_allclose(table.loc[(table.seed==seed)&(table.method==name),'ndcg@10'],ndcg,atol=1e-12)
    print(f'Passed {checks} common-test artifact checks');return checks


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path,nargs='?',default=Path('runs/common_test'));verify(p.parse_args().root)
