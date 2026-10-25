import json
from pathlib import Path
import runpy
import numpy as np
import pandas as pd
import pytest
from kion_recsys.selection import CMIMSelector
from kion_recsys.selection_stability import (validate_protocol,negative_sample,sampled_users,select_users,
    baseline_bins,check_oof_dates,execute)


def test_nested_cohorts_and_negatives():
    f=pd.DataFrame([(u,i,int(i==0)) for u in range(50) for i in range(30)],columns=['user_id','item_id','label'])
    a=sampled_users(f,10,17,'2021-01-01');b=sampled_users(f,25,17,'2021-01-01')
    assert set(a)<set(b)
    small=negative_sample(select_users(f,a),5,17,'2021-01-01')
    big=negative_sample(select_users(f,b),5,17,'2021-01-01')
    assert set(zip(small.user_id,small.item_id))<set(zip(big.user_id,big.item_id))
    assert len(small)==10*6
    assert not np.array_equal(b,sampled_users(f,25,42,'2021-01-01'))


def test_no_forbidden_test_overlap():
    config={'train_pool_users':10,'train_users':[5],'seeds':[1,2],'train_snapshots':4,
        'validation_cutoffs':['2021-08-01'],'forbidden_test_cutoffs':['2021-08-02']}
    with pytest.raises(ValueError,match='overlaps'):validate_protocol(config)
    config.update(train_pool_users=8000,train_users=[5000],max_train_users_allowed=2000,validation_cutoffs=['2021-07-12'])
    with pytest.raises(ValueError,match='limit'):validate_protocol(config)


def test_forward_horizon_and_complete_query_bins():
    assert check_oof_dates(['2021-01-01'],'2021-01-08')=='2021-01-08'
    with pytest.raises(ValueError,match='horizon'):check_oof_dates(['2021-01-01'],'2021-01-07')
    scores=np.arange(100);bins=baseline_bins(scores,np.ones(100),5)
    np.testing.assert_array_equal(np.bincount(bins),[20]*5)
    assert len(np.unique(baseline_bins(np.ones(100),np.ones(100),5)))==1


def test_explicit_extended_context_keeps_original_guardrails():
    rng=np.random.default_rng(3);n=2000
    x=pd.DataFrame({'signal':rng.integers(0,10,n)});y=(x.signal>4).astype(int)
    c=pd.DataFrame({'activity':rng.integers(0,3,n),'genre':rng.integers(0,2,n),
        'popularity':rng.integers(0,3,n),'baseline':rng.integers(0,5,n)})
    with pytest.raises(ValueError,match='at most 3'):CMIMSelector().fit(x,y,c)
    m=CMIMSelector(max_context_columns=4,max_context_states=128).fit(x,y,c)
    assert 32<m.n_context_states_<=90
    bad=c.rename(columns={'baseline':'user_id'})
    with pytest.raises(ValueError,match='IDs'):CMIMSelector(max_context_columns=4,max_context_states=128).fit(x,y,bad)


def test_end_to_end_oof_selection_study(tmp_path):
    rng=np.random.default_rng(41);rows=[]
    for uid in range(100):
        for item in rng.choice(100,70,replace=False):
            rows.append((uid,int(item),pd.Timestamp('2021-04-01')+pd.Timedelta(days=int(rng.integers(130)))))
    events=pd.DataFrame(rows,columns=['user_id','item_id','timestamp']);data=tmp_path/'data';data.mkdir()
    events.rename(columns={'timestamp':'last_watch_dt'}).to_csv(data/'interactions.csv',index=False)
    pd.DataFrame({'item_id':range(100),'genres':['a, b' if i%2 else 'c' for i in range(100)],
        'countries':['x' if i%3 else 'z' for i in range(100)],'content_type':['film']*100}).to_csv(data/'items.csv',index=False)
    pd.DataFrame({'user_id':range(100)}).to_csv(data/'users.csv',index=False)
    config=json.loads((Path(__file__).resolve().parents[1]/'configs'/'selection_stability.json').read_text())
    config.update(data_dir=str(data),output_dir=str(tmp_path/'output'),validation_cutoffs=['2021-07-12','2021-07-19'],
        train_users=[20,40],train_pool_users=80,validation_users=30,seeds=[17,42],threads=1,candidate_k=20,
        max_negatives=8,bootstrap=20,selected_features=4)
    config['ranker']['iterations']=5;config['retrieval']['ease']={'regularization':10,'max_items':30}
    config['retrieval']['knn']={'neighbors':20,'shrinkage':2,'block_size':16}
    execute(config)
    root=tmp_path/'output';assert json.loads((root/'complete.json').read_text())['no_test_evaluation']
    table=pd.read_csv(root/'metrics.csv');assert len(table)==2*2*2*4
    assert set(table.method)=={'manual','mi','cmim','cmim_oof'}
    verifier=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts'/'verify_selection_stability.py'))
    assert verifier['verify'](root)>0
    execute(config)  # Resume completed cases without retraining or reselection.
