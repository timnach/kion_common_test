import json
from pathlib import Path
import runpy
import numpy as np
import pandas as pd
import pytest
from kion_recsys.features import DecayConfig,decay_affinity
from kion_recsys.decay_ablation import temporal_features,effective_half_life,final_specs,validate,execute


def test_kernel_matches_existing_formula_without_numerical_cutoff():
    ages=np.array([1.,3.,7.,20.]);s=np.array([[.2,.9,.4,0],[1,.2,0,.3]])
    r=np.array([.2,.4,.1,.7]);spec={'kind':'decay','half_life':7,'alpha':1,'beta':2,'adaptive_power':.5}
    np.testing.assert_allclose(temporal_features(ages,s,r,spec,3),decay_affinity(ages,s,r,DecayConfig(7,alpha=1,beta=2,adaptive_power=.5),3),rtol=1e-12)


def test_uniform_and_numerical_scale_invariance():
    s=np.array([[1.,0.],[.2,.6]]);ages=np.array([1.,2.]);r=np.zeros(2)
    uniform=temporal_features(ages,s,r,{'kind':'uniform'})
    np.testing.assert_allclose(uniform[:,0],s.mean(axis=1));np.testing.assert_allclose(uniform[:,1],np.log1p(s.sum(axis=1)))
    np.testing.assert_allclose(uniform[:,2],[2,2])
    spec={'kind':'decay','half_life':1}
    normal=temporal_features(ages,s,r,spec);old=temporal_features(ages+10000,s,r,spec)
    np.testing.assert_allclose(normal[:,[0,2]],old[:,[0,2]],rtol=1e-10)
    assert old[0,0]==pytest.approx(2/3);assert old[0,2]==pytest.approx(1.8);assert np.isfinite(old).all()
    with pytest.raises(ValueError,match='before cutoff'):temporal_features([0.,1.],s,r,spec)


def test_individual_weighting_factors_are_identifiable():
    ages=np.array([1.,3.,5.]);s=np.array([[.1,.4,.9],[.8,.2,.3]]);r=np.zeros(3)
    base={'kind':'decay','half_life':7}
    np.testing.assert_allclose(temporal_features(ages,s,r,base),temporal_features(ages,s,r,{**base,'alpha':2}))
    a=temporal_features(ages,s,np.ones(3),base);b=temporal_features(ages,s,np.ones(3),{**base,'alpha':1})
    np.testing.assert_allclose(a[:,[0,2]],b[:,[0,2]])
    np.testing.assert_allclose(np.expm1(b[:,1]),2*np.expm1(a[:,1]))
    c=temporal_features(ages,s,r,{**base,'beta':2})
    assert np.all(c[:,0]>=a[:,0])


def test_shared_adaptation_and_factor_isolation():
    assert effective_half_life(7,1,7,1)==7
    assert abs(np.log(effective_half_life(7,2,14,1)/7))<abs(np.log(effective_half_life(7,100,14,1)/7))
    cfg={'alpha_values':[.5,1,2],'beta_values':[.5,1,2],'adaptive_powers':[.5,1]}
    for spec in final_specs(cfg,7).values():
        if spec['kind']=='decay':assert sum(spec.get(k,0)>0 for k in ['alpha','beta','adaptive_power'])<=1
    base=json.loads((Path(__file__).resolve().parents[1]/'configs'/'decay_ablation.json').read_text())
    base['train_users']=2001
    with pytest.raises(ValueError,match='2000'):validate(base)
    base['train_users']=20;base['calibration_cutoff']='2021-07-12'
    with pytest.raises(ValueError,match='precede'):validate(base)


def test_full_controlled_study_and_saved_prediction_verifier(tmp_path):
    from kion_recsys.selection_stability import execute as build_source
    rng=np.random.default_rng(51);rows=[]
    for uid in range(100):
        for item in rng.choice(100,70,replace=False):
            rows.append((uid,int(item),pd.Timestamp('2021-04-01')+pd.Timedelta(days=int(rng.integers(130)))))
    events=pd.DataFrame(rows,columns=['user_id','item_id','timestamp']);data=tmp_path/'data';data.mkdir()
    events.rename(columns={'timestamp':'last_watch_dt'}).to_csv(data/'interactions.csv',index=False)
    pd.DataFrame({'item_id':range(100),'genres':['a, b' if i%2 else 'c' for i in range(100)],'countries':['x']*100,'content_type':['film']*100}).to_csv(data/'items.csv',index=False)
    pd.DataFrame({'user_id':range(100)}).to_csv(data/'users.csv',index=False)
    project=Path(__file__).resolve().parents[1]
    source=json.loads((project/'configs'/'selection_stability.json').read_text())
    source.update(data_dir=str(data),output_dir=str(tmp_path/'source'),validation_cutoffs=['2021-07-12','2021-07-19'],train_users=[20],
        train_pool_users=80,validation_users=30,seeds=[17],threads=1,candidate_k=20,max_negatives=8,bootstrap=20,selected_features=4)
    source['ranker']['iterations']=5;source['retrieval']['ease']={'regularization':10,'max_items':30}
    source['retrieval']['knn']={'neighbors':20,'shrinkage':2,'block_size':16}
    build_source(source)
    cfg=json.loads((project/'configs'/'decay_ablation.json').read_text())
    for key in ['validation_cutoffs','validation_users','seeds','threads','candidate_k','max_negatives','bootstrap','ranker']:cfg[key]=source[key]
    cfg.update(source_study=source['output_dir'],output_dir=str(tmp_path/'decay'),train_users=20,half_lives=[1,7],alpha_values=[1],beta_values=[1],adaptive_powers=[1])
    execute(cfg)
    out=tmp_path/'decay';assert json.loads((out/'complete.json').read_text())['no_test_evaluation']
    verifier=runpy.run_path(str(project/'scripts'/'verify_decay_ablation.py'))
    assert verifier['verify'](out)>0
    execute(cfg)  # Completed checkpoints must be reusable without reselection.
