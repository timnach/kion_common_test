import json
from pathlib import Path
import runpy
import numpy as np
import pandas as pd
import pytest
from kion_recsys.common_test import execute, validate


def test_validation_cannot_overlap_test_or_claim_untouched():
    c=json.loads((Path(__file__).resolve().parents[1]/'configs/common_test.json').read_text())
    c['test_cutoff']='2021-08-12'
    with pytest.raises(ValueError,match='overlap'):validate(c)
    c['test_cutoff']='2021-08-16';c['test_status']='untouched'
    with pytest.raises(ValueError,match='untouched'):validate(c)


def test_complete_common_test_and_resume(tmp_path):
    rng=np.random.default_rng(771);rows=[]
    for u in range(100):
        for item in rng.choice(100,70,replace=False):
            rows.append((u,int(item),pd.Timestamp('2021-04-01')+pd.Timedelta(days=int(rng.integers(130)))))
    data=tmp_path/'data';data.mkdir()
    pd.DataFrame(rows,columns=['user_id','item_id','last_watch_dt']).to_csv(data/'interactions.csv',index=False)
    pd.DataFrame({'item_id':range(100),'genres':['a,b' if i%2 else 'c' for i in range(100)],'countries':['x' if i%3 else 'z' for i in range(100)],'content_type':['film']*100}).to_csv(data/'items.csv',index=False)
    pd.DataFrame({'user_id':range(100)}).to_csv(data/'users.csv',index=False)
    project=Path(__file__).resolve().parents[1];c=json.loads((project/'configs/common_test.json').read_text())
    c.update(data_dir=str(data),output_dir=str(tmp_path/'result'),validation_cutoff='2021-07-19',test_cutoff='2021-07-26',train_snapshots=3,
        train_users=[20],train_pool_users=50,validation_users=30,seeds=[17],threads=1,candidate_k=20,max_negatives=8,bootstrap=20,
        selected_features=4,search_trials=3,half_lives=[1,7],alpha_values=[1],beta_values=[1],adaptive_powers=[1])
    c['ranker']['iterations']=5;c['retrieval']['ease']={'regularization':10,'max_items':30};c['retrieval']['knn']={'neighbors':20,'shrinkage':2,'block_size':16}
    execute(c)
    out=Path(c['output_dir']);before=(out/'ALL_DECISIONS_FROZEN.json').read_bytes()
    contrasts=pd.read_csv(out/'paired_contrasts.csv')
    assert set(contrasts.seed)==set(c['seeds'])
    assert set(contrasts.bootstrap_seed)=={c['pool_seed']}
    verify=runpy.run_path(str(project/'scripts/verify_common_test.py'))['verify'];assert verify(out)>20
    execute(c);assert (out/'ALL_DECISIONS_FROZEN.json').read_bytes()==before
