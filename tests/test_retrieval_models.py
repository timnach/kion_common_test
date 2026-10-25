import numpy as np
import pandas as pd
import pytest
from kion_recsys.retrieval_models import BinarySnapshot,MatrixRetriever
from kion_recsys.retrieval import SnapshotRetriever
from kion_recsys.retrieval_benchmark import complete_candidates,evaluate


def data():
    e=pd.DataFrame([(1,10),(2,10),(2,20),(3,20),(3,30),(4,30)],columns=['user_id','item_id'])
    e['timestamp']=pd.Timestamp('2021-01-01')
    return e,pd.DataFrame({'item_id':[10,20,30,40],'genres':['a']*4})


def test_ease_reference_and_batch_invariance():
    e,i=data();x=BinarySnapshot(e,i,'2021-01-02')
    m=MatrixRetriever(x,'ease',regularization=2)
    a=x.matrix.toarray().astype(float);p=np.linalg.inv(a.T@a+2*np.eye(3));b=-p/np.diag(p)[None,:];np.fill_diagonal(b,0)
    np.testing.assert_allclose(m.weights,b,rtol=1e-6,atol=1e-7)
    pd.testing.assert_frame_equal(m.candidates_for([1,2,4],k=2,batch_size=1),m.candidates_for([1,2,4],k=2,batch_size=8))
    assert m.candidates_for([1],k=2).item_id.tolist()==[20,30]


@pytest.mark.parametrize('method,params',[('ease',{'regularization':2}),('als',{'factors':3,'iterations':3})])
def test_future_invariance_and_seen_exclusion(method,params):
    e,i=data();future=pd.DataFrame({'user_id':[1,999],'item_id':[40,30],'timestamp':pd.to_datetime(['2021-01-02','2021-01-03'])})
    a=MatrixRetriever(BinarySnapshot(e,i,'2021-01-02'),method,threads=1,**params)
    b=MatrixRetriever(BinarySnapshot(pd.concat([e,future]),i,'2021-01-02'),method,threads=1,**params)
    pd.testing.assert_frame_equal(a.candidates_for([1,2,999]),b.candidates_for([1,2,999]))
    rows=b.candidates_for([1,2]);assert set(zip(rows.user_id,rows.item_id)).isdisjoint(set(zip(e.user_id,e.item_id)))
    assert 40 not in b.item_ids


def test_vocabulary_cap_and_cold_vocabulary_user():
    e,i=data();m=MatrixRetriever(BinarySnapshot(e,i,'2021-01-02'),'ease',max_items=1)
    assert m.item_ids.tolist()==[10]  # Equal past counts: smallest ID wins.
    assert m.interaction_coverage==pytest.approx(2/6)
    assert m.candidates_for([4]).empty


def test_common_fallback_exact_budget_and_no_seen_items():
    e,i=data();legacy=SnapshotRetriever(e,i,'2021-01-02')
    raw=pd.DataFrame({'user_id':[1],'item_id':[20],'score':[1.]})
    result=complete_candidates({'ease':raw},['ease'],legacy,[1,999],2)
    assert result.groupby('user_id').size().to_dict()=={1:2,999:2}
    assert result.loc[result.user_id==1].item_id.tolist()==[20,30]
    assert result.loc[result.user_id==1].fallback.tolist()==[0,1]
    assert not result.duplicated(['user_id','item_id']).any()
    assert 40 not in result.item_id.values


def test_recall_counts_missing_candidates_and_macro_vs_micro():
    f=pd.DataFrame({'user_id':[1,1],'item_id':[10,20]})
    result,per=evaluate(f,{1:{10},2:{30,40,50}})
    assert result['candidate_recall']==.5
    assert result['micro_recall']==.25
    assert result['no_hit_fraction']==.5
    assert len(per)==2


def test_complete_benchmark_with_archived_cohorts(tmp_path):
    import json
    from kion_recsys.retrieval_benchmark import execute
    rng=np.random.default_rng(32);events=[]
    for uid in range(40):
        for iid in rng.choice(100,60,replace=False):
            events.append((uid,int(iid),pd.Timestamp('2021-05-01')+pd.Timedelta(days=int(rng.integers(110)))))
    frame=pd.DataFrame(events,columns=['user_id','item_id','timestamp']);raw=tmp_path/'raw';raw.mkdir()
    frame.rename(columns={'timestamp':'last_watch_dt'}).to_csv(raw/'interactions.csv',index=False)
    pd.DataFrame({'item_id':range(100),'genres':['a' if i%2 else 'b' for i in range(100)]}).to_csv(raw/'items.csv',index=False)
    pd.DataFrame({'user_id':range(40)}).to_csv(raw/'users.csv',index=False)
    folds=[]
    for index,date in enumerate(['2021-07-12','2021-07-19']):
        folder=tmp_path/f'fold{index}';folder.mkdir();snapshots=[]
        for phase,cutoff in [('validation',pd.Timestamp(date)-pd.Timedelta(days=7)),('evaluation',pd.Timestamp(date))]:
            future=frame.loc[(frame.timestamp>=cutoff)&(frame.timestamp<cutoff+pd.Timedelta(days=7))]
            warm=set(frame.loc[frame.timestamp<cutoff].user_id)
            users=sorted(warm&set(future.user_id))[:20]
            snapshots.append({'phase':phase,'cutoff':str(cutoff.date()),'selected_users':users})
            if phase=='evaluation':
                truth={str(u):sorted(map(int,g.item_id)) for u,g in future.loc[future.user_id.isin(users)].groupby('user_id')}
                (folder/'test_truth.json').write_text(json.dumps(truth,sort_keys=True))
        (folder/'snapshots.json').write_text(json.dumps(snapshots));folds.append({'original_run':str(folder)})
    config={'data_dir':str(raw),'output_dir':str(tmp_path/'out'),'candidate_k':10,'seed':32,'threads':1,'bootstrap':20,
        'legacy':{'global_k':10,'recent_k':10,'genre_k':5,'max_candidates':20},
        'knn':{'neighbors':10,'shrinkage':2,'block_size':16},
        'als_grid':[{'factors':4,'regularization':.1,'iterations':2}],
        'ease_grid':[{'regularization':5,'max_items':30}],'folds':folds}
    execute(config)
    out=tmp_path/'out';assert json.loads((out/'complete.json').read_text())['status']=='complete'
    table=pd.read_csv(out/'metrics.csv');assert len(table)==16
    assert table.mean_candidates.eq(10).all()
    assert (out/'decisions_before_evaluation.json').exists()
    assert (out/'REPORT.md').exists()
    for p in out.glob('2021-*/candidates_*.csv.gz'):
        exported=pd.read_csv(p)
        assert len(exported)==200
        assert not exported.duplicated(['user_id','item_id']).any()

    # Archive JSON may order numeric user IDs lexicographically. The verifier
    # must retain the saved experiment order for the seeded bootstrap.
    from pathlib import Path
    import runpy
    namespace=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts'/'verify_retrieval_benchmark.py'))
    namespace['main'](out)
