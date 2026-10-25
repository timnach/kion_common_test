"""Causal implicit ALS / top-vocabulary EASE with bounded batch scoring.

All interactions are binary. ALS uses confidence alpha for observed entries;
unobserved entries have implicit confidence 1. EASE vocabulary selection uses
past user counts only. No future-only metadata item establishes availability.
"""
from time import perf_counter
import numpy as np
import pandas as pd
from scipy import linalg, sparse
from .collaborative import CollaborativeRetriever


class BinarySnapshot:
    def __init__(self, events, items, cutoff):
        self.cutoff=pd.Timestamp(cutoff)
        if pd.isna(self.cutoff): raise ValueError('Invalid cutoff')
        if self.cutoff.tzinfo is not None:
            self.cutoff=self.cutoff.tz_convert('UTC').tz_localize(None)
        dates=pd.to_datetime(events.timestamp,utc=True).dt.tz_convert(None)
        if dates.isna().any(): raise ValueError('Missing event timestamp')
        history=events.loc[(dates<self.cutoff)&events.item_id.isin(items.item_id),['user_id','item_id']].drop_duplicates()
        self.user_ids=np.sort(history.user_id.unique()).astype(np.int64)
        self.item_ids=np.sort(history.item_id.unique()).astype(np.int64)
        self.user_index=pd.Index(self.user_ids);self.item_index=pd.Index(self.item_ids)
        self.matrix=sparse.csr_matrix((np.ones(len(history),dtype=np.float32),
            (self.user_index.get_indexer(history.user_id),self.item_index.get_indexer(history.item_id))),
            shape=(len(self.user_ids),len(self.item_ids)))
        self.matrix.sort_indices()
        self.counts=np.asarray(self.matrix.sum(axis=0)).ravel()


class MatrixRetriever:
    def __init__(self,snapshot,method,*,factors=32,regularization=.1,alpha=20.,
                 iterations=15,seed=123,threads=6,max_items=4000):
        if method not in {'als','ease'}: raise ValueError('Unknown method')
        if not np.isfinite(regularization) or regularization<=0: raise ValueError('Positive regularization required')
        if isinstance(max_items,bool) or not isinstance(max_items,int) or max_items<1: raise ValueError('Positive max_items required')
        start=perf_counter();self.snapshot=snapshot;self.method=method
        if method=='ease':
            order=np.lexsort((snapshot.item_ids,-snapshot.counts))[:max_items]
            positions=np.sort(order)
        else: positions=np.arange(len(snapshot.item_ids))
        self.positions=positions;self.item_ids=snapshot.item_ids[positions]
        self.matrix=snapshot.matrix[:,positions].tocsr()
        self.interaction_coverage=self.matrix.nnz/max(1,snapshot.matrix.nnz)
        self.model=None;self.weights=None
        if len(positions) and method=='ease':
            gram=(self.matrix.T@self.matrix).toarray().astype(np.float64)
            gram.flat[::len(gram)+1]+=regularization
            precision=linalg.solve(gram,np.eye(len(gram)),assume_a='pos',check_finite=False)
            self.weights=np.ascontiguousarray(-precision/np.diag(precision)[None,:],dtype=np.float32)
            np.fill_diagonal(self.weights,0.)
        elif len(positions):
            from implicit.cpu.als import AlternatingLeastSquares
            self.model=AlternatingLeastSquares(factors=factors,regularization=regularization,
                alpha=alpha,iterations=iterations,random_state=seed,num_threads=threads)
            self.model.fit(self.matrix,show_progress=False)
        self.fit_seconds=perf_counter()-start

    def candidates_for(self,user_ids,k=100,batch_size=128):
        if isinstance(k,bool) or not isinstance(k,int) or k<0: raise ValueError('Nonnegative k required')
        if batch_size<1: raise ValueError('Positive batch_size required')
        requested=np.asarray(sorted(set(map(int,user_ids))),dtype=np.int64)
        positions=self.snapshot.user_index.get_indexer(requested)
        valid=positions>=0;requested=requested[valid];positions=positions[valid]
        result=[]
        if not len(self.item_ids) or k==0:
            return pd.DataFrame({'user_id':pd.Series(dtype='int64'),'item_id':pd.Series(dtype='int64'),'score':pd.Series(dtype='float64')})
        for start in range(0,len(requested),batch_size):
            upos=positions[start:start+batch_size];uids=requested[start:start+batch_size]
            histories=self.matrix[upos]
            if self.method=='als':
                scores=self.model.user_factors[upos]@self.model.item_factors.T
            else:
                scores=histories@self.weights
            scores=np.asarray(scores)
            for row,uid in enumerate(uids):
                seen=histories.indices[histories.indptr[row]:histories.indptr[row+1]]
                scores[row,seen]=-np.inf
                if self.method=='ease' and histories.indptr[row]==histories.indptr[row+1]:
                    continue  # No history in this vocabulary: use shared fallback.
                eligible=np.flatnonzero(np.isfinite(scores[row]))
                best=CollaborativeRetriever._top_positions(self.item_ids[eligible],scores[row,eligible],k)
                result.extend((int(uid),int(self.item_ids[eligible[p]]),float(scores[row,eligible[p]])) for p in best)
        return pd.DataFrame(result,columns=['user_id','item_id','score']).astype({'user_id':'int64','item_id':'int64','score':'float64'})
