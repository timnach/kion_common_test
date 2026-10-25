"""Train/validation decisions first, then all methods on one common test.

August is previously examined: this experiment is explicitly exploratory.
No existing study or result is overwritten.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from catboost import CatBoostRanker

from .data import load_data
from .features import SnapshotFeatures
from .decay_ablation import temporal_features, SUFFIXES
from .selection_stability import (candidate_snapshot, dates_for, run_case,
    sampled_users, select_users, negative_sample, fit_model, evaluate_model,
    save_npz, seed_for)
from .retrieval_benchmark import save, read
from .metrics import paired_bootstrap


def validate(c):
    if c['train_users'] != [2000] and (max(c['train_users']) > 2000 or min(c['train_users']) < 1):
        raise ValueError('Train cap is 2000 users per snapshot')
    if len(c['train_users']) != 1: raise ValueError('One common train size required')
    if c['train_pool_users'] < max(max(c['train_users']), c['validation_users']):
        raise ValueError('Insufficient feature pool')
    if pd.Timestamp(c['validation_cutoff']) + pd.Timedelta(days=7) > pd.Timestamp(c['test_cutoff']):
        raise ValueError('Validation labels overlap test')
    if len(set(c['seeds'])) != len(c['seeds']): raise ValueError('Duplicate seeds')
    if c['test_status'] != 'previously_examined_exploratory':
        raise ValueError('Do not label reused August as untouched')


def specs(c):
    result = {'uniform': {'kind': 'uniform'}}
    for h in c['half_lives']:
        base = {'kind':'decay', 'half_life':h, 'alpha':0., 'beta':0., 'adaptive_power':0.}
        result[f'h_{h}'] = base
        for key,values in [('alpha',c['alpha_values']),('beta',c['beta_values']),('adaptive_power',c['adaptive_powers'])]:
            short = 'adaptive' if key=='adaptive_power' else key
            for v in values: result[f'h_{h}_{short}_{v}'] = {**base,key:v}
    return result


def snapshot(date, c, out, events, items, users):
    date=str(pd.Timestamp(date).date());cache=out/'cache'/'features';path=cache/(date+'.pkl')
    r=candidate_snapshot(events,items,users,date,c,cache)
    schema=specs(c);fingerprint=hashlib.sha256(json.dumps(schema,sort_keys=True).encode()).hexdigest()
    if 'temporal_sha256' in r['info']:
        if r['info']['temporal_sha256'] != fingerprint: raise ValueError('Temporal specs changed')
        return
    started=perf_counter();frame=r['frame'];metadata=SnapshotFeatures(events,items,users,pd.Timestamp(date))
    relevant=metadata.history.loc[metadata.history.user_id.isin(frame.user_id.unique())]
    metadata.groups={int(u):g for u,g in relevant.groupby('user_id',sort=False)}
    names=[name+'_'+s for name in schema for s in SUFFIXES]
    values=np.empty((len(frame),len(names)),dtype=np.float32)
    for uid,rows in frame.groupby('user_id',sort=False).indices.items():
        ages,sim,_,regularity,gap=metadata._history_arrays(uid,frame.iloc[rows].item_id.astype('int64').to_numpy())
        for index,spec in enumerate(schema.values()):
            values[rows,index*3:index*3+3]=temporal_features(ages,sim,regularity,spec,gap)
    r['frame']=pd.concat([frame.reset_index(drop=True),pd.DataFrame(values,columns=names)],axis=1)
    r['columns']['temporal']={name:[name+'_'+s for s in SUFFIXES] for name in schema}
    r['info'].update(temporal_sha256=fingerprint,temporal_seconds=perf_counter()-started,fixed_S='genre_jaccard')
    temp=path.with_suffix('.tmp');pd.to_pickle(r,temp);temp.replace(path)
    save(out/'snapshots'/(date+'.json'),r['info'])
    print('Temporal snapshot',date,round(perf_counter()-started,1),'s',flush=True)


def load(date,out): return pd.read_pickle(out/'cache'/'features'/(str(pd.Timestamp(date).date())+'.pkl'))


def evaluation_frame(date,c,out):
    r=load(date,out);ids=sampled_users(r['frame'],c['validation_users'],c['validation_seed'],date)
    frame=select_users(r['frame'],ids);truth={int(u):r['truth'][int(u)] for u in ids}
    return frame,truth,r['columns']


def training(c,out,seed):
    fits=[];cohorts=[]
    for date in dates_for(c['validation_cutoff'],c):
        date=str(date.date());r=load(date,out)
        ids=sampled_users(r['frame'],c['train_users'][0],seed,date)
        frame=select_users(r['frame'],ids)
        fits.append(negative_sample(frame,c['max_negatives'],seed,date))
        cohorts.append({'cutoff':date,'users':ids.tolist()})
        del r,frame;gc.collect()
    return pd.concat(fits,ignore_index=True),cohorts


def trial(name,cols,train,valid,truth,columns,c,seed,directory):
    path=directory/(name+'.cbm');meta=directory/(name+'_fit.json')
    if meta.exists() and 'validation_metrics' in read(meta):
        record=read(meta)
        if record['columns']!=cols or not path.exists():raise ValueError('Trial checkpoint mismatch')
        return record['validation_metrics']
    if path.exists():
        model=CatBoostRanker();model.load_model(str(path));seconds=read(meta)['fit_seconds']
        if model.feature_names_!=cols:raise ValueError('Model columns mismatch')
    else:
        print('Validation fit',seed,name,flush=True)
        model,seconds=fit_model(train,cols,columns['cats'],c,seed)
        temp=path.with_suffix('.tmp');model.save_model(str(temp));temp.replace(path)
    met,_=evaluate_model(model,valid,cols,truth,c,directory,name)
    save(meta,{'columns':cols,'fit_seconds':seconds,'train_rows':len(train),'validation_metrics':met})
    del model;gc.collect();return met


def choose(names,metrics):
    # Exact ties are resolved deterministically in the predeclared input order.
    return max(names,key=lambda n:metrics[n]['ndcg@10'])


def select_case(c,out,seed,valid,truth,columns):
    directory=out/'validation'/f'seed{seed}';directory.mkdir(parents=True,exist_ok=True)
    if (directory/'decisions.json').exists():return read(directory/'decisions.json')
    date=c['validation_cutoff']
    paths=[out/'cache'/'features'/(str(d.date())+'.pkl') for d in dates_for(date,c)]
    # This existing routine generates forward OOF and the three train-only selectors.
    source=run_case((pd.read_pickle(p)['frame'] for p in paths),valid,truth,columns,c,date,seed,c['train_users'][0],directory)
    selection=read(directory/'selection.json');train,cohorts=training(c,out,seed)
    if cohorts!=read(directory/'cohorts.json'):raise ValueError('Training cohorts changed')
    baseline=columns['manual'];families={};metrics={};cols_by_trial={}
    def fit(name,extra):
        cols=baseline+list(extra);cols=list(dict.fromkeys(cols));cols_by_trial[name]=cols
        metrics[name]=trial(name,cols,train,valid,truth,columns,c,seed,directory)
        return metrics[name]['ndcg@10']
    for name,extra in selection['features'].items():
        fit(name,extra);families[name]=name
    bank=selection['bank_kept'];fit('all_relational',bank);families['all_relational']='all_relational'
    k=min(c['selected_features'],len(bank));rng=np.random.default_rng(seed_for(seed,date,'feature_search'))
    random_sets=[]
    for i in range(c['search_trials']):
        subset=sorted(rng.choice(bank,k,replace=False).tolist())
        while subset in random_sets:subset=sorted(rng.choice(bank,k,replace=False).tolist())
        random_sets.append(subset);fit(f'random_{i}',subset)
    families['random_search']=choose([f'random_{i}' for i in range(c['search_trials'])],metrics)
    priority=sorted(bank,key=lambda f:(-selection['cmim']['relevance'][f],f))
    guided_sets=[];guided_names=[]
    for i in range(c['search_trials']):
        if i==0:subset=sorted(selection['features']['cmim']);name='cmim'
        elif i==1:subset=sorted(selection['features']['mi']);name='mi'
        else:
            parent=guided_sets[int(np.argmax([metrics[n]['ndcg@10'] for n in guided_names]))]
            keep=rng.choice(parent,max(0,k-c['mutation_size']),replace=False).tolist()
            available=[f for f in priority if f not in keep];p=1/np.sqrt(np.arange(1,len(available)+1,dtype=float));p/=p.sum()
            subset=sorted(keep+rng.choice(available,k-len(keep),replace=False,p=p).tolist());name=f'guided_{i}'
        while subset in guided_sets:
            keep=rng.choice(subset,max(0,k-c['mutation_size']),replace=False).tolist()
            available=[f for f in priority if f not in keep];p=1/np.sqrt(np.arange(1,len(available)+1,dtype=float));p/=p.sum()
            subset=sorted(keep+rng.choice(available,k-len(keep),replace=False,p=p).tolist());name=f'guided_{i}'
        if name not in metrics:fit(name,subset)
        guided_sets.append(subset);guided_names.append(name)
    families['guided_search']=choose(guided_names,metrics)
    temporal=columns['temporal'];fit('uniform',temporal['uniform']);families['uniform']='uniform'
    hs=[f'h_{h}' for h in c['half_lives']]
    for name in hs:fit(name,temporal[name])
    best_h=choose(hs,metrics);families['constant_decay']=best_h
    for factor,values in [('alpha',c['alpha_values']),('beta',c['beta_values']),('adaptive',c['adaptive_powers'])]:
        options=[best_h]+[best_h+'_'+factor+'_'+str(v) for v in values]
        for name in options[1:]:fit(name,temporal[name])
        families[{'alpha':'tuned_alpha','beta':'tuned_beta','adaptive':'activity_adaptive'}[factor]]=choose(options,metrics)
    # Each optional factor includes its zero/base option; validation can reject it.
    multiscale=[f for name in hs for f in temporal[name]]
    fit('multiscale_decay',multiscale);families['multiscale_decay']='multiscale_decay'
    guided_extra=[f for f in cols_by_trial[families['guided_search']] if f not in baseline]
    fit('guided_multiscale',guided_extra+multiscale);families['guided_multiscale']='guided_multiscale'
    fit('cmim_decay',selection['features']['cmim']+temporal[best_h]);families['cmim_decay']='cmim_decay'
    arms={name:{'trial':t,'columns':cols_by_trial[t],'model_sha256':hashlib.sha256((directory/(t+'.cbm')).read_bytes()).hexdigest(),
        'validation_metrics':metrics[t]} for name,t in families.items()}
    decision={'seed':seed,'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'arms':arms,
        'validation_winner':max(arms,key=lambda a:arms[a]['validation_metrics']['ndcg@10']),
        'best_h':best_h,'random_sets':random_sets,'guided_sets':guided_sets,'guided_names':guided_names,
        'test_used_for_selection':False,'primary_contrast':['manual','guided_search'],
        'selector_rows_common_for_mi_cmim_oof':selection['selector_rows']}
    save(directory/'trial_metrics.json',metrics);save(directory/'decisions.json',decision)
    print('Decisions frozen',seed,'h',best_h,'winner',decision['validation_winner'],flush=True)
    del train;gc.collect();return decision


def evaluate_test(c,out,seed,decision,frame,truth):
    directory=out/'test'/f'seed{seed}';directory.mkdir(parents=True,exist_ok=True)
    if (directory/'complete.json').exists():return read(directory/'complete.json')
    metrics={};pers={};costs={}
    for name,arm in decision['arms'].items():
        path=out/'validation'/f'seed{seed}'/(arm['trial']+'.cbm')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=arm['model_sha256']:raise ValueError('Frozen model changed')
        print('Test inference',seed,name,flush=True);start=perf_counter()
        model=CatBoostRanker();model.load_model(str(path))
        met,per=evaluate_model(model,frame,arm['columns'],truth,c,directory,name)
        metrics[name]=met;pers[name]=per;costs[name]=perf_counter()-start
        del model;gc.collect()
    comparisons={name:paired_bootstrap(pers['manual'],per,metric='ndcg',n_boot=c['bootstrap'],seed=c['pool_seed']) for name,per in pers.items() if name!='manual'}
    comparisons['guided_minus_random']=paired_bootstrap(pers['random_search'],pers['guided_search'],metric='ndcg',n_boot=c['bootstrap'],seed=c['pool_seed'])
    comparisons['oof_minus_cmim']=paired_bootstrap(pers['cmim'],pers['cmim_oof'],metric='ndcg',n_boot=c['bootstrap'],seed=c['pool_seed'])
    comparisons['constant_minus_uniform']=paired_bootstrap(pers['uniform'],pers['constant_decay'],metric='ndcg',n_boot=c['bootstrap'],seed=c['pool_seed'])
    record={'seed':seed,'metrics':metrics,'bootstrap':comparisons,'inference_and_evaluation_seconds':costs,
        'test_cutoff':c['test_cutoff'],'validation_winner':decision['validation_winner']}
    save(directory/'complete.json',record);return record


def summarize(c,out):
    rows=[];contrasts=[];costs=[]
    for seed in c['seeds']:
        r=read(out/'test'/f'seed{seed}'/'complete.json');d=read(out/'validation'/f'seed{seed}'/'decisions.json')
        for name,m in r['metrics'].items():
            rows.append({'seed':seed,'method':name,'n_features':len(d['arms'][name]['columns']),**m,
                'delta_manual':m['ndcg@10']-r['metrics']['manual']['ndcg@10']})
            fit=read(out/'validation'/f'seed{seed}'/(d['arms'][name]['trial']+'_fit.json'))
            costs.append({'seed':seed,'method':name,'selected_model_fit_seconds':fit['fit_seconds'],
                'inference_and_evaluation_seconds':r['inference_and_evaluation_seconds'][name]})
        for name,v in r['bootstrap'].items():contrasts.append({**v,'bootstrap_seed':v['seed'],'seed':seed,'comparison':name})
    t=pd.DataFrame(rows);t.to_csv(out/'metrics.csv',index=False)
    summary=t.groupby('method').agg(ndcg_mean=('ndcg@10','mean'),ndcg_std=('ndcg@10','std'),map_mean=('map@10','mean'),
        recall_mean=('recall@10','mean'),delta_manual=('delta_manual','mean'),positive_seeds=('delta_manual',lambda s:int((s>0).sum())),features_min=('n_features','min'),features_max=('n_features','max'))
    summary.to_csv(out/'summary.csv');pd.DataFrame(contrasts).to_csv(out/'paired_contrasts.csv',index=False);pd.DataFrame(costs).to_csv(out/'costs.csv',index=False)
    # Aggregate user metrics across the same three fitted seeds before bootstrapping.
    average={}
    for name in summary.index:
        frames=[pd.read_csv(out/'test'/f'seed{s}'/(name+'_per_user.csv')) for s in c['seeds']]
        average[name]=pd.concat(frames).groupby('user_id',as_index=False).mean(numeric_only=True)
    pooled={name:paired_bootstrap(average['manual'],per,metric='ndcg',n_boot=c['bootstrap'],seed=c['pool_seed']) for name,per in average.items() if name!='manual'}
    save(out/'seed_averaged_user_bootstrap.json',{'scope':'Conditional on the three fixed fitted seeds; does not include model-selection or general training uncertainty','comparisons':pooled})
    lines=['# Сравнение методов на одном общем test','',
        f"Test: {c['test_cutoff']} — 2021-08-22; validation: {c['validation_cutoff']} — 2021-08-15. Три seeds {c['seeds']}, до {c['train_users'][0]} train-пользователей на snapshot.",
        'Общий retrieval: popularity/genre + ItemKNN + EASE; ровно 100 кандидатов и 5000 warm future-active пользователей. Все методы получают одинаковые кандидаты, историю, ручной baseline и train sample внутри seed. Общий CatBoost: 200 деревьев, depth 4, lr 0.07; дополнительного tuning ranker нет.',
        'Все feature-наборы и параметры выбирались только на train/validation, решения всех seeds сохранены до test inference. Test не используется для выбора нового метода. Август уже исследовался раньше: это общий exploratory test, а не восстановленный нетронутый holdout.','',
        '## Качество','', '| Метод | NDCG@10 mean | Std seeds | MAP@10 | Recall@10 | Δ к manual |','|---|---:|---:|---:|---:|---:|']
    for name,r in summary.iterrows():lines.append(f"| {name} | {r.ndcg_mean:.6f} | {r.ndcg_std:.6f} | {r.map_mean:.6f} | {r.recall_mean:.6f} | {r.delta_manual:+.6f} |")
    lines+=['','## Выбор до test','']
    for seed in c['seeds']:
        d=read(out/'validation'/f'seed{seed}'/'decisions.json');lines.append(f"- Seed {seed}: validation winner `{d['validation_winner']}`, half-life `{d['best_h']}`; выбранные alpha/beta/adaptive trials: "+', '.join(d['arms'][n]['trial'] for n in ['tuned_alpha','tuned_beta','activity_adaptive'])+'.')
    lines+=['','## Выводы и границы','',
        'Основной contrast — guided_search минус manual; random_search имеет столько же уникальных feature-set trials (8). Guided дополнительно оплачивает selector. У каждого optional decay-фактора есть нулевая опция; validation может выбрать обычный constant decay. Multi-scale — фиксированная конкатенация kernels, а не обучаемая смесь.',
        'Внутри каждой пары одинаковы test-пользователи; per-seed user-bootstrap сохранён в paired_contrasts.csv. Среднее по seeds описательное. Дополнительный bootstrap усреднённых per-user метрик условен на трёх обученных моделях и не охватывает неопределённость feature search или полного процесса обучения. Поправки на множественные сравнения нет.',
        'MAP использует min(10, |truth|) в знаменателе AP. Полная future truth остаётся в Recall и IDCG, включая объекты вне candidate pool. Нет impression log: sampled нули не означают подтверждённый отказ после показа.',
        'Сравнение приводит все методы к одному протоколу и не переносит абсолютные числа старых исследований. Независимость от ранее просмотренных августовских метрик этим не восстанавливается.','',
        '## Воспроизведение','', '```bash','OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=6 PYTHONPATH=src python -m kion_recsys.common_test --config configs/common_test.json','PYTHONPATH=src python scripts/verify_common_test.py','```']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    from .common_test_analysis import enrich
    enrich(c,out)


def execute(c):
    validate(c);out=Path(c['output_dir']);out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists() and read(out/'config.json')!=c:raise ValueError('Config changed')
    save(out/'config.json',c)
    if not (out/'protocol_before_results.json').exists():save(out/'protocol_before_results.json',{'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'config':c,'primary_contrast':['manual','guided_search'],'test_is_untouched':False})
    normalized=out/'cache'/'inputs.pkl'
    if normalized.exists():events,items,users=pd.read_pickle(normalized)
    else:
        events,items,users=load_data(Path(c['data_dir']));events=events[['user_id','item_id','timestamp']].copy()
        normalized.parent.mkdir(parents=True,exist_ok=True);pd.to_pickle((events,items,users),normalized)
    for date in [*dates_for(c['validation_cutoff'],c),c['validation_cutoff']]:
        snapshot(date,c,out,events,items,users);gc.collect()
    valid,truth,columns=evaluation_frame(c['validation_cutoff'],c,out)
    vd=out/'validation';vd.mkdir(exist_ok=True);save(vd/'truth.json',{str(u):sorted(v) for u,v in truth.items()})
    save_npz(vd/'candidates.npz',user_id=valid.user_id.to_numpy(),item_id=valid.item_id.astype('int64').to_numpy(),label=valid.label.to_numpy(dtype=np.uint8))
    decisions={str(seed):select_case(c,out,seed,valid,truth,columns) for seed in c['seeds']}
    frozen=out/'ALL_DECISIONS_FROZEN.json'
    if not frozen.exists():save(frozen,{'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'decisions':decisions})
    elif read(frozen)['decisions']!=decisions:raise ValueError('Frozen decisions differ')
    del valid,truth;gc.collect()
    # Test candidate/feature construction and any scoring occur after every decision.
    if not (out/'TEST_STARTED.json').exists():save(out/'TEST_STARTED.json',{'started_at_utc':datetime.now(timezone.utc).isoformat(),'frozen_decisions_sha256':hashlib.sha256(frozen.read_bytes()).hexdigest()})
    snapshot(c['test_cutoff'],c,out,events,items,users)
    del events,items,users;gc.collect()
    frame,truth,_=evaluation_frame(c['test_cutoff'],c,out);td=out/'test';td.mkdir(exist_ok=True)
    save(td/'truth.json',{str(u):sorted(v) for u,v in truth.items()})
    save_npz(td/'candidates.npz',user_id=frame.user_id.to_numpy(),item_id=frame.item_id.astype('int64').to_numpy(),label=frame.label.to_numpy(dtype=np.uint8))
    for seed in c['seeds']:evaluate_test(c,out,seed,decisions[str(seed)],frame,truth)
    summarize(c,out);save(out/'complete.json',{'status':'complete','test_cutoff':c['test_cutoff'],'test_is_untouched':False,'all_decisions_before_test':True})


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();execute(read(a.config))

if __name__=='__main__':main()
