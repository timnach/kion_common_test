"""One-factor temporal ablations at a fixed genre-Jaccard similarity.

h is selected on an earlier calibration week, shared by all subsequent arms.
Alpha, beta and activity adaptation are evaluated separately, never combined.
"""
import argparse
from datetime import datetime,timezone
import gc
import hashlib
import itertools
import json
from pathlib import Path
import shutil
from time import perf_counter
import numpy as np
import pandas as pd
from catboost import CatBoostRanker
from .features import SnapshotFeatures
from .retrieval_benchmark import read,save
from .selection_stability import (sampled_users,select_users,negative_sample,fit_model,
    evaluate_model,save_npz,dates_for)
from .metrics import paired_bootstrap


SUFFIXES=['affinity','mass','ess']


def effective_half_life(base,n,gap,power=0.):
    if not np.isfinite(base) or base<=0 or not np.isfinite(gap) or gap<0 or not 0<=power<=1:
        raise ValueError('Invalid shared half-life parameters')
    return float(np.clip(base*((gap+1)/8)**(power*n/(n+10)),.25,180))


def temporal_features(ages,similarity,regularity,spec,median_gap=7.):
    """Numerically stable equivalent of weighted affinity/log-mass/ESS.

    Affinity and ESS are invariant to a common positive weight scale. Log-mass
    keeps that physical scale; very old histories must not get spurious ESS=0.
    """
    ages=np.asarray(ages,dtype=float);similarity=np.asarray(similarity,dtype=float);r=np.asarray(regularity,dtype=float)
    if ages.ndim!=1 or similarity.ndim!=2 or similarity.shape[1]!=len(ages) or r.shape!=ages.shape:
        raise ValueError('Shape mismatch')
    if not np.isfinite(similarity).all() or np.any((similarity<0)|(similarity>1)):
        raise ValueError('S must be finite in [0,1]')
    if len(ages)==0:return np.zeros((similarity.shape[0],3))
    if np.any(ages<=0) or not np.isfinite(ages).all() or not np.isfinite(r).all() or np.any(r<0):
        raise ValueError('History must be finite and strictly before cutoff')
    alpha=float(spec.get('alpha',0));beta=float(spec.get('beta',0));power=float(spec.get('adaptive_power',0))
    if not np.isfinite(alpha+beta) or alpha<0 or beta<0:raise ValueError('Invalid weighting coefficients')
    if spec['kind']=='uniform':
        if alpha or beta or power:raise ValueError('Uniform control cannot mix weighting factors')
        loga=np.zeros(len(ages))
    else:
        h=effective_half_life(spec['half_life'],len(ages),median_gap,power)
        loga=-np.log(2)*ages/h+np.log1p(alpha*r)
    offset=float(loga.max());a=np.exp(loga-offset)
    weights=a[None,:]*(1+beta*similarity)
    denom=weights.sum(axis=1);mass=(weights*similarity).sum(axis=1)
    affinity=mass/denom;ess=denom**2/np.square(weights).sum(axis=1)
    logmass=np.full(len(mass),-np.inf);np.log(mass,out=logmass,where=mass>0)
    return np.column_stack([affinity,np.logaddexp(0,logmass+offset),ess])


def h_specs(config):
    result={'uniform':{'kind':'uniform'}}
    result.update({f'h_{h}':{'kind':'decay','half_life':h,'alpha':0.,'beta':0.,'adaptive_power':0.} for h in config['half_lives']})
    return result


def final_specs(config,h):
    base={'kind':'decay','half_life':h,'alpha':0.,'beta':0.,'adaptive_power':0.}
    specs={'uniform':{'kind':'uniform'},'constant':dict(base)}
    for a in config['alpha_values']:specs['alpha_'+str(a)]={**base,'alpha':a}
    for b in config['beta_values']:specs['beta_'+str(b)]={**base,'beta':b}
    for p in config['adaptive_powers']:specs['adaptive_'+str(p)]={**base,'adaptive_power':p}
    return specs


def validate(config):
    if config['train_users']>2000 or config['train_users']<1:raise ValueError('Maximum 2000 train users per snapshot')
    if config['similarity']!='genre_jaccard':raise ValueError('S is fixed to genre Jaccard')
    if pd.Timestamp(config['calibration_cutoff'])+pd.Timedelta(days=7)>min(map(pd.Timestamp,config['validation_cutoffs'])):
        raise ValueError('Calibration labels must precede every diagnostic validation')
    for start in [config['calibration_cutoff'],*config['validation_cutoffs']]:
        a=pd.Timestamp(start);b=a+pd.Timedelta(days=7)
        if any(a<pd.Timestamp(t)+pd.Timedelta(days=7) and b>pd.Timestamp(t) for t in config['forbidden_test_cutoffs']):
            raise ValueError('Previously examined test horizon is forbidden')


def base_snapshot(date,config,out):
    date=str(pd.Timestamp(date).date());path=out/'cache'/'base'/(date+'.pkl')
    if path.exists():return path
    source=Path(config['source_study'])/'cache'/'features'/(date+'.pkl')
    if not source.exists():raise FileNotFoundError(f'Rebuild the selection_stability feature cache first: {source}')
    record=pd.read_pickle(source);cols=record['columns']['manual']
    names=list(dict.fromkeys(['user_id','item_id','label','query_id',*cols]))
    frame=record['frame'][names].copy();truth=record['truth'];info=record['info'];columns=record['columns']
    del record;gc.collect();path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');pd.to_pickle({'frame':frame,'truth':truth,'info':info,'columns':{'manual':cols,'cats':columns['cats']}},temp);temp.replace(path)
    return path


def bank(date,config,out,specs,phase,events,items,users):
    date=str(pd.Timestamp(date).date());path=out/'cache'/phase/(date+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
    fingerprint=hashlib.sha256(json.dumps(specs,sort_keys=True).encode()).hexdigest()
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            if str(z['spec_sha256'])!=fingerprint:raise ValueError('Decay bank configuration changed')
        return path
    start=perf_counter();record=pd.read_pickle(base_snapshot(date,config,out));frame=record['frame']
    metadata=SnapshotFeatures(events,items,users,pd.Timestamp(date))
    relevant=metadata.history.loc[metadata.history.user_id.isin(frame.user_id.unique())]
    metadata.groups={int(u):g for u,g in relevant.groupby('user_id',sort=False)}
    names=[name+'_'+s for name in specs for s in SUFFIXES]
    result=np.zeros((len(frame),len(names)),dtype=np.float32);diagnostics=[]
    for uid,rows in frame.groupby('user_id',sort=False).indices.items():
        ids=frame.iloc[rows].item_id.astype('int64').to_numpy()
        ages,sim,_,regularity,gap=metadata._history_arrays(uid,ids)
        for index,(name,spec) in enumerate(specs.items()):
            result[rows,3*index:3*index+3]=temporal_features(ages,sim,regularity,spec,gap)
        diagnostics.append({'user_id':int(uid),'history_n':len(ages),'median_gap':gap,
            'regularity_nonzero':int((regularity>0).sum()),'regularity_range':float(np.ptp(regularity)) if len(ages) else 0.,
            **{name+'_effective_h':effective_half_life(spec['half_life'],len(ages),gap,spec.get('adaptive_power',0))
               for name,spec in specs.items() if spec['kind']=='decay' and spec.get('adaptive_power',0)>0}})
    save_npz(path,user_id=frame.user_id.to_numpy(),item_id=frame.item_id.astype('int64').to_numpy(),values=result,
        names=np.asarray(names),spec_sha256=np.asarray(fingerprint))
    directory=out/'feature_diagnostics'/phase;directory.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(diagnostics).to_csv(directory/(date+'.csv'),index=False)
    save(directory/(date+'.json'),{'cutoff':date,'feature_seconds':perf_counter()-start,'n_users':len(diagnostics),'rows':len(frame),
        'fixed_S':'genre_jaccard','candidate_sha256':record['info']['candidate_sha256'],'specs':specs})
    print('Decay bank',phase,date,len(frame),'rows',round(perf_counter()-start,1),'s',flush=True)
    return path


def load_frame(date,config,out,phase):
    r=pd.read_pickle(base_snapshot(date,config,out));path=out/'cache'/phase/(str(pd.Timestamp(date).date())+'.npz')
    with np.load(path,allow_pickle=False) as z:
        if not np.array_equal(z['user_id'],r['frame'].user_id) or not np.array_equal(z['item_id'],r['frame'].item_id.astype('int64')):
            raise ValueError('Decay rows do not match the shared candidates')
        extra=pd.DataFrame(z['values'],columns=z['names'].tolist())
    r['frame']=pd.concat([r['frame'].reset_index(drop=True),extra],axis=1)
    return r


def training(dates,seed,config,out,phase):
    fits=[];cohorts=[]
    for date in dates:
        date=str(pd.Timestamp(date).date());r=load_frame(date,config,out,phase);frame=r['frame']
        ids=sampled_users(frame,config['train_users'],seed,date)
        selected=select_users(frame,ids);fits.append(negative_sample(selected,config['max_negatives'],seed,date))
        cohorts.append({'cutoff':date,'users':ids.tolist()})
        del r,frame,selected;gc.collect()
    return pd.concat(fits,ignore_index=True),cohorts


def validation(date,config,out,phase):
    date=str(pd.Timestamp(date).date());r=load_frame(date,config,out,phase);frame=r['frame']
    ids=sampled_users(frame,config['validation_users'],config['validation_seed'],date)
    result=select_users(frame,ids);truth={int(u):r['truth'][int(u)] for u in ids}
    return result,truth,r['columns']


def model_arm(train,valid,truth,baseline,cats,name,cols,seed,config,directory,reuse=None):
    path=directory/(name+'.cbm');meta=directory/(name+'_fit.json');start=perf_counter()
    if path.exists():model=CatBoostRanker();model.load_model(str(path));seconds=read(meta)['fit_seconds']
    elif reuse is not None:
        model=CatBoostRanker();model.load_model(str(reuse))
        if model.feature_names_!=cols:raise ValueError('Reused baseline feature schema mismatch')
        shutil.copy2(reuse,path);seconds=None
        save(meta,{'columns':cols,'fit_seconds':None,'reused_baseline':str(reuse),'source_sha256':hashlib.sha256(Path(reuse).read_bytes()).hexdigest()})
    else:
        print('Fit',directory.name,name,flush=True)
        model,seconds=fit_model(train,cols,cats,config,seed)
        temp=path.with_suffix('.tmp');model.save_model(str(temp));temp.replace(path)
        save(meta,{'columns':cols,'fit_seconds':seconds,'train_rows':len(train)})
    met,per=evaluate_model(model,valid,cols,truth,config,directory,name)
    del model;gc.collect()
    return met,per,seconds


def calibrate(config,out):
    choice=out/'h_choice_before_validation.json'
    if choice.exists():return read(choice)['half_life']
    fold=config['calibration_cutoff'];dates=[pd.Timestamp(fold)-pd.Timedelta(days=7*i) for i in range(config['calibration_train_snapshots'],0,-1)]
    valid,truth,columns=validation(fold,config,out,'h_grid');baseline=columns['manual'];rows=[]
    save(out/'calibration_truth.json',{str(u):sorted(v) for u,v in truth.items()})
    save_npz(out/'calibration_candidates.npz',user_id=valid.user_id.to_numpy(),item_id=valid.item_id.astype('int64').to_numpy(),label=valid.label.to_numpy(dtype=np.uint8))
    for seed in config['seeds']:
        directory=out/'calibration'/f'seed{seed}';directory.mkdir(parents=True,exist_ok=True)
        train,cohorts=training(dates,seed,config,out,'h_grid');save(directory/'cohorts.json',cohorts)
        for h in config['half_lives']:
            name='h_'+str(h);cols=baseline+[name+'_'+s for s in SUFFIXES]
            metrics,_,seconds=model_arm(train,valid,truth,baseline,columns['cats'],name,cols,seed,config,directory)
            rows.append({'half_life':h,'seed':seed,'fit_seconds':seconds,**metrics})
        del train;gc.collect()
    table=pd.DataFrame(rows);table.to_csv(out/'h_grid_metrics.csv',index=False)
    means=table.groupby('half_life')[f"ndcg@{config['k']}"].mean();best=float(means.idxmax())
    save(choice,{'half_life':best,'selected_at_utc':datetime.now(timezone.utc).isoformat(),'calibration_cutoff':fold,
        'rule':'maximum mean calibration NDCG across predeclared seeds; smallest h breaks exact ties','mean_ndcg':{str(h):float(v) for h,v in means.items()},
        'no_diagnostic_validation_used':True})
    print('Frozen h',best,flush=True);return best


def case(fold,seed,h,config,out):
    directory=out/fold/f'seed{seed}';directory.mkdir(parents=True,exist_ok=True)
    if (directory/'complete.json').exists():return
    valid,truth,columns=validation(fold,config,out,'ablation');dates=dates_for(fold,config)
    train,cohorts=training(dates,seed,config,out,'ablation');save(directory/'cohorts.json',cohorts)
    source=Path(config['source_study'])/fold/f"n{config['train_users']}_seed{seed}"
    if cohorts!=read(source/'cohorts.json'):raise ValueError('Cohorts differ from the shared baseline experiment')
    save(out/fold/'truth.json',{str(u):sorted(v) for u,v in truth.items()})
    save_npz(out/fold/'candidates.npz',user_id=valid.user_id.to_numpy(),item_id=valid.item_id.astype('int64').to_numpy(),label=valid.label.to_numpy(dtype=np.uint8))
    metrics={};pers={};times={};baseline=columns['manual'];specs=final_specs(config,h)
    for name in ['manual',*specs]:
        cols=baseline if name=='manual' else baseline+[name+'_'+s for s in SUFFIXES]
        reuse=source/'manual.cbm' if name=='manual' else None
        met,per,seconds=model_arm(train,valid,truth,baseline,columns['cats'],name,cols,seed,config,directory,reuse)
        metrics[name]=met;pers[name]=per;times[name]=seconds
        if name=='manual':
            expected=read(source/'complete.json')['metrics']['manual']
            for key in ['ndcg@10','map@10','recall@10']:np.testing.assert_allclose(met[key],expected[key],atol=1e-12)
    contrasts={name:paired_bootstrap(pers['constant'],per,metric='ndcg',n_boot=config['bootstrap'],seed=config['pool_seed'])
        for name,per in pers.items() if name not in {'manual','constant'}}
    contrasts['constant_minus_uniform']=paired_bootstrap(pers['uniform'],pers['constant'],metric='ndcg',n_boot=config['bootstrap'],seed=config['pool_seed'])
    contrasts['constant_minus_manual']=paired_bootstrap(pers['manual'],pers['constant'],metric='ndcg',n_boot=config['bootstrap'],seed=config['pool_seed'])
    save(directory/'complete.json',{'fold':fold,'seed':seed,'half_life':h,'metrics':metrics,'bootstrap_vs_constant':contrasts,'fit_seconds':times})
    print('Case complete',fold,seed,flush=True)


def summarize(out,config,h):
    rows=[]
    for fold,seed in itertools.product(config['validation_cutoffs'],config['seeds']):
        r=read(out/fold/f'seed{seed}'/'complete.json')
        for name,met in r['metrics'].items():rows.append({'fold':fold,'seed':seed,'method':name,**met,
            'delta_constant':met['ndcg@10']-r['metrics']['constant']['ndcg@10'],
            'delta_manual':met['ndcg@10']-r['metrics']['manual']['ndcg@10']})
    table=pd.DataFrame(rows);table.to_csv(out/'metrics.csv',index=False)
    summary=table.groupby(['fold','method'])['ndcg@10'].agg(['mean','std','min','max']).reset_index();summary.to_csv(out/'quality_summary.csv',index=False)
    lines=['# Разделение эффектов temporal decay','',
        'S фиксировано: Jaccard жанров кандидата и исторического объекта. Retrieval, кандидаты, train cohorts, negatives, ручной baseline и настройки ранкера общие. Максимум 2000 train-пользователей на snapshot. Индивидуальные свободные lambda не обучаются.','',
        '## Протокол до результатов','',
        f"Калибровка h: {config['calibration_cutoff']}, варианты {config['half_lives']}, выбор по среднему NDCG@10 трёх seeds. Выбран h={h:g} дней до расчёта последующих диагностических фолдов {config['validation_cutoffs']}.",
        f"Затем при h={h:g}: alpha={config['alpha_values']} при beta=gamma=0; beta={config['beta_values']} при alpha=gamma=0; activity gamma={config['adaptive_powers']} при alpha=beta=0. Эти ветки не смешиваются. Это одномерные ограниченные проверки, не полный joint search.",
        'Baseline переиспользован из серии устойчивости только после проверки совпадения когорт, признаков, параметров и NDCG/MAP/Recall. Ни один контроль не получает дополнительного подбора ранкера. Все временные варианты добавляют ровно три колонки: affinity, log-mass, ESS. Uniform использует ту же S без временного затухания; он отделяет эффект времени от добавления агрегатов сходства.',
        f"Ранкер: {config['loss']}, {config['ranker']}; seeds {config['seeds']}; train {config['train_snapshots']} предыдущих недель, {config['train_users']} пользователей на неделю. Candidate budget=100. Августовские просмотренные test исключены. Июль — development-диагностика на перекрывающихся фолдах, а не независимый подтверждающий test.",'',
        '## Формула','',
        'w_ui,j = 2^(-age_j / h_u) × (1 + alpha R_u,j) × (1 + beta S_i,j). Affinity = sum(w S)/sum(w); log-mass = log(1+sum(w S)); ESS = sum(w)^2/sum(w^2).',
        'R — прежняя регулярность жанров в истории: shrinkage по числу уникальных дней и штраф за CV интервалов, усреднённые по жанрам исторического объекта. Alpha меняет только этот множитель; beta — только усиление похожих исторических объектов.',
        'Shared activity-adaptive h_u = clip(h × ((median_gap_u + 1)/8)^(gamma n_u/(n_u+10)), 0.25, 180). Gamma общая для всех пользователей; индивидуальных fitted коэффициентов нет. Median gap считается по наблюдаемым дням активности, не по полному журналу просмотров. Для коротких историй адаптация сжимается к общему h.',
        'Численно веса масштабируются общим положительным множителем для affinity/ESS; log-mass сохраняет исходный масштаб. Это убирает артефакт старого порога 1e-30, который мог обнулять ESS старой истории при малом h. Старый kernel и результаты не переписаны; новый kernel проверен по формуле и эквивалентности при обычных весах.','',
        '## Подбор только h на ранней неделе','', '| h, дней | Mean calibration NDCG@10 | Std по seeds |','|---:|---:|---:|']
    hs=pd.read_csv(out/'h_grid_metrics.csv').groupby('half_life')['ndcg@10'].agg(['mean','std'])
    for x,r in hs.iterrows():lines.append(f"| {x:g} | {r['mean']:.6f} | {r['std']:.6f} |")
    lines+=['','## Диагностические фолды','', '| Неделя | Вариант | Mean NDCG@10 | Std |','|---|---|---:|---:|']
    for _,r in summary.iterrows():lines.append(f"| {r.fold} | {r.method} | {r['mean']:.6f} | {r['std']:.6f} |")
    lines+=['','## Отдельные эффекты относительно constant-h','', '| Вариант | Средняя Δ NDCG | Std Δ | Min Δ | Max Δ | Положительных повторов |','|---|---:|---:|---:|---:|---:|']
    for name,g in table.loc[table.method!='constant'].groupby('method'):
        d=g.delta_constant;lines.append(f'| {name} | {d.mean():+.6f} | {d.std():.6f} | {d.min():+.6f} | {d.max():+.6f} | {int((d>0).sum())}/{len(d)} |')
    lines+=['','## Ограничения интерпретации','',
        'Парные bootstrap-интервалы в case/complete.json условны на фиксированных фолде и модели. Они не учитывают подбор h, множественные сравнения и seed-неопределённость. Положительные разности не означают универсального превосходства; взаимодействия alpha×beta×gamma здесь не проверялись.',
        'Если R равномерно нулевое, alpha вообще не меняет признаки; при постоянном R affinity/ESS не меняются, хотя mass масштабируется. Эти случаи отражены в feature_diagnostics. Больше ESS не обязательно означает лучшее ранжирование.',
        'KION — таблица последних взаимодействий пар, не полный поток событий. Сохраняются ограничения восстановления истории и интерпретации temporal windows. Отсутствие наблюдаемого события не равно отрицательному отклику.','',
        '## Воспроизведение','', '```bash','python -m pip install -e ".[dev,retrieval]"','python scripts/download_data.py','# При отсутствии локального feature cache сначала восстановить selection_stability:','OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=6 python -m kion_recsys.selection_stability --config configs/selection_stability.json','OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=6 python -m kion_recsys.decay_ablation --config configs/decay_ablation.json','PYTHONPATH=src python scripts/verify_decay_ablation.py','```','',
        'Все решения h и одномерные grids сохранены до diagnostic validation. Повтор команды восстанавливает банки и модели; для нового протокола используйте новый output_dir. Большие caches не входят в архив; не загружайте чужие pickle caches.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def execute(config):
    validate(config);out=Path(config['output_dir']);out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists() and read(out/'config.json')!=config:raise ValueError('Changed protocol: use a new output directory')
    save(out/'config.json',config)
    if not (out/'protocol_before_results.json').exists():save(out/'protocol_before_results.json',{'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'config':config,
        'design':'one-factor-at-a-time; h selected on earlier calibration only; no individual lambda; no test selection'})
    source=read(Path(config['source_study'])/'config.json')
    for key in ['ranker','loss','threads','pool_seed','validation_seed','max_negatives','k','validation_users','train_snapshots']:
        if config[key]!=source[key]:raise ValueError(f'Baseline protocol mismatch: {key}')
    events,items,users=pd.read_pickle(Path(config['source_study'])/'cache'/'inputs.pkl')
    users=users[['user_id']].copy();items=items[[x for x in ['item_id','genres','content_type'] if x in items]].copy()
    calibration=pd.Timestamp(config['calibration_cutoff']);cal_dates=[calibration-pd.Timedelta(days=7*i) for i in range(config['calibration_train_snapshots'],0,-1)]+[calibration]
    for date in cal_dates:bank(date,config,out,h_specs(config),'h_grid',events,items,users)
    h=calibrate(config,out)
    save(out/'factor_specs_before_validation.json',{'half_life':h,'specs':final_specs(config,h),'S':config['similarity'],'no_joint_combinations':True})
    dates=sorted({d for fold in config['validation_cutoffs'] for d in [*dates_for(fold,config),pd.Timestamp(fold)]})
    for date in dates:bank(date,config,out,final_specs(config,h),'ablation',events,items,users)
    del events,items,users;gc.collect()
    for fold,seed in itertools.product(config['validation_cutoffs'],config['seeds']):case(fold,seed,h,config,out)
    summarize(out,config,h)
    from .decay_analysis import enrich
    enrich(out,config)
    save(out/'complete.json',{'status':'complete','no_test_evaluation':True,'shared_half_life':h})


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();execute(read(a.config))

if __name__=='__main__':main()
