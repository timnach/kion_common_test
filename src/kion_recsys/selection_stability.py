"""Frozen diagnostic rolling validation: cohort/seeds/windows/OOF CMIM.

No previously examined August test labels are used. No method, hyperparameter
or feature budget is selected from the diagnostic validation results.
"""
import argparse
from collections import Counter
import gc
import hashlib
import itertools
from io import BytesIO
import os
from pathlib import Path
from time import perf_counter
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from catboost import CatBoostRanker,Pool
from .data import load_data
from .retrieval import SnapshotRetriever
from .collaborative import CollaborativeRetriever
from .retrieval_models import BinarySnapshot,MatrixRetriever
from .retrieval_benchmark import save,read,complete_candidates
from .features import SnapshotFeatures
from .relational import SnapshotRelationalFeatures,manual_specs,candidate_specs
from .selection import CMIMSelector
from .feature_utils import numeric_dedup
from .metrics import evaluate_ranking,paired_bootstrap


def seed_for(seed,date,purpose,user=''):
    key=f'{seed}|{date}|{purpose}|{user}'.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4],'little')


def save_npz(path,**arrays):
    path=Path(path);temp=path.with_suffix('.tmp');buffer=BytesIO()
    np.savez_compressed(buffer,**arrays);payload=buffer.getvalue()
    with temp.open('wb') as f:
        if f.write(payload)!=len(payload):raise IOError('Short NPZ write')
        f.flush();os.fsync(f.fileno())
    if temp.stat().st_size!=len(payload):raise IOError('NPZ size mismatch')
    temp.replace(path)


def dates_for(fold,config):
    end=pd.Timestamp(fold)
    return [end-pd.Timedelta(days=7*i) for i in range(config['train_snapshots'],0,-1)]


def validate_protocol(config):
    if config['train_pool_users']<max(config['train_users']):raise ValueError('Pool smaller than training cohort')
    if max(config['train_users'])>config.get('max_train_users_allowed',max(config['train_users'])):
        raise ValueError('Training cohort exceeds the user-defined limit')
    if len(set(config['seeds']))!=len(config['seeds']):raise ValueError('Duplicate seeds')
    if config['train_snapshots']<2:raise ValueError('OOF requires at least two train weeks')
    for fold in config['validation_cutoffs']:
        start=pd.Timestamp(fold);end=start+pd.Timedelta(days=7)
        if any(start<pd.Timestamp(stop)+pd.Timedelta(days=7) and end>pd.Timestamp(stop) for stop in config['forbidden_test_cutoffs']):
            raise ValueError('Diagnostic validation overlaps a previously examined test')


def candidate_snapshot(events,items,users,date,config,cache):
    date=pd.Timestamp(date);path=cache/(str(date.date())+'.pkl')
    if path.exists():
        print('Restore feature snapshot',date.date(),flush=True);return pd.read_pickle(path)
    start=perf_counter();past=events.loc[events.timestamp<date]
    future=events.loc[(events.timestamp>=date)&(events.timestamp<date+pd.Timedelta(days=7))]
    eligible=np.intersect1d(past.user_id.unique(),future.user_id.unique())
    cohort=np.sort(np.random.default_rng(seed_for(config['pool_seed'],str(date.date()),'pool')).choice(eligible,min(config['train_pool_users'],len(eligible)),replace=False))
    if len(cohort)<max(config['train_users']):raise ValueError('Not enough eligible users')
    truth={int(u):set(map(int,g.item_id)) for u,g in future.loc[future.user_id.isin(cohort)].groupby('user_id')}
    legacy=SnapshotRetriever(events,items,date,**config['retrieval']['legacy'])
    native=legacy.candidates_for(cohort).sort_values(['user_id','retrieval_score','item_id'],ascending=[True,False,True]).groupby('user_id',sort=False).head(config['candidate_k'])
    native=native.copy();native['legacy_rank']=native.groupby('user_id').cumcount()+1
    knn=CollaborativeRetriever(events,items,date,**config['retrieval']['knn'])
    raw_knn=knn.candidates_for(cohort,k=config['candidate_k'])
    matrix=BinarySnapshot(events,items,date)
    ease=MatrixRetriever(matrix,'ease',threads=config['threads'],seed=config['pool_seed'],**config['retrieval']['ease'])
    raw_ease=ease.candidates_for(cohort,k=config['candidate_k']);raw_ease['ease_rank']=raw_ease.groupby('user_id').cumcount()+1
    sources={'legacy':native.rename(columns={'retrieval_score':'score'})[['user_id','item_id','score']],
             'knn':raw_knn.rename(columns={'knn_score':'score'})[['user_id','item_id','score']],
             'ease':raw_ease[['user_id','item_id','score']]}
    c=complete_candidates(sources,['legacy','knn','ease'],legacy,cohort,config['candidate_k']).rename(columns={'score':'fusion_score'})
    c=c.merge(native[['user_id','item_id','source_global','source_recent','source_genre','legacy_rank']],on=['user_id','item_id'],how='left',validate='one_to_one')
    c=c.merge(raw_knn[['user_id','item_id','knn_rank']],on=['user_id','item_id'],how='left',validate='one_to_one')
    c=c.merge(raw_ease.rename(columns={'score':'ease_score'}),on=['user_id','item_id'],how='left',validate='one_to_one')
    for name in ['legacy','knn','ease']:
        if name!='legacy':c['source_'+name]=c[name+'_rank'].notna().astype('uint8')
        c[name+'_rank']=c[name+'_rank'].fillna(config['candidate_k']+1).astype('int16')
    c['ease_score']=c.ease_score.fillna(0).astype('float32');c['knn_score']=knn.scores_for(c).astype('float32')
    for name in ['source_global','source_recent','source_genre']:c[name]=c[name].fillna(0).astype('uint8')
    c['retrieval_score']=c.item_id.map(legacy._global_scores).fillna(0)+c.item_id.map(legacy._recent_scores).fillna(0)
    metadata=SnapshotFeatures(events,items,users,date)
    # Original manual definitions are unchanged; do not calculate its unused
    # legacy automatic window bank, since the relational bank is the search space.
    metadata.windows=[1,7,30];metadata.lags=[]
    histories={int(u):g.item_id.tolist() for u,g in past.loc[past.user_id.isin(cohort)].groupby('user_id',sort=False)}
    personalized=np.zeros(len(c))
    for uid,positions in c.groupby('user_id',sort=False).indices.items():
        counts=Counter(g for item in histories.get(int(uid),[]) for g in legacy.genre_map.get(int(item),()))
        denom=sum(counts.values()) or 1
        shares=np.zeros(len(metadata.genre_idx))
        for genre,n in counts.items():shares[metadata.genre_idx[genre]]=n/denom
        ids=[metadata.idx[int(i)] for i in c.iloc[positions].item_id]
        personalized[positions]=metadata.genre_matrix[ids]@shares
    c['retrieval_score']+=personalized
    c['label']=[int(int(i) in truth[int(u)]) for u,i in zip(c.user_id,c.item_id)]
    c['query_id']=[f'{date.date()}:{u}' for u in c.user_id]
    frame=metadata.transform(c,{})
    frame=frame.drop(columns=metadata.bank_columns+metadata.trend_columns)
    manual=manual_specs();specs={s.name:s for s in candidate_specs()}
    relational=SnapshotRelationalFeatures(events,items,date)
    rel=relational.transform(c,list(specs.values()))
    frame=pd.concat([frame.reset_index(drop=True),rel.reset_index(drop=True)],axis=1)
    for name in frame.select_dtypes(include='float64'):frame[name]=frame[name].astype('float32')
    baseline=list(dict.fromkeys(metadata.manual_columns+['knn_score','source_knn','knn_rank','legacy_rank','fusion_score','ease_score','ease_rank','source_ease']+[s.name for s in manual]))
    columns={'manual':baseline,'cats':metadata.cat_columns,'bank':[s.name for s in specs.values() if s.name not in baseline]}
    info={'cutoff':str(date.date()),'label_end':str((date+pd.Timedelta(days=7)).date()),'users':len(cohort),'rows':len(frame),
          'build_seconds':perf_counter()-start,'candidate_sha256':hashlib.sha256(c[['user_id','item_id']].to_numpy(dtype=np.int64).tobytes()).hexdigest(),
          'knn_fit_seconds':knn.fit_seconds,'ease_fit_seconds':ease.fit_seconds,'manual_features':len(baseline)}
    result={'frame':frame,'truth':truth,'columns':columns,'info':info}
    cache.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');pd.to_pickle(result,temp);temp.replace(path)
    print('Built feature snapshot',date.date(),len(frame),'rows',round(info['build_seconds'],1),'s',flush=True)
    return result


def sampled_users(frame,count,seed,date):
    available=np.sort(frame.user_id.unique())
    return np.sort(np.random.default_rng(seed_for(seed,date,'cohort')).permutation(available)[:count])


def select_users(frame,ids):return frame.loc[frame.user_id.isin(ids)].reset_index(drop=True)


def negative_sample(frame,maximum,seed,date):
    """Per-query RNG makes smaller-size training rows nested in larger size."""
    selected=[];labels=frame.label.to_numpy()
    for uid,positions in frame.groupby('user_id',sort=False).indices.items():
        positive=positions[labels[positions]>0];negative=positions[labels[positions]==0]
        if len(negative)>maximum:
            negative=np.random.default_rng(seed_for(seed,date,'negative',uid)).choice(negative,maximum,replace=False)
        selected.extend(positive);selected.extend(negative)
    return frame.iloc[np.sort(selected)].reset_index(drop=True)


def fit_model(frame,columns,cats,config,seed):
    start=perf_counter();frame=frame.sort_values(['query_id','item_id'])
    frame=frame.loc[frame.groupby('query_id').label.transform('max')>0]
    if frame.empty or frame.label.nunique()<2:raise ValueError('No informative training groups')
    model=CatBoostRanker(loss_function=config['loss'],random_seed=seed,thread_count=config['threads'],
        verbose=False,allow_writing_files=False,one_hot_max_size=65535,**config['ranker'])
    pool=Pool(frame[columns],frame.label,cat_features=[name for name in cats if name in columns],group_id=frame.query_id,thread_count=1)
    model.fit(pool)
    return model,perf_counter()-start


def predict(model,frame,columns):
    return np.concatenate([model.predict(frame.iloc[start:start+4096][columns],thread_count=1) for start in range(0,len(frame),4096)])


def baseline_bins(scores,queries,n_bins):
    ranks=pd.Series(scores).groupby(np.asarray(queries)).rank(method='average',pct=True).to_numpy()
    return np.clip(np.ceil(ranks*n_bins).astype(int)-1,0,n_bins-1)


def check_oof_dates(train_dates,target):
    end=max(pd.Timestamp(d)+pd.Timedelta(days=7) for d in train_dates)
    if end>pd.Timestamp(target):raise ValueError('OOF train label horizon crosses prediction cutoff')
    return str(end.date())


def history_context(frame):
    edges=np.unique(np.quantile(frame.item_count_7d,[1/3,2/3]))
    context=pd.DataFrame({'activity':np.digitize(frame.user_count_all,[5,20]),
        'genre_match':(frame.genre_affinity_all.to_numpy()>0).astype(int),
        'popularity':np.digitize(frame.item_count_7d,edges)})
    return context,edges.tolist()


def evaluate_model(model,frame,columns,truth,config,directory,name):
    scores=predict(model,frame,columns)
    pairs=frame[['user_id','item_id','label']].copy();pairs.item_id=pairs.item_id.astype('int64')
    metrics,per=evaluate_ranking(pairs,scores,truth,k=config['k'])
    top=pairs.assign(score=scores).sort_values(['user_id','score','item_id'],ascending=[True,False,True]).groupby('user_id',sort=False).head(config['k'])
    save_npz(directory/(name+'_top10.npz'),user_id=top.user_id.to_numpy(),item_id=top.item_id.to_numpy(),score=top.score.to_numpy())
    per.to_csv(directory/(name+'_per_user.csv'),index=False)
    return metrics,per


def run_case(training,validation,truth,columns,config,fold,seed,count,directory):
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'complete.json').exists():
        print('Restore completed case',fold,count,seed,flush=True);return read(directory/'complete.json')
    dates=[str(d.date()) for d in dates_for(fold,config)];full=[];fits=[];cohorts=[]
    for frame,date in zip(training,dates):
        ids=sampled_users(frame,count,seed,date);cohorts.append({'cutoff':date,'users':ids.tolist()})
        selected=select_users(frame,ids);selected['row_id']=np.arange(len(selected))
        full.append(selected);fits.append(negative_sample(selected,config['max_negatives'],seed,date))
    del selected,frame
    full[0]=None  # First week is used for fitting, never an OOF target.
    save(directory/'cohorts.json',cohorts)
    selected_path=directory/'selection.json'
    if selected_path.exists():selection=read(selected_path)
    else:
        oof_frames=[];contexts=[];audit=[];fit_times=[]
        for index in range(1,len(full)):
            target=full[index];label_end=check_oof_dates(dates[:index],dates[index]);oof_path=directory/f'oof_{index}.npz'
            if oof_path.exists() and (directory/f'oof_{index}.json').exists():
                with np.load(oof_path,allow_pickle=False) as record:
                    if not np.array_equal(record['user_id'],target.user_id) or not np.array_equal(record['item_id'],target.item_id.astype('int64')):raise ValueError('OOF checkpoint cohort mismatch')
                    scores=record['baseline_score'].copy();bins=record['baseline_bin'].copy()
                seconds=read(directory/f'oof_{index}.json')['fit_seconds']
            else:
                print('OOF',fold,count,seed,index,flush=True)
                train=pd.concat(fits[:index],ignore_index=True)
                model,seconds=fit_model(train,columns['manual'],columns['cats'],config,seed)
                scores=predict(model,target,columns['manual']);bins=baseline_bins(scores,target.query_id,config['baseline_bins'])
                model.save_model(str(directory/f'oof_{index}.cbm'))
                save_npz(oof_path,user_id=target.user_id.to_numpy(),item_id=target.item_id.astype('int64').to_numpy(),
                    label=target.label.to_numpy(dtype=np.uint8),baseline_score=scores,baseline_bin=bins.astype(np.uint8),
                    cutoff=np.asarray(dates[index]),train_label_end=np.asarray(label_end))
                save(directory/f'oof_{index}.json',{'fit_seconds':seconds,'train_label_end':label_end,'target_cutoff':dates[index],
                    'train_rows':len(train),'prediction_rows':len(target),'source_weeks':dates[:index]})
                del train,model;gc.collect()
            row=fits[index].copy();positions=row.row_id.to_numpy();row['oof_score']=scores[positions]
            oof_frames.append(row);contexts.append(bins[positions]);fit_times.append(seconds)
            audit.append(read(directory/f'oof_{index}.json'))
            full[index]=None  # Release completed target pool before next fit.
        full.clear();del target,row
        selector_frame=pd.concat(oof_frames,ignore_index=True);baseline_context=np.concatenate(contexts)
        bank,rejected=numeric_dedup(selector_frame,columns['manual'],columns['bank'])
        context,edges=history_context(selector_frame);extended=context.assign(baseline_rank_bin=baseline_context)
        print('Selecting',fold,count,seed,len(selector_frame),'rows',flush=True)
        started=perf_counter();cmim=CMIMSelector(config['selected_features'],config['feature_bins']).fit(selector_frame[bank],selector_frame.label,context)
        history_seconds=perf_counter()-started
        mi=sorted(bank,key=lambda c:(-cmim.marginal_relevance_[c],c))[:config['selected_features']]
        started=perf_counter();conditional=CMIMSelector(config['selected_features'],config['feature_bins'],
            max_context_columns=4,max_context_states=128).fit(selector_frame[bank],selector_frame.label,extended)
        conditional_seconds=perf_counter()-started
        sizes=extended.groupby(list(extended.columns),dropna=False).size()
        selection={'features':{'manual':[],'mi':mi,'cmim':cmim.selected_features_,'cmim_oof':conditional.selected_features_},
            'cmim':cmim.to_dict(),'cmim_oof':conditional.to_dict(),'popularity_edges':edges,'bank_kept':bank,'bank_rejected':rejected,
            'selector_rows':len(selector_frame),'train_rows':sum(map(len,fits)),'oof_audit':audit,
            'oof_fit_seconds':sum(fit_times),'cmim_seconds':history_seconds,'cmim_oof_seconds':conditional_seconds,
            'context_state_rows':{'states':len(sizes),'min':int(sizes.min()),'median':float(sizes.median()),'max':int(sizes.max())}}
        save(selected_path,selection);del selector_frame,oof_frames;gc.collect()
    full.clear()
    final_train=pd.concat(fits,ignore_index=True);fits.clear();gc.collect()
    metrics={};pers={};model_seconds={}
    for name,extra in selection['features'].items():
        cols=columns['manual']+extra
        if any(c.startswith('oof_') for c in cols):raise ValueError('OOF scores must not enter ranker')
        path=directory/(name+'.cbm');meta=directory/(name+'_fit.json')
        if path.exists():model=CatBoostRanker();model.load_model(str(path));seconds=read(meta)['fit_seconds'] if meta.exists() else None
        else:
            print('Ranker',fold,count,seed,name,flush=True)
            model,seconds=fit_model(final_train,cols,columns['cats'],config,seed)
            temp=path.with_suffix('.tmp');model.save_model(str(temp));temp.replace(path)
            save(meta,{'fit_seconds':seconds,'columns':cols,'train_rows':len(final_train)})
        met,per=evaluate_model(model,validation,cols,truth,config,directory,name)
        metrics[name]=met;pers[name]=per;model_seconds[name]=seconds
        del model;gc.collect()
    comparisons={name:paired_bootstrap(pers['manual'],per,metric='ndcg',n_boot=config['bootstrap'],seed=config['pool_seed']) for name,per in pers.items() if name!='manual'}
    comparisons['cmim_oof_minus_cmim']=paired_bootstrap(pers['cmim'],pers['cmim_oof'],metric='ndcg',n_boot=config['bootstrap'],seed=config['pool_seed'])
    result={'fold':fold,'train_users':count,'seed':seed,'metrics':metrics,'bootstrap':comparisons,'model_fit_seconds':model_seconds}
    save(directory/'complete.json',result);print('Case complete',fold,count,seed,flush=True)
    return result


def summarize(out,config):
    rows=[];selection_sets={};frequencies=Counter();costs=[]
    for fold,count,seed in itertools.product(config['validation_cutoffs'],config['train_users'],config['seeds']):
        directory=out/fold/f'n{count}_seed{seed}';record=read(directory/'complete.json');selection=read(directory/'selection.json')
        for name,metrics in record['metrics'].items():
            rows.append({'fold':fold,'train_users':count,'seed':seed,'method':name,**metrics,
                'ndcg_delta':metrics[f"ndcg@{config['k']}"]-record['metrics']['manual'][f"ndcg@{config['k']}"]})
            if name!='manual':
                features=selection['features'][name];selection_sets[(fold,count,seed,name)]=set(features)
                for feature in features:frequencies[(fold,count,name,feature)]+=1
        costs.append({'fold':fold,'train_users':count,'seed':seed,'oof_fit_seconds':selection['oof_fit_seconds'],
            'cmim_seconds':selection['cmim_seconds'],'cmim_oof_seconds':selection['cmim_oof_seconds'],**{name+'_fit_seconds':value for name,value in record['model_fit_seconds'].items()}})
    table=pd.DataFrame(rows);table.to_csv(out/'metrics.csv',index=False)
    metric=f"ndcg@{config['k']}";summary=table.groupby(['fold','train_users','method'])[metric].agg(['mean','std','min','max']).reset_index()
    summary.to_csv(out/'quality_summary.csv',index=False)
    stability=[]
    for method in ['mi','cmim','cmim_oof']:
        keys=[key for key in selection_sets if key[-1]==method]
        for first,second in itertools.combinations(keys,2):
            changed=[i for i in range(3) if first[i]!=second[i]]
            if len(changed)!=1:continue
            axis={0:'time',1:'size',2:'seed'}[changed[0]];a=selection_sets[first];b=selection_sets[second]
            wa={v.split('__age_')[-1] for v in a};wb={v.split('__age_')[-1] for v in b}
            stability.append({'axis':axis,'method':method,'fold_a':first[0],'fold_b':second[0],'users_a':first[1],'users_b':second[1],
                'seed_a':first[2],'seed_b':second[2],'feature_jaccard':len(a&b)/len(a|b) if a|b else 1.,
                'window_jaccard':len(wa&wb)/len(wa|wb) if wa|wb else 1.})
    pd.DataFrame(stability).to_csv(out/'stability.csv',index=False)
    pd.DataFrame([{'fold':f,'train_users':n,'method':m,'feature':x,'relation':x.split('__')[1],'selected_seeds':v}
        for (f,n,m,x),v in frequencies.items()]).to_csv(out/'selection_frequencies.csv',index=False)
    pd.DataFrame(costs).to_csv(out/'costs.csv',index=False)
    generate_report(out,config,table,summary,pd.DataFrame(stability))


def generate_report(out,config,table,summary,stability):
    lines=['# Устойчивость отбора временных признаков','',
        '**Диагностическая rolling validation. Новые методы по просмотренным test не выбирались.**','',
        f"Фолды: {config['validation_cutoffs']}; train-users на каждую из {config['train_snapshots']} недель: {config['train_users']}; seeds: {config['seeds']}. Retrieval фиксирован: popularity/genre + ItemKNN + EASE, 100 кандидатов.",'',
        '## Протокол','',
        'Для каждой diagnostic-недели используются четыре непосредственно предшествующие train-недели. Каждый OOF-baseline обучен только на ещё более ранних train-неделях; их label horizons заканчиваются не позже prediction cutoff. Первая train-неделя не имеет OOF-прогноза и исключается из всех selectors, но входит в обучение всех финальных ранкеров.',
        'MI, CMIM и CMIM+OOF используют одни и те же строки train и один банк. CMIM контролирует activity/genre-match/popularity; CMIM+OOF добавляет пять групп ранга baseline, вычисленного внутри полного запроса из 100 кандидатов. Исходный OOF-score и использованный bin сохраняются. Они не передаются финальному ранкеру: проверяется отбор, не stacking.',
        'Для CMIM+OOF разрешены четыре столбца контекста и до 128 совместных состояний (теоретически здесь до 90). Старые defaults CMIM остаются 3/32. Большее conditioning увеличивает риск разреженности histogram estimates; фактические размеры страт сохранены в selection.json.',
        f"Внутри seed {min(config['train_users'])} пользователей и их sampled pairs вложены в {max(config['train_users'])}. Обе когорты меняются между seeds, выбираясь из фиксированного пула {config['train_pool_users']} на snapshot. Это feature pool, не число пользователей обучения: размер каждой train-недели ограничен {max(config['train_users'])}. Полная историческая таблица для retrieval/признаков не сэмплируется. Validation-пользователи фиксированы внутри фолда для всех вариантов.",
        f"Ранкер фиксирован: {config['loss']}, {config['ranker']}, one-hot для категорий. Нет повторного подбора и early stopping по диагностическим результатам. Это другой вычислительный протокол, чем старые CTR/180-candidate прогоны: абсолютные метрики напрямую не сравниваются.",
        'Retrieval и его параметры зафиксированы по предшествующему исследованию, без нового выбора. Июльские недели — development-диагностика; они уже могут встречаться как train/search в прежних исследованиях. Это не независимый подтверждающий test. Фолды перекрываются и статистически зависимы.',
        'KION содержит последние взаимодействия пар. Отсутствие пары в истории не доказывает отсутствие более ранних показов/просмотров. Negatives — implicit нули заданного горизонта. Оцениваются warm и future-active пользователи, не вся аудитория сервиса.','',
        '## Качество: NDCG@10 по seeds','', '| Неделя | Train users | Метод | Mean | Std | Min | Max |','|---|---:|---|---:|---:|---:|---:|']
    for _,r in summary.iterrows():lines.append(f"| {r.fold} | {r.train_users} | {r.method} | {r['mean']:.6f} | {r['std']:.6f} | {r['min']:.6f} | {r['max']:.6f} |")
    lines += ['', '## Парные разности к manual','', '| Неделя | Users | Метод | Средняя Δ NDCG | Min Δ | Max Δ |','|---|---:|---|---:|---:|---:|']
    deltas=table.loc[table.method!='manual'].groupby(['fold','train_users','method']).ndcg_delta.agg(['mean','min','max']).reset_index()
    for _,r in deltas.iterrows():lines.append(f"| {r.fold} | {r.train_users} | {r.method} | {r['mean']:+.6f} | {r['min']:+.6f} | {r['max']:+.6f} |")
    lines += ['', '## Устойчивость признаков и окон','',
        'Feature Jaccard сравнивает точные спецификации. Window Jaccard объединяет признаки с одинаковыми временными границами и half-life независимо от отношения и агрегации. Это разные характеристики.', '',
        '| Ось | Метод | Feature Jaccard | Window Jaccard |','|---|---|---:|---:|']
    for _,r in stability.groupby(['axis','method'])[['feature_jaccard','window_jaccard']].mean().reset_index().iterrows():
        lines.append(f'| {r.axis} | {r.method} | {r.feature_jaccard:.3f} | {r.window_jaccard:.3f} |')
    lines += ['', '## Ограничения выводов','',
        'Три seeds дают описательный разброс, не точное распределение качества. Они одновременно меняют train-когорты, negative sampling и стохастическое обучение. Bootstrap в каждом case условен на фиксированной модели и фолде: он не заменяет разброс по seeds, не учитывает множественные сравнения и выбор гипотез.',
        'Изменение набора при стабильном NDCG не обязательно означает проблему: коррелированные временные признаки могут заменять друг друга. Ранжирование внутри query и построчная MI — разные цели. Попарный CMIM не равен полной совместной CMI или Information Bottleneck.',
        'На этих результатах не выбирается новый победитель для test. Для подтверждающего вывода потребуется отдельный заранее отложенный период/датасет.','',
        '## Воспроизведение','', '```bash','python -m pip install -e ".[dev,retrieval]"','python scripts/download_data.py','OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=6 python -m kion_recsys.selection_stability --config configs/selection_stability.json','PYTHONPATH=src python scripts/verify_selection_stability.py','```', '',
        'Каждый snapshot, OOF-прогноз и финальная модель имеют checkpoints. Повтор команды продолжает тот же протокол. Для изменённого config нужен новый output_dir. Cache не входит в поставку; не загружайте чужие pickle caches. В итоговой поставке есть модели, OOF scores, top-10 predictions, per-user метрики и аудит временных границ.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def execute(config):
    validate_protocol(config);out=Path(config['output_dir']);out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists() and read(out/'config.json')!=config:raise ValueError('Protocol changed: use a different output directory')
    save(out/'config.json',config)
    if not (out/'protocol_before_results.json').exists():
        save(out/'protocol_before_results.json',{'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'config':config,
            'primary_comparison':'cmim_oof minus cmim','selection_from_diagnostic_metrics':False})
    normalized=out/'cache'/'inputs.pkl'
    if normalized.exists():events,items,users=pd.read_pickle(normalized)
    else:
        print('Loading KION',flush=True);events,items,users=load_data(Path(config['data_dir']))
        events=events[['user_id','item_id','timestamp']].copy();gc.collect()
        normalized.parent.mkdir(parents=True,exist_ok=True);pd.to_pickle((events,items,users),normalized)
    dates=sorted({date for fold in config['validation_cutoffs'] for date in [*dates_for(fold,config),pd.Timestamp(fold)]})
    for date in dates:
        record=candidate_snapshot(events,items,users,date,config,out/'cache'/'features')
        save(out/'snapshots'/f'{date.date()}.json',record['info']);del record;gc.collect()
    del events,items,users;gc.collect()
    for fold in config['validation_cutoffs']:
        train_paths=[out/'cache'/'features'/f'{date.date()}.pkl' for date in dates_for(fold,config)]
        columns=pd.read_pickle(train_paths[0])['columns']
        gc.collect()
        validation_record=pd.read_pickle(out/'cache'/'features'/f'{fold}.pkl');pool=validation_record['frame']
        ids=sampled_users(pool,config['validation_users'],config['validation_seed'],fold)
        validation=select_users(pool,ids);truth={int(u):validation_record['truth'][int(u)] for u in ids}
        directory=out/fold;directory.mkdir(exist_ok=True)
        save(directory/'truth.json',{str(u):sorted(v) for u,v in truth.items()})
        save_npz(directory/'candidates.npz',user_id=validation.user_id.to_numpy(),item_id=validation.item_id.astype('int64').to_numpy(),label=validation.label.to_numpy(dtype=np.uint8))
        del pool,validation_record;gc.collect()
        for count,seed in itertools.product(config['train_users'],config['seeds']):
            training=(pd.read_pickle(path)['frame'] for path in train_paths)
            run_case(training,validation,truth,columns,config,fold,seed,count,directory/f'n{count}_seed{seed}')
        del validation;gc.collect()
    summarize(out,config)
    from .stability_analysis import enrich
    enrich(out,config)
    save(out/'complete.json',{'status':'complete','folds':config['validation_cutoffs'],'no_test_evaluation':True})


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args();execute(read(args.config))

if __name__=='__main__':main()
