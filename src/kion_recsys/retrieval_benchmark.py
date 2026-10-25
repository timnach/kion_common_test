"""Standalone retrieval study. Original rankers/results are never overwritten.

python -m kion_recsys.retrieval_benchmark --config configs/retrieval_benchmark.json
"""
import argparse
import gc
import gzip
import os
import hashlib
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from .data import load_data
from .collaborative import CollaborativeRetriever
from .retrieval import SnapshotRetriever
from .retrieval_models import BinarySnapshot,MatrixRetriever
from .metrics import paired_bootstrap


def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
    temp.replace(path)


def write_candidate_file(path,frame):
    # Compress fully before the atomic write; verify byte count before rename.
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    payload=gzip.compress(frame.to_csv(index=False).encode('utf-8'),mtime=0)
    with temporary.open('wb') as stream:
        if stream.write(payload)!=len(payload):raise IOError('Short candidate export write')
        stream.flush();os.fsync(stream.fileno())
    if temporary.stat().st_size!=len(payload):raise IOError('Candidate export size mismatch')
    temporary.replace(path)


def read(path): return json.loads(Path(path).read_text())


def cohorts(config):
    result={}
    for fold in config['folds']:
        snapshots=read(Path(fold['original_run'])/'snapshots.json')
        phases={x['phase']:x for x in snapshots}
        for phase in ['validation','evaluation']:
            row=phases[phase];date=row['cutoff'];ids=list(map(int,row['selected_users']))
            result.setdefault(date,set()).update(ids)
    return {date:sorted(ids) for date,ids in result.items()}


def truth_for(events,date,user_ids):
    date=pd.Timestamp(date)
    future=events.loc[(events.timestamp>=date)&(events.timestamp<date+pd.Timedelta(days=7))&events.user_id.isin(user_ids)]
    by_user={int(u):set(map(int,g.item_id)) for u,g in future.groupby('user_id',sort=False)}
    if set(by_user)!=set(user_ids): raise ValueError('Archived cohort does not match active target users')
    return by_user


def complete_candidates(sources,names,legacy,user_ids,k=100):
    """RRF at 60, shared past-popularity fill; no target labels are consulted."""
    pieces=[]
    for name in names:
        f=sources[name].loc[sources[name].user_id.isin(user_ids)].sort_values(['user_id','score','item_id'],ascending=[True,False,True]).copy()
        f['value']=1./(60+f.groupby('user_id').cumcount()+1)
        pieces.append(f[['user_id','item_id','value']])
    combined=pd.concat(pieces).groupby(['user_id','item_id'],as_index=False).value.sum()
    combined=combined.sort_values(['user_id','value','item_id'],ascending=[True,False,True]).groupby('user_id',sort=False).head(k)
    grouped={int(u):g for u,g in combined.groupby('user_id',sort=False)}
    previous={int(u):set(map(int,g.item_id)) for u,g in legacy.history.loc[legacy.history.user_id.isin(user_ids)].groupby('user_id',sort=False)}
    popularity=legacy._top(legacy.item_counts,len(legacy.item_counts))
    rows=[]
    for uid in sorted(map(int,user_ids)):
        group=grouped.get(uid)
        existing=[] if group is None else list(zip(group.item_id,group.value))
        chosen={int(i) for i,_ in existing};rank=0
        for item,value in existing:
            rank+=1;rows.append((uid,int(item),float(value),rank,0))
        for item in popularity:
            if rank>=k: break
            if item not in chosen and item not in previous.get(uid,set()):
                chosen.add(item);rank+=1;rows.append((uid,item,0.,rank,1))
    return pd.DataFrame(rows,columns=['user_id','item_id','score','rank','fallback'])


def evaluate(frame,truth):
    grouped={int(u):set(map(int,g.item_id)) for u,g in frame.groupby('user_id',sort=False)}
    rows=[]
    for uid,targets in truth.items():
        ids=grouped.get(int(uid),set());hits=len(ids&targets)
        rows.append((uid,hits/len(targets),hits,len(targets),len(ids)))
    per=pd.DataFrame(rows,columns=['user_id','candidate_recall','hits','n_truth','n_candidates'])
    fallback=float(frame.fallback.mean()) if len(frame) and 'fallback' in frame else 0.
    summary={'candidate_recall':float(per.candidate_recall.mean()),
             'micro_recall':float(per.hits.sum()/per.n_truth.sum()),
             'mean_candidates':float(per.n_candidates.mean()),'min_candidates':int(per.n_candidates.min()),
             'max_candidates':int(per.n_candidates.max()),'no_hit_fraction':float((per.hits==0).mean()),
             'fallback_fraction':fallback,'users':len(per)}
    return summary,per


def build_sources(events,items,date,user_ids,config,cache):
    path=cache/(date+'.pkl')
    fingerprint=hashlib.sha256(json.dumps({'date':date,'users':user_ids,'config':config},sort_keys=True).encode()).hexdigest()
    if path.exists():
        result=pd.read_pickle(path)
        if result['fingerprint']!=fingerprint: raise ValueError('Cache configuration mismatch')
        print('Restored candidates',date,flush=True);return result
    started=perf_counter();legacy=SnapshotRetriever(events,items,date,**config['legacy'])
    native=legacy.candidates_for(user_ids).sort_values(['user_id','retrieval_score','item_id'],ascending=[True,False,True]).groupby('user_id',sort=False).head(config['candidate_k'])
    sources={'legacy':native.rename(columns={'retrieval_score':'score'})[['user_id','item_id','score']]}
    info={}
    print('Fit ItemKNN',date,flush=True)
    start=perf_counter();knn=CollaborativeRetriever(events,items,date,**config['knn'])
    fit=perf_counter()-start;start=perf_counter()
    sources['knn']=knn.candidates_for(user_ids,k=config['candidate_k']).rename(columns={'knn_score':'score'})[['user_id','item_id','score']]
    info['knn']={'fit_seconds':fit,'score_seconds':perf_counter()-start,'catalog_items':knn.n_items}
    del knn;gc.collect()
    start=perf_counter();matrix=BinarySnapshot(events,items,date);matrix_seconds=perf_counter()-start
    vocabularies={}
    for method in ['als','ease']:
        for index,params in enumerate(config[method+'_grid']):
            tag=f'{method}_{index}'
            print('Fit',date,tag,params,flush=True)
            model=MatrixRetriever(matrix,method,threads=config['threads'],seed=config['seed'],**params)
            start=perf_counter();sources[tag]=model.candidates_for(user_ids,k=config['candidate_k'])
            info[tag]={'fit_seconds':model.fit_seconds,'score_seconds':perf_counter()-start,
                       'catalog_items':len(model.item_ids),'history_interaction_coverage':model.interaction_coverage,'params':params}
            vocabularies[tag]=model.item_ids.tolist()
            print('Ready',date,tag,round(model.fit_seconds,2),'s fit',flush=True)
            del model;gc.collect()
    # Keep only small data for fusion/fallback, never cached weights or raw history.
    fallback={int(u):set(map(int,g.item_id)) for u,g in legacy.history.loc[legacy.history.user_id.isin(user_ids)].groupby('user_id',sort=False)}
    result={'fingerprint':fingerprint,'sources':sources,'timings':info,'vocabularies':vocabularies,
            'matrix_seconds':matrix_seconds,'seconds':perf_counter()-started,
            'catalog':matrix.item_ids.tolist(),'popularity':legacy._top(legacy.item_counts,len(legacy.item_counts)),
            'seen':fallback}
    cache.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');pd.to_pickle(result,temp);temp.replace(path)
    return result


class FusionSnapshot:
    """Minimal adapter reusing cached past-only candidate/fallback state."""
    def __init__(self,record):
        self.record=record
        self.item_counts=pd.Series({i:len(record['popularity'])-r for r,i in enumerate(record['popularity'])})
        self.history=pd.DataFrame([(u,i) for u,seen in record['seen'].items() for i in seen],columns=['user_id','item_id'])
    @staticmethod
    def _top(counts,k):return SnapshotRetriever._top(counts,k)


def methods(record,users,config,als_tag,ease_tag):
    source=record['sources'];legacy=FusionSnapshot(record);k=config['candidate_k']
    combinations={'legacy':['legacy'],'knn':['knn'],'als':[als_tag],'ease':[ease_tag],
        'legacy+knn':['legacy','knn'],'legacy+knn+als':['legacy','knn',als_tag],
        'legacy+knn+ease':['legacy','knn',ease_tag],'legacy+knn+als+ease':['legacy','knn',als_tag,ease_tag]}
    return {name:complete_candidates(source,names,legacy,users,k) for name,names in combinations.items()}


def execute(config):
    out=Path(config['output_dir']);out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists() and read(out/'config.json')!=config:raise ValueError('Use a new output directory for changed protocol')
    save(out/'config.json',config)
    normalized=out/'cache'/'inputs.pkl'
    if normalized.exists(): events,items=pd.read_pickle(normalized)
    else:
        print('Loading validated KION',flush=True)
        events,items,users=load_data(Path(config['data_dir']))
        events=events[['user_id','item_id','timestamp']].copy();items=items[['item_id','genres']].copy();del users;gc.collect()
        normalized.parent.mkdir(parents=True,exist_ok=True);pd.to_pickle((events,items),normalized)
    all_users=cohorts(config);records={}
    for date,ids in sorted(all_users.items()):
        if pd.Timestamp(date)+pd.Timedelta(days=7)>events.timestamp.max()+pd.Timedelta(days=1):raise ValueError('Incomplete target horizon')
        records[date]=build_sources(events,items,date,ids,config,out/'cache')
    decisions={}
    for fold in config['folds']:
        phases={x['phase']:x for x in read(Path(fold['original_run'])/'snapshots.json')}
        validation=phases['validation'];date=validation['cutoff'];users=list(map(int,validation['selected_users']))
        record=records[date];truth=truth_for(events,date,users);adapter=FusionSnapshot(record)
        tuning=[];best={}
        for name in ['als','ease']:
            for index in range(len(config[name+'_grid'])):
                tag=f'{name}_{index}';frame=complete_candidates(record['sources'],[tag],adapter,users,config['candidate_k'])
                met,_=evaluate(frame,truth);tuning.append({'source':tag,**met})
            best[name]=max([r for r in tuning if r['source'].startswith(name+'_')],key=lambda r:r['candidate_recall'])['source']
        tables=methods(record,users,config,best['als'],best['ease'])
        metrics={name:evaluate(frame,truth)[0] for name,frame in tables.items()}
        unions=[name for name in metrics if '+' in name]
        chosen=max(unions,key=lambda name:metrics[name]['candidate_recall'])
        evaluation_date=phases['evaluation']['cutoff']
        decisions[evaluation_date]={'validation_cutoff':date,'evaluation_cutoff':evaluation_date,
            'als_tag':best['als'],'ease_tag':best['ease'],'chosen_union':chosen,
            'tuning':tuning,'validation_metrics':metrics,'original_run':fold['original_run']}
    save(out/'decisions_before_evaluation.json',{'saved_at_utc':datetime.now(timezone.utc).isoformat(),'folds':decisions,
        'note':'Rolling folds overlap; earlier evaluation week is later validation week. All comparisons predeclared, exploratory.'})
    print('Frozen decisions',json.dumps({d:r['chosen_union'] for d,r in decisions.items()}),flush=True)
    rows=[]
    for date,decision in decisions.items():
        original=Path(decision['original_run']);phases={x['phase']:x for x in read(original/'snapshots.json')}
        users=list(map(int,phases['evaluation']['selected_users']));truth=truth_for(events,date,users)
        expected={int(u):set(v) for u,v in read(original/'test_truth.json').items()}
        if truth!=expected: raise ValueError('Evaluation truth differs from archived original experiment')
        record=records[date];tables=methods(record,users,config,decision['als_tag'],decision['ease_tag'])
        directory=out/date;directory.mkdir(exist_ok=True);pers={};metrics={}
        for name,frame in tables.items():
            met,per=evaluate(frame,truth);metrics[name]=met;pers[name]=per
            per.to_csv(directory/('per_user_'+name+'.csv'),index=False)
            write_candidate_file(directory/('candidates_'+name+'.csv.gz'),frame)
            rows.append({'cutoff':date,'method':name,**met})
        raw_metrics={}
        for name,tag in [('legacy','legacy'),('knn','knn'),('als',decision['als_tag']),('ease',decision['ease_tag'])]:
            raw=record['sources'][tag].loc[record['sources'][tag].user_id.isin(users)].sort_values(['user_id','score','item_id'],ascending=[True,False,True]).groupby('user_id',sort=False).head(config['candidate_k'])
            raw_metrics[name]=evaluate(raw,truth)[0]
        comparisons={name:paired_bootstrap(pers['legacy+knn'],per,metric='candidate_recall',n_boot=config['bootstrap'],seed=config['seed'])
                     for name,per in pers.items() if name!='legacy+knn'}
        full=set(record['catalog']);vocab=set(record['vocabularies'][decision['ease_tag']]);n=sum(map(len,truth.values()))
        coverage={'catalog_positive_coverage':sum(len(t&full) for t in truth.values())/n,
                  'ease_positive_coverage':sum(len(t&vocab) for t in truth.values())/n,
                  'n_positives':n,'ease_history_coverage':record['timings'][decision['ease_tag']]['history_interaction_coverage']}
        save(directory/'results.json',{'metrics':metrics,'raw_metrics':raw_metrics,'bootstrap_vs_legacy_knn':comparisons,
             'coverage':coverage,'timings':record['timings'],'shared_matrix_seconds':record['matrix_seconds']})
        print('Evaluation',date,{name:round(m['candidate_recall'],6) for name,m in metrics.items()},flush=True)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','implicit']}
    save(out/'environment.json',versions)
    write_report(out,config,decisions)
    save(out/'complete.json',{'status':'complete','evaluation_cutoffs':list(decisions)})


def write_report(out,config,decisions):
    lines=['# Усиление retrieval: ItemKNN, ALS и EASE','',
        'Основная метрика — macro candidate recall: средняя по пользователям доля будущих positives, вошедших в кандидатный пул. Все методы получают ровно 100 кандидатов, если допустимый каталог позволяет. Это не MAP@10 и не NDCG@10.','',
        '## Протокол','',
        'Те же validation/evaluation пользователи и positives, что в исходных research_v2-запусках. История всех пользователей берётся целиком до cutoff. Наблюдаемые ранее пары исключены из кандидатов. Параметры ALS/EASE и состав union выбираются на предшествующей validation-неделе; evaluation не используется для выбора. RRF имеет фиксированную константу 60 и равные веса.',
        'Короткие списки дополняются общей прошлой популярностью, с исключением истории пользователя. Отдельно показаны сырые списки и доля дополнения. Метаданные будущих объектов не устанавливают доступность: каталог состоит из объектов с прошлым взаимодействием.',
        'ALS: binary implicit feedback, observed confidence alpha=20, 15 итераций, factors 32/64, regularization 0.1. EASE: regularization 500/1000, словарь top-4000 по прошлой популярности. EASE на полном каталоге здесь не проверялся.',
        'Эти календарные периоды уже исследовались раньше: результат exploratory. Фолды перекрываются (ранняя evaluation-неделя — поздняя validation), поэтому не являются независимыми. Когорта условна на прошлой активности и наличии будущего события. KION хранит последние взаимодействия пар, не полный журнал показов. Implicit нули не доказывают отказ от объекта.','']
    for date,decision in decisions.items():
        r=read(out/date/'results.json');base=r['metrics']['legacy+knn']['candidate_recall'];chosen=decision['chosen_union']
        ci=r['bootstrap_vs_legacy_knn'].get(chosen,{'delta':0,'ci_low':0,'ci_high':0})
        als=config['als_grid'][int(decision['als_tag'].split('_')[1])];ease=config['ease_grid'][int(decision['ease_tag'].split('_')[1])]
        lines += [f'## Evaluation {date}','',f"Validation {decision['validation_cutoff']}: ALS {als}; EASE {ease}; выбран **{chosen}**.",'',
            '| Метод | Candidate recall | Δ к legacy+knn, п.п. | Среднее кандидатов | Fallback |','|---|---:|---:|---:|---:|']
        for name,met in r['metrics'].items():lines.append(f"| {name} | {met['candidate_recall']:.5f} | {(met['candidate_recall']-base)*100:+.3f} | {met['mean_candidates']:.1f} | {met['fallback_fraction']:.3%} |")
        lines += ['',f"Выбранный заранее union минус legacy+knn: **{100*ci['delta']:+.3f} п.п.**, парный 95% bootstrap CI **[{100*ci['ci_low']:+.3f}; {100*ci['ci_high']:+.3f}] п.п.** Интервал условен на дату и выбранную конфигурацию, не учитывает перебор вариантов и множественные сравнения.",'',
            '| Источник без fallback | Recall | Среднее кандидатов |','|---|---:|---:|']
        for name,met in r['raw_metrics'].items():lines.append(f"| {name} | {met['candidate_recall']:.5f} | {met['mean_candidates']:.1f} |")
        c=r['coverage'];lines += ['',f"Прошлый каталог покрывает {c['catalog_positive_coverage']:.2%} будущих positives; словарь EASE — {c['ease_positive_coverage']:.2%}, при покрытии {c['ease_history_coverage']:.2%} прошлых взаимодействий.",'',
            '| Модель | Fit, с | Scoring всей рабочей когорты, с | Объектов |','|---|---:|---:|---:|']
        for tag in ['knn',decision['als_tag'],decision['ease_tag']]:
            t=r['timings'][tag];lines.append(f"| {tag} | {t['fit_seconds']:.2f} | {t['score_seconds']:.2f} | {t['catalog_items']} |")
        lines += ['',f"Общее построение binary matrix для ALS/EASE: {r['shared_matrix_seconds']:.2f} с. Scoring мог охватывать объединение validation/evaluation когорт; времена описательные, не строгий speed benchmark.",'']
    lines += ['## Итоговые выводы','']
    studies={date:read(out/date/'results.json') for date in decisions}
    singles=['legacy','knn','als','ease']
    if all(max(singles,key=lambda n:r['metrics'][n]['candidate_recall'])=='ease' for r in studies.values()):
        lines += ['**EASE — сильнейший одиночный источник на обоих срезах среди проверенных конфигураций.** При этом смесь с прежними источниками лучше одиночного EASE.','']
    for date,r in studies.items():
        base=r['metrics']['legacy+knn']['candidate_recall'];ease=r['metrics']['legacy+knn+ease']['candidate_recall']
        ci=r['bootstrap_vs_legacy_knn']['legacy+knn+ease']
        extra=r['metrics']['legacy+knn+als+ease']['candidate_recall']-ease
        lines += [f"- {date}: добавление EASE к legacy+knn меняет recall **{base:.2%} → {ease:.2%}**, прирост **{100*(ease-base):+.3f} п.п.**, 95% CI [{100*ci['ci_low']:+.3f}; {100*ci['ci_high']:+.3f}] п.п. Добавление ALS поверх этой смеси даёт {100*extra:+.3f} п.п."]
    lines += ['', 'По этим exploratory результатам EASE — наиболее убедительное дополнительное направление. ALS слабее как одиночный источник и не даёт устойчивого преимущества поверх смеси с EASE: его включение следует выбирать на validation, а не добавлять автоматически. Наблюдаемая смена знака — описание этих двух недель, не универсальное свойство ALS.',
        'Заранее выбранные на validation смеси сохранены в решениях каждого фолда; они не заменены лучшими постфактум evaluation-вариантами. Доверительные интервалы не учитывают выбор гипотез и не доказывают перенос на другие периоды.',
        'Прежние candidate recall около 53% считались при среднем пуле около 167 объектов. Их нельзя напрямую сравнивать с новым фиксированным бюджетом 100. Здесь вывод об улучшении основан на общем бюджете и одинаковой когорте.', '']
    lines += ['## Интерпретация','', 
        'Нужно разделять лучший одиночный источник и полезность источника в смеси: более слабый самостоятельно retriever может приносить дополнительные positives. При одинаковом лимите добавление источника может вытеснять полезные объекты — union не гарантирует прирост.',
        'Прирост candidate recall означает, что ранкер получает больше релевантных объектов. Он не доказывает прирост финального NDCG/MAP или качество negatives: для этого нужен отдельный контролируемый эксперимент с ранкером.','',
        '## Воспроизведение','', '```bash','python -m pip install -e ".[dev,retrieval]"','python scripts/download_data.py','OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=6 python -m kion_recsys.retrieval_benchmark --config configs/retrieval_benchmark.json','python scripts/verify_retrieval_benchmark.py','```','',
        'Повтор команды восстанавливает локальные candidate caches; изменённый конфиг требует нового output_dir. Cache не входит в архив и должен создаваться только доверенным локальным кодом. В папках evaluation сохранены списки кандидатов, per-user метрики, bootstrap, параметры и стоимость. Исходные ранкеры и отчёты не переписаны.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();execute(read(a.config))

if __name__=='__main__':main()
