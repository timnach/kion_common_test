"""Describe fixed contrasts on the common test without choosing a new winner."""
from pathlib import Path
import json
import importlib.metadata
import platform
import pandas as pd


def enrich(c,out):
    out=Path(out);read=lambda p:json.loads(Path(p).read_text())
    summary=pd.read_csv(out/'summary.csv').set_index('method')
    contrasts=pd.read_csv(out/'paired_contrasts.csv');costs=[]
    for seed in c['seeds']:
        directory=out/'validation'/f'seed{seed}';selection=read(directory/'selection.json')
        fits=[read(p) for p in directory.glob('*_fit.json')]
        costs.append({'seed':seed,'unique_validation_model_fits':len(fits),
            'validation_model_fit_seconds':sum(r['fit_seconds'] for r in fits),
            'forward_oof_fit_seconds':selection['oof_fit_seconds'],
            'cmim_selection_seconds':selection['cmim_seconds'],
            'cmim_oof_selection_seconds':selection['cmim_oof_seconds']})
    pd.DataFrame(costs).to_csv(out/'search_costs.csv',index=False)
    environment={'python':platform.python_version(),'platform':platform.platform(),
        'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','catboost','implicit','pytest']},
        'cost_scope':'Wall-clock fit times; exclude prediction/evaluation, feature build and data loading unless separately recorded. OOF and selectors shown separately.'}
    (out/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    report=out/'REPORT.md';text=report.read_text().split('\n## Анализ общего test')[0]
    text+='\n## Анализ общего test\n\n'
    for name,label in [('guided_search','Guided − manual (основное сравнение)'),('guided_minus_random','Guided − random'),('oof_minus_cmim','CMIM+OOF − CMIM'),('constant_minus_uniform','Constant decay − uniform')]:
        g=contrasts.loc[contrasts.comparison==name]
        text+=f"- {label}: mean Δ NDCG {g.delta.mean():+.6f}; положительных seeds {int((g.delta>0).sum())}/{len(g)}; CI целиком выше нуля {int((g.ci_low>0).sum())}, ниже нуля {int((g.ci_high<0).sum())}.\n"
    text+='\nЧастоты по трём seeds не являются вероятностью выигрыша или самостоятельным тестом значимости. Нет поправки на множественные проверки. Средние — усреднение метрик трёх отдельных ранкеров, не качество ансамбля их scores.\n\n'
    text+='### Качество заранее выбранного validation winner\n\n| Seed | Выбран на validation | Test NDCG@10 | Test Δ к manual |\n|---|---|---:|---:|\n'
    metrics=pd.read_csv(out/'metrics.csv')
    for seed in c['seeds']:
        d=read(out/'validation'/f'seed{seed}'/'decisions.json');r=metrics.loc[(metrics.seed==seed)&(metrics.method==d['validation_winner'])].iloc[0]
        text+=f"| {seed} | {d['validation_winner']} | {r['ndcg@10']:.6f} | {r.delta_manual:+.6f} |\n"
    text+='\nЭта таблица оценивает решение, принятое до test. Она не заменяется лучшим наблюдаемым test-вариантом.\n\n'
    text+='### Оценка заранее заданных гипотез\n\n'
    primary=contrasts.loc[contrasts.comparison=='guided_search']
    if (primary.ci_low>0).all():
        text+='Guided превосходит manual с положительными условными интервалами во всех трёх seeds. Это локальный результат на ранее просмотренной неделе; независимого подтверждения на новой популяции/периоде он не даёт.\n\n'
    else:
        text+='Основное сравнение guided с manual следует читать вместе с per-seed интервалами: устойчивого улучшения с интервалами выше нуля во всех трёх seeds не получено. Это не доказывает равенство методов.\n\n'
    text+='Разности all-bank/MI/CMIM оценивают добавление или отбор реляционных агрегатов при общей информации и одном candidate pool. Разность CMIM+OOF и CMIM отделяет дополнительный контроль baseline. Constant против uniform отделяет время от добавления трёх агрегатов; alpha/beta/adaptive выбирают нулевой вариант, если он лучше на validation. Multiscale добавляет 15 входов и не является автоматически обученной смесью.\n\n'
    text+='Все эти выводы относятся к общему test и текущему ограниченному tuning. Более высокий test-score не используется для назначения нового финального победителя.\n\n'
    text+='### Вычислительная стоимость\n\n'
    text+='Полные времена уникальных validation fits, OOF и selectors находятся в search_costs.csv; selected-model fit и inference+evaluation — в costs.csv. Эти компоненты нельзя суммировать как чистое serving latency. При одинаковом числе random/guided trials guided дополнительно использует MI/CMIM. Затраты других семейств не выровнены до общего AutoML-бюджета.\n'
    report.write_text(text)
