"""Descriptive analysis only; never selects a model for any test period."""
import itertools
from pathlib import Path
import json
import pandas as pd


def enrich(out,config):
    out=Path(out);table=pd.read_csv(out/'metrics.csv');stability=pd.read_csv(out/'stability.csv')
    comparisons=[]
    for fold,count,seed in itertools.product(config['validation_cutoffs'],config['train_users'],config['seeds']):
        folder=out/fold/f'n{count}_seed{seed}';result=json.loads((folder/'complete.json').read_text());s=json.loads((folder/'selection.json').read_text())
        ci=result['bootstrap']['cmim_oof_minus_cmim'];a=set(s['features']['cmim']);b=set(s['features']['cmim_oof'])
        comparisons.append({'fold':fold,'train_users':count,'seed':seed,'ndcg_delta':ci['delta'],
            'ci_low':ci['ci_low'],'ci_high':ci['ci_high'],'feature_jaccard_between_methods':len(a&b)/len(a|b) if a|b else 1.,
            'oof_context_states':s['context_state_rows']['states'],'min_rows_per_state':s['context_state_rows']['min'],
            'median_rows_per_state':s['context_state_rows']['median'],'selector_rows':s['selector_rows']})
    comparison=pd.DataFrame(comparisons);comparison.to_csv(out/'primary_comparison.csv',index=False)
    grouped=comparison.groupby(['fold','train_users']).ndcg_delta.agg(['mean','std','min','max']).reset_index()
    by_size=stability.loc[stability.axis.isin(['seed','time'])].groupby(['axis','users_a','method'])[['feature_jaccard','window_jaccard']].mean().reset_index()
    by_size.to_csv(out/'stability_by_train_size.csv',index=False)
    frequency=pd.read_csv(out/'selection_frequencies.csv')
    relation=frequency.groupby(['train_users','method','relation']).selected_seeds.sum().reset_index();relation.to_csv(out/'relation_frequency.csv',index=False)
    report=out/'REPORT.md';text=report.read_text().split('\n## Основное сравнение')[0]
    text+='\n## Основное сравнение: контроль baseline в CMIM\n\n'
    text+='Здесь сравниваются одинаковые CMIM с историческим контекстом; второй дополнительно контролирует OOF-ранг baseline. Положительная Δ означает пользу дополнительного контроля.\n\n'
    text+='| Неделя | Users | Средняя Δ NDCG | Std по seeds | Min Δ | Max Δ |\n|---|---:|---:|---:|---:|---:|\n'
    for _,r in grouped.iterrows():text+=f"| {r.fold} | {r.train_users} | {r['mean']:+.6f} | {r['std']:.6f} | {r['min']:+.6f} | {r['max']:+.6f} |\n"
    text+='\nПолные per-seed интервалы находятся в primary_comparison.csv. Это user-bootstrap при фиксированном seed, не доверительный интервал среднего по seeds.\n\n'
    text+='## Устойчивость отдельно по размеру обучения\n\n| Ось | Train users | Метод | Feature Jaccard | Window Jaccard |\n|---|---:|---|---:|---:|\n'
    for _,r in by_size.iterrows():text+=f'| {r.axis} | {r.users_a} | {r.method} | {r.feature_jaccard:.3f} | {r.window_jaccard:.3f} |\n'
    text+=f"\nПул фиксирован на {config['train_pool_users']} пользователей: при {max(config['train_users'])} разные train-когорты сильнее пересекаются, чем при {min(config['train_users'])}. Это оценка устойчивости при заданном finite-pool sampling, не доказательство эффекта расширения всей популяции.\n"
    text+='\nВ ранкере группы без единого positive кандидата исключаются как неинформативные для ранжирования. Selector при этом оценивается на всех sampled train-парах OOF-доступных недель. Поэтому построчная MI и качество порядка внутри пользователя остаются разными целями.\n'
    protocol=json.loads((out/'protocol_before_results.json').read_text())
    if protocol.get('amended_after_partial_results'):
        text+='\n## Уточнение объёма по запросу пользователя\n\nПосле части запусков пользователь установил максимум 2000 train-пользователей на snapshot. Поэтому итоговая серия сравнивает 700/2000. Запуски на 5000 исключены и сохранены отдельно; готовые результаты и OOF на 2000 переиспользованы без изменения данных и параметров. Исходный протокол сохранён. Новая группа 700 задана до её результатов; это не полностью проспективная регистрация всей серии.\n'
    text+='\n## Среднее качество по всем диагностическим повторам\n\n'
    means=table.groupby(['method','train_users'])['ndcg@10'].mean().unstack('train_users')
    sizes=sorted(config['train_users'])
    text+='| Метод | '+ ' | '.join(f'NDCG@10, users={n}' for n in sizes)+' |\n|---|'+ '|'.join(['---:']*len(sizes))+'|\n'
    for method,row in means.iterrows():text+='| '+method+' | '+' | '.join(f'{row[n]:.6f}' for n in sizes)+' |\n'
    text+='\nЭто описательное среднее по трём перекрывающимся фолдам и трём seeds, а не новая независимая test-метрика.\n'
    text+='\n## Как контроль baseline меняет состав\n\n| Train users | Метод | Включений item-признаков | Всего включений | Доля item |\n|---|---|---:|---:|---:|\n'
    for (count,method),group in relation.groupby(['train_users','method']):
        total=int(group.selected_seeds.sum());items=int(group.loc[group.relation=='item','selected_seeds'].sum())
        text+=f'| {count} | {method} | {items} | {total} | {items/total:.1%} |\n'
    text+='\nЧастоты считают включения по запускам, не уникальные признаки. OOF conditioning смещает отбор между отношениями; само изменение состава не доказывает улучшение персонализации. Полная разбивка country/content_type/user/genre сохранена в relation_frequency.csv.\n'
    text+='\n## Описательные выводы\n\n'

    for method in ['mi','cmim','cmim_oof']:
        rows=table.loc[table.method==method];positive=int((rows.ndcg_delta>0).sum())
        text+=f'- {method}: положительная разность к manual в {positive} из {len(rows)} фиксированных запусков; средняя Δ NDCG {rows.ndcg_delta.mean():+.6f}. Усреднение здесь описательное, фолды зависимы.\n'
    text+=f"- Дополнительное conditioning на baseline: положительная Δ к CMIM в {int((comparison.ndcg_delta>0).sum())} из {len(comparison)} запусков, средняя Δ {comparison.ndcg_delta.mean():+.6f}.\n"
    text+='- Эти частоты не являются тестом значимости или вероятностью выигрыша на новых данных. Выводы об устойчивости качества следует читать вместе с Jaccard, разбросом по seeds и размерами контекстных страт. Новый test-победитель не выбирается.\n'
    report.write_text(text)
