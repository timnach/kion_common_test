"""Descriptive OFAT effects and history slices; no model selection here."""
import itertools
from pathlib import Path
import json
import numpy as np
import pandas as pd


def enrich(root,config):
    root=Path(root);table=pd.read_csv(root/'metrics.csv');case_contrasts=[];slices=[];costs=[]
    for fold,seed in itertools.product(config['validation_cutoffs'],config['seeds']):
        directory=root/fold/f'seed{seed}';record=json.loads((directory/'complete.json').read_text())
        profile=pd.read_csv(root/'feature_diagnostics'/'ablation'/(fold+'.csv')).set_index('user_id')
        base=pd.read_csv(directory/'constant_per_user.csv').set_index('user_id')
        history=profile.loc[base.index,'history_n']
        groups=pd.cut(history,[0,5,20,np.inf],labels=['1-5','6-20','21+'])
        for name,met in record['metrics'].items():
            per=pd.read_csv(directory/(name+'_per_user.csv')).set_index('user_id').loc[base.index]
            data=pd.DataFrame({'group':groups,'ndcg':per.ndcg,'delta_constant':per.ndcg-base.ndcg})
            for label,g in data.groupby('group',observed=True):
                slices.append({'fold':fold,'seed':seed,'method':name,'history_group':str(label),'users':len(g),
                    'mean_ndcg':g.ndcg.mean(),'mean_delta_constant':g.delta_constant.mean()})
            if name!='manual':costs.append({'fold':fold,'seed':seed,'method':name,'fit_seconds':record['fit_seconds'][name]})
        for name,ci in record['bootstrap_vs_constant'].items():case_contrasts.append({'fold':fold,'seed':seed,'comparison':name,**ci})
    contrast=pd.DataFrame(case_contrasts);contrast.to_csv(root/'paired_contrasts.csv',index=False)
    sliced=pd.DataFrame(slices);sliced.to_csv(root/'history_slices.csv',index=False);pd.DataFrame(costs).to_csv(root/'fit_costs.csv',index=False)
    profile_rows=[]
    for fold in config['validation_cutoffs']:
        d=pd.read_csv(root/'feature_diagnostics'/'ablation'/(fold+'.csv'));d['history_group']=pd.cut(d.history_n,[0,5,20,np.inf],labels=['1-5','6-20','21+'])
        for label,g in d.groupby('history_group',observed=True):
            profile_rows.append({'fold':fold,'history_group':str(label),'users':len(g),
                'zero_regularity_fraction':float((g.regularity_nonzero==0).mean()),
                'constant_regularity_fraction':float((g.regularity_range<1e-12).mean()),
                **{c+'_median':float(g[c].median()) for c in g if c.endswith('_effective_h')}})
    pd.DataFrame(profile_rows).to_csv(root/'history_diagnostics.csv',index=False)
    report=root/'REPORT.md';text=report.read_text().split('\n## Дополнительный разбор эффектов')[0]
    text+='\n## Дополнительный разбор эффектов\n\n'
    text+='| Сравнение | Средняя Δ NDCG | Положительных повторов | CI целиком >0 | CI целиком <0 |\n|---|---:|---:|---:|---:|\n'
    for name,g in contrast.groupby('comparison'):
        text+=f"| {name} | {g.delta.mean():+.6f} | {int((g.delta>0).sum())}/{len(g)} | {int((g.ci_low>0).sum())} | {int((g.ci_high<0).sum())} |\n"
    text+='\nРазности считаются в соответствующем направлении: строки alpha/beta/adaptive — относительно constant-h; constant_minus_uniform — эффект временного взвешивания относительно тех же трёх признаков без decay. Интервалы не скорректированы на множественные сравнения.\n'
    text+='\n### Активность и регулярность истории\n\n| Неделя | Длина истории | Users в feature-пуле | R=0 | R постоянно по истории |\n|---|---|---:|---:|---:|\n'
    for r in profile_rows:text+=f"| {r['fold']} | {r['history_group']} | {r['users']} | {r['zero_regularity_fraction']:.1%} | {r['constant_regularity_fraction']:.1%} |\n"
    text+='\nЭто диагностический feature-пул, а не дополнительная train-когорта. При R=0 alpha не меняет ни один из трёх признаков; при постоянном положительном R меняется mass, но не affinity/ESS. Медианные effective h общей adaptive-формулы сохранены в history_diagnostics.csv.\n'
    text+='\n### Activity adaptation по длине истории\n\n| Вариант | История | Средняя Δ NDCG к constant-h |\n|---|---|---:|\n'
    values=sliced.loc[sliced.method.str.startswith('adaptive_')].groupby(['method','history_group']).mean_delta_constant.mean()
    for (name,label),value in values.items():text+=f'| {name} | {label} | {value:+.6f} |\n'
    text+='\nЭто описательное среднее per-case разностей внутри группы. Размеры групп сохранены в CSV; эти разрезы не используются для выбора отдельной модели/параметра каждому пользователю.\n'
    text+='\nActivity-adaptive правило меняет и распределение h между пользователями, и средний масштаб памяти. Без отдельного matched-global-h контроля его эффект нельзя приписать одной персонализации. Калибровочный h выбран только из заявленной сетки; значения вне неё не исследовались.\n'
    text+='\n### Что можно заключить\n\n'
    for name in ['constant_minus_uniform','constant_minus_manual']:
        g=contrast.loc[contrast.comparison==name]
        text+=f"- {name}: средняя Δ {g.delta.mean():+.6f}, положительная разность в {int((g.delta>0).sum())}/{len(g)} повторов.\n"
    for prefix,label in [('alpha_','alpha'),('beta_','beta'),('adaptive_','shared activity-adaptive h')]:
        g=contrast.loc[contrast.comparison.str.startswith(prefix)]
        text+=f"- Отдельная ветка {label}: знак положителен в {int((g.delta>0).sum())}/{len(g)} фиксированных сравнений. Значения параметров оцениваются раздельно в таблице; здесь не выбирается новый победитель по test.\n"
    text+='- Все выводы условны на фиксированных S, retrieval, выбранном калибровочном h и ранкере. Из этой серии нельзя делать выводы о полном совместном оптимуме или полезности свободных индивидуальных lambda.\n'
    means=table.groupby('method')['ndcg@10'].mean()
    text+='\n### Практическая интерпретация этой серии\n\n'
    text+=f"Средние NDCG@10: manual {means['manual']:.6f}, uniform {means['uniform']:.6f}, constant-h {means['constant']:.6f}. "
    text+='Близость constant-h к manual сама по себе не подтверждает пользу затухания: основной контроль времени — uniform с теми же агрегатами.\n\n'
    text+='Для оценки alpha/beta/adaptive следует читать каждую заранее заданную величину отдельно, вместе со знаком по повторам и интервалами. Положительное среднее и частый положительный знак без интервалов выше нуля — слабое диагностическое свидетельство, а не подтверждённое улучшение. Отсутствие убедительного улучшения также не доказывает эквивалентность методов.\n\n'
    text+='Девять повторов не являются девятью независимыми тестами: seeds используют общие validation-когорты, временные train-фолды перекрываются. User-bootstrap характеризует парную разность на конкретной неделе при конкретном seed; общий доверительный интервал из этих девяти средних не строится.\n'
    report.write_text(text)
