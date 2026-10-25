# Автоматический отбор признаков на KION

Исследование автоматического построения и отбора исторических агрегатов для
двухэтапной рекомендательной системы: извлечение 100 кандидатов и CatBoostRanker.
Сравниваются ручные признаки, полный банк, MI, CMIM, CMIM с временными OOF,
случайный и направленный поиск, а также варианты временного затухания.

[Отчёт PDF](report/automatic_feature_selection.pdf)

## Основной эксперимент

- Train-срезы: 12/19/26 июля и 2 августа 2021; до 2000 query-пользователей на срез.
- Validation: 9–15 августа. Test: 16–22 августа.
- На test: общие 5000 warm future-active пользователей и 100 кандидатов.
- Seeds: 17/42/73. Общие настройки CatBoost, train sample внутри seed и доступная история.
- Retrieval: популярность/жанры + ItemKNN + EASE.
- Сохраняется вся доступная прошлая история для признаков и retrieval.

| Метод | Mean NDCG@10 | Число входов |
|---|---:|---:|
| Ручной baseline | 0,154842 | 56 |
| Весь банк | 0,156234 | 187 |
| MI | 0,156136 | 68 |
| CMIM | 0,156365 | 68 |
| CMIM + OOF | 0,156265 | 68 |
| Случайный поиск | 0,154898 | 68 |
| Направленный поиск | 0,156701 | 68 |
| CMIM + затухание | 0,156904 | 71 |

Компактный набор даёт близкое наблюдаемое качество при сокращении 187 → 68
входов. Эквивалентность качества, снижение переобучения и ускорение чистого
инференса отдельно не доказаны. По сравнению с ручным baseline есть небольшой
локальный прирост; общее преимущество на новых данных не установлено.
Автоматизация ограничена заранее заданной грамматикой отношений и агрегатов.

Августовские периоды уже просматривались раньше: общий test обеспечивает
сопоставимость, но не является нетронутым holdout. Mean по seeds — среднее
метрик отдельных ранкеров, не качество ансамбля. Нули KION не означают
подтверждённый отказ после показа: impression log отсутствует.

Полные таблицы и парные интервалы: `results/common_test/`.
Дополнительные исследования retrieval, устойчивости и decay представлены
в соседних каталогах `results/` и в отчёте.

## Структура

```text
configs/    четыре действующих протокола
src/        код признаков, моделей, отбора и экспериментов
scripts/    загрузка данных и независимые проверки
tests/      тесты актуального кода
results/    небольшие опубликованные таблицы и условия экспериментов
report/     один актуальный PDF, LaTeX, два графика и генератор отчёта
```

Два каталога `results/research_v2*` содержат только исходные когорты и truth,
которые реально использует retrieval benchmark. Это входные данные протокола,
а не дополнительные старые решения.

## Установка

Рекомендуется Python 3.12; версии основной среды перечислены в
`requirements-common-test-lock.txt`. Для строгого повторения предпочтительна
та же версия Python и библиотек, что в `results/common_test/environment.json`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,retrieval]'
python -m pytest -q
```

В Windows активация окружения: `.venv\Scripts\activate`.
Переменные окружения в примерах ниже заданы в синтаксисе Bash; в другой
оболочке используйте её синтаксис.

## Запуск с нуля

```bash
python scripts/download_data.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=6 PYTHONPATH=src \
  python -m kion_recsys.common_test --config configs/common_test.json
PYTHONPATH=src python scripts/verify_common_test.py
```

Новые расчёты по умолчанию пишутся в `runs/`, отдельно от опубликованных
таблиц `results/`. Поэтому отсутствие больших файлов в репозитории не создаёт
неполные checkpoints. Повтор запуска использует checkpoints своего `runs/`.
Для изменения протокола задайте другой `output_dir` в копии конфигурации.

Данные загружаются из закреплённого commit KION с проверкой SHA-256.
Не меняйте опубликованные таблицы, чтобы подставить новый test-победитель.

## Дополнительные эксперименты

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=6 PYTHONPATH=src \
  python -m kion_recsys.retrieval_benchmark --config configs/retrieval_benchmark.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=6 PYTHONPATH=src \
  python -m kion_recsys.selection_stability --config configs/selection_stability.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=6 PYTHONPATH=src \
  python -m kion_recsys.decay_ablation --config configs/decay_ablation.json
```

Decay использует feature-кэш серии устойчивости, поэтому запускается после неё.
У каждого исследования есть соответствующий `scripts/verify_*.py`;
без аргумента он проверяет новый расчёт в `runs/`.

## Полные сохранённые результаты

Большие модели, top-10, per-user метрики, OOF-массивы и подробный аудит вынесены
в **experiment_artifacts.zip**. Этот файл следует приложить к GitHub Release,
а не коммитить в репозиторий. Распакуйте его содержимое в корень проекта:
оно дополнит `results/` и не заменит код или опубликованные таблицы.

После распаковки проверка прежних результатов не требует нового обучения:

```bash
PYTHONPATH=src python scripts/verify_common_test.py results/common_test
PYTHONPATH=src python scripts/verify_retrieval_benchmark.py results/retrieval_benchmark
PYTHONPATH=src python scripts/verify_selection_stability.py results/selection_stability
PYTHONPATH=src python scripts/verify_decay_ablation.py results/decay_ablation
```

Нельзя запускать полную проверку прогнозов только по сводным CSV без архива:
для неё необходимы сами сохранённые прогнозы и модели.

## Сборка отчёта

Из каталога `report`:

```bash
python build_report.py
xelatex -interaction=nonstopmode -halt-on-error automatic_feature_selection.tex
xelatex -interaction=nonstopmode -halt-on-error automatic_feature_selection.tex
```

Генератор читает небольшие таблицы `../results/`; модели и сырые данные
ему не нужны. Для XeLaTeX требуются пакеты из преамбулы, Nimbus Roman OTF
и DejaVu Sans / Sans Mono. Готовые графики и PDF уже приложены.
