# TODO: баги и предупреждения из CI

Самодостаточная рабочая записка: читать и чинить можно без чата, скриншотов лога и знания «как мы сюда пришли». Все пути — от корня репозитория `epi_pinn_llm_param`.

**Снимок лога:** GitHub Actions, job `pytest` в `.github/workflows/tests.yml`, Ubuntu, Python 3.13. На момент разбора: **2 failed, 577 passed, 27 warnings**, ~100 с. Числа покрытия и предупреждений устареют; список корневых багов — нет, пока их не починят.

**Как воспроизвести локально** (после `pip install` по тому же workflow: CPU-torch + `requirements.txt` без строк `torch`/`torchvision` + `requirements-dev.txt`):

```bash
pytest tests/test_pipeline_graph.py::TestRunRejectPath::test_more_than_five_rejections_exceed_the_recursion_limit
pytest tests/test_main_entry.py::TestMain::test_no_acceptance_hits_the_langgraph_recursion_limit
# оба теста помечены slow; полный прогон как в CI:
pytest --cov=agents --cov=utils --cov=formats --cov=config --cov=main_test --cov-report=term-missing
```

В CI LLM не вызывается: подставляется `ScriptedClient` из `tests/support.py`. PINN в e2e-тестах укорачивают до 1–2 эпох (`monkeypatch` в `tests/test_main_entry.py`, фикстура `fast_verification` в `tests/test_pipeline_graph.py`). Суррогат SIRD, граф LangGraph и запись файлов — настоящие.

---

## Зачем этот репозиторий (минимум, чтобы понять баги)

Пайплайн переводит текстовый комментарий эпидемиолога («пик должен быть выше») в параметры SIRD-модели (`β` заражение, `γ`/`ν` выздоровление, `μ` смертность).

1. **Суррогат** (`agents/SurrogateModel.py`) — классический SIRD, быстро считает peak day / height / deaths.
2. **Критик** (детерминированный, `agents/DeterministicCriticAgent*.py`) — accept/reject относительно baseline и ожиданий из комментария.
3. **Генератор** (`agents/EpiParamGeneratorAgent.py`) — LLM предлагает новые `β, γ, μ`. В тестах LLM фейковый.
4. **PINN** (`agents/PINN_const.py`, `PINNAgent.py`) — нейросеть с физическим штрафом; в графе узел `pinn_verification` запускается **только после accept**.
5. Оркестрация — LangGraph в `main_test.py`: класс `OptimizationPipeline` и функция `main()`.

Два слоя, которые путают лог:

| Слой | Вход | Что тестируется |
|------|------|-----------------|
| `OptimizationPipeline.run(...)` | уже готовые baseline-параметры | только граф до `END` |
| `main_test.main()` | CSV + PINN baseline + тот же `run()` + шаги 8–10 (сравнение PINN, summary, full plot) | весь скрипт |

Упавший тест графа бьёт в `run()`. Упавший тест entry point бьёт в `main()`: граф уже **успешно** заканчивается, падение — в шаге 10 после него.

Важно не смешивать два PINN-пути:

- узел графа `pinn_verification` → кладёт `result['pinn_verification']`, пишет `PINN_verification_results/`; **на reject не вызывается**;
- шаг 8 в `main()` — `run_pinn_comparison(...)` → `PINN_comparison_results/`; сейчас вызывается даже если accept не было (берётся последний эпизод);
- шаг 10 читает **`pinn_verification` из result графа** (или JSON с диска), а не `comparison_results`. Поэтому comparison в шаге 8 может пройти, а шаг 10 всё равно падает с `t_final`.

### Словарь

| Термин | Смысл здесь |
|--------|-------------|
| baseline / iter 0 | исходные `β, γ, μ`; в истории всегда `accepted=True` |
| reject-путь | эксперт хочет выше пик, скрипт всегда отдаёт меньший `β` → critic reject до `max_iterations` |
| `GraphRecursionError` | LangGraph: превышен `recursion_limit` (число super-step графа, не `max_iterations`) |
| characterization-тест | фиксирует фактическое поведение, включая баги; см. шапки `tests/test_main_entry.py` и `tests/test_pipeline_graph.py` |
| `# NOTE: current behavior` | «похоже на баг, но сейчас так»; сначала код, потом тест |
| `ScriptedClient` | фейковый LLM: на каждый промпт — заранее заданный JSON (`tests/support.py`) |

Правило тестов: **нельзя** менять только assert, чтобы скрыть изменение кода. Сначала починить прод, потом осознанно обновить тест.

---

## Как читать приоритеты

| Метка | Смысл |
|--------|--------|
| P0 | Красные тесты (симптом в CI). Сами тесты **не** чинить первыми. |
| P1 | Корневая причина в проде. Без этого P0 либо останется красным, либо «позеленеет», закрепив баг. |
| P2 | Warning в логе; на pass/fail не влияет. |
| P3 | Quirk / мёртвый код / покрытие. Не блокер. |
| не баг | Выглядит плохо в логе, но так задумано тестом. |

Порядок работ: **P1 `t_final` → P1 `recursion_limit` в `invoke` → P0 тесты → по желанию P2**.

Менять только тесты недостаточно: `test_no_acceptance_hits_the_langgraph_recursion_limit` падает уже не из‑за LangGraph, а из‑за `UnboundLocalError` в `main_test.py`. Тест, который начнёт ждать этот краш, закрепит баг как норму.

---

## Сводка CI (что именно упало)

| Тест | Ожидание теста | Факт в CI |
|------|----------------|-----------|
| `tests/test_pipeline_graph.py::TestRunRejectPath::test_more_than_five_rejections_exceed_the_recursion_limit` | `langgraph.errors.GraphRecursionError` | `DID NOT RAISE` — граф дошёл до `max_iterations=10` |
| `tests/test_main_entry.py::TestMain::test_no_acceptance_hits_the_langgraph_recursion_limit` | тот же `GraphRecursionError` | граф тоже дошёл до конца, затем `main()` упал в шаге 10: `UnboundLocalError: cannot access local variable 't_final'` |

Оба теста описывают **старый** дефолт LangGraph `recursion_limit=25`. В проекте стоит `langgraph==1.1.3` (`requirements.txt`). С ~1.0.6 дефолт уже **1000+** (в 1.1.x fallback около 10007). Лимит в `run()` не передаётся, поэтому 10 reject-итераций больше не обрываются на 6-й.

Цепочка одного reject-цикла (при `use_pinn=True`):

`sensitivity → intent → generate → surrogate → critic → history → (continue → generate)…`

На reject PINN-верификация **не** запускается — узел `pinn_verification` висит только на ветке `accept` (`main_test.py`, `route_after_history`).

---

## P1. `UnboundLocalError: t_final` в шаге 10 `main()`

**Это реальная ошибка в коде, не в тестах.** Раньше её маскировал `GraphRecursionError` на 6-й итерации. Тест это прямо комментирует: *«before the `t_final` NameError in step 10 is reached»*.

### Где

`main_test.py`, шаг 10 («Creating full comparison plot»), примерно строки 2103–2161.

### Что происходит

1. При одних reject в `result` нет успешного `pinn_verification` (узел графа не вызывался, каталога `PINN_verification_results` нет). Шаг 8 (`run_pinn_comparison` → `PINN_comparison_results/`) тут ни при чём: шаг 10 его не читает.
2. Fallback с диска (`glob("PINN_verification_results/verification_*.json")`) тоже ничего не находит.
3. Ветка `if pinn_verification.get('success'):` не выполняется, `t_final` **не создаётся**.
4. В `else` кладётся только `final_pred = synthetic_data`.
5. Следующая строка всё равно читает `t_final`:

```python
t_train_split = t_final[train_size_points - 1] if len(t_final) > train_size_points else None
```

На Python 3.13 это `UnboundLocalError` (не `NameError`: переменная есть в той же функции, но не связана со значением).

### Соседний скрытый краш (починить вместе)

Даже после появления `t_final` строка

```python
train_split_time=int(t_train_split*2.5),
```

упадёт с `TypeError`, если ряд короче `train_size_points` и `t_train_split is None`.

### Как чинить код

В ветке без успешной верификации взять время из fallback, например `final_pred['t']` / `synthetic_data['t']`.

`train_split_time` считать только если `t_train_split is not None`, иначе передать `None` (у `create_comparison_plot` аргумент уже optional).

### Поведенческий вопрос (решать отдельно, не обязательно для зелёного CI)

Сейчас при «ничего не принято» `main()` не останавливается:

```python
if not optimized_episode:
    print("⚠️ No optimized episode accepted, using last episode")
    optimized_episode = history[-1]
```

Дальше PINN-сравнение и графики идут по **последнему отклонённому** эпизоду, как будто это optimized. Имеет смысл либо early-return после warning, либо явно помечать графики как «rejected fallback». Это дизайн, не причина текущего traceback.

### Тесты после фикса

`test_no_acceptance_hits_the_langgraph_recursion_limit` переименовать/переписать: `main()` завершается, `GraphRecursionError` нет, артефакты шага 10 не падают. Не ждать `UnboundLocalError`.

---

## P1. `recursion_limit` не задан при `graph.invoke`

### Где

`main_test.py`, `OptimizationPipeline.run()`, примерно 1225–1227:

```python
config = {"configurable": {"thread_id": "optimization_1"}}
final_state = self.graph.invoke(initial_state, config)
```

Граф собирается без лимита: `workflow.compile(checkpointer=self.memory)` (~1036).

### Почему это баг, хотя сейчас «само починилось»

Старые тесты верно описывали проблему: при дефолте 25 шагов `max_iterations > 5` был недостижим (оценка в тесте: стартовые шаги + ~4 на цикл). Смена дефолта в LangGraph **скрыла** ограничение, а не сделала лимит частью контракта пайплайна. Следующий апдейт библиотеки снова может сломать `max_iterations=10`.

`recursion_limit` — ключ верхнего уровня `invoke`, не внутри `configurable`. Документация LangGraph: `graph.invoke(inputs, config={"recursion_limit": N, "configurable": {...}})`.

### Как чинить код

Передавать лимит от `max_iterations`, с запасом под узлы до цикла и ветку `pinn_verification`. Ориентир: что-то вроде `6 + 4 * max_iterations` (уточнить по фактическому числу super-step), плюс константа.

Имеет смысл тот же лимит прокинуть и в другие `invoke` графа, если появятся.

### Тесты после фикса

`test_more_than_five_rejections_exceed_the_recursion_limit` больше не должен ждать `GraphRecursionError`.

Новое поведение для закрепления:

- `max_iterations=10`, все предложения reject → история `[0..10]`, 1 accepted (baseline) + 10 rejected;
- `pipeline.run(...)` возвращает state, не исключение;
- промпты генератора: 10 файлов, не 6;
- печать `⏹️  Max iterations reached → END` и `⚠️ No episodes were accepted during optimization`.

Отдельный тест (по желанию): при **явно маленьком** `recursion_limit` (например 5) `GraphRecursionError` всё ещё возникает — чтобы зафиксировать, что лимит реально передаётся.

---

## P0. Устаревшие characterization-тесты (делать после P1)

Оба теста считают дефолт LangGraph равным 25 и обрыв на 6-й итерации. Это уже неправда.

### `tests/test_pipeline_graph.py` (~308–318)

`TestRunRejectPath.test_more_than_five_rejections_exceed_the_recursion_limit`

- Сейчас: `pytest.raises(GraphRecursionError)`, `len(_generator_prompts) == 6`.
- Нужно: успешный прогон до `max_iterations=10` (см. P1 выше).
- Обновить `# NOTE: current behavior` — старый текст про default 25 больше не описывает код.

Рядом уже есть корректный короткий reject-сценарий: `test_loop_until_max_iterations` (`max_iterations=2`) — на него ориентироваться.

### `tests/test_main_entry.py` (~84–95)

`TestMain.test_no_acceptance_hits_the_langgraph_recursion_limit`

- Сейчас: ждёт `GraphRecursionError`, отсутствие `PINN_verification_results`, ровно 6 логов генератора.
- Нужно: после фикса `t_final` — полный `main()` без traceback; шаг 10 не падает; верификация по-прежнему не обязана появляться (reject → нет узла PINN verification).
- Переименовать тест: имя с `hits_the_langgraph_recursion_limit` станет ложью.

Нельзя ограничиться заменой ожидаемого исключения на `UnboundLocalError`.

---

## P2. Warning: `torch.tensor(sourceTensor)` в `TorchStandardScaler.fit`

### Где

`agents/PINN_const.py:67`

```python
x = torch.tensor(x).float().to(device)
```

Лог: *To copy construct from a tensor, it is recommended to use `sourceTensor.detach().clone()` rather than `torch.tensor(sourceTensor)`.*

### Откуда в CI

В проде `EINN_PINN` обычно вызывает `fit` на numpy (`.cpu().numpy()`). Warning даёт тест `tests/test_pinn.py::TestTorchStandardScaler.test_fit_transform`, который передаёт уже тензор: `fit_transform(torch.tensor([2.0, 4.0]), "cpu")`.

Путь в коде всё равно существует: `fit` не проверяет тип.

### Как чинить

```python
if torch.is_tensor(x):
    x = x.detach().clone().float().to(device)
else:
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
```

Тесты scaler после этого должны пройти; `test_fit_transform` не должен шуметь.

### Рядом (не из этого warning, но тот же класс)

`tests/test_pinn.py` уже помечает in-place `transform()` как возможный баг: `x -= mean` портит тензор/массив вызывающего. Чинить только если решите, что scaler не должен мутировать вход.

---

## P2. Matplotlib: нет глифов `✓` / `✗` / `✅` в CI

Ubuntu CI: Liberation Sans и DejaVu Sans Mono. В логе:

- Glyph 10003 (Check Mark `✓`) missing from Liberation Sans
- Glyph 10007 (Ballot X `✗`) missing from Liberation Sans
- Glyph 9989 (White Heavy Check Mark `✅`) missing from DejaVu Sans Mono
- variation selector-16 (хвост эмодзи)

На логику тестов не влияет; в PNG вместо символов — «тофу»/пустые клетки.

### Где в графиках (это важно; консольный `print` шрифтов не трогает)

| Символ | Место | Зачем |
|--------|--------|--------|
| `✓` / `✗` | `main_test.py` ~2277, `create_summary_report`, колонка Status таблицы истории | статус accept/reject |
| `✅` | `main_test.py` ~2370, тот же отчёт, блок SUMMARY, `family='monospace'` | «Optimization successfully changed…» |

Консольная таблица в `OptimizationPipeline.run()` (~1304, `✅`/`❌`) warning **не** вызывает.

### Как чинить

На фигурах matplotlib не использовать эти глифы. Варианты: `OK`/`X`, `ACC`/`REJ`, `Y`/`N`, или подключить шрифт с символами (Noto Sans / Noto Emoji) в CI — тяжелее, чем ASCII.

Если после замены есть assert на точный текст PNG/summary — обновить его. Сейчас тесты в основном смотрят на stdout и наличие файлов, не на пиксели.

---

## P2. Warning: `tight_layout` несовместим с осями фигуры

Лог: *This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.* Срабатывает на `plt.tight_layout()`.

### Где почти наверняка

`create_summary_report` (`main_test.py` ~2191–2401):

- `plt.figure` + `GridSpec(5, 2, …)`;
- несколько `axis('off')` (заголовок, таблицы, ASCII-рамки);
- `ax.table(...)`;
- в конце `plt.tight_layout()` **и** `savefig(..., bbox_inches='tight')`.

`tight_layout` плохо дружит с GridSpec + «выключенными» осями + таблицами. `bbox_inches='tight'` уже делает обрезку при сохранении.

Другие `plt.tight_layout()` в файле (сравнение 2×2 и т.п.) могут быть безвредны; в логе ругается, скорее всего, summary report.

### Как чинить

Убрать `plt.tight_layout()` у этой фигуры **или** создать её с `constrained_layout=True` / `layout='constrained'` и не смешивать с `tight_layout`. Не вызывать оба сразу.

---

## P3. Рассинхрон «Max iterations: 5» vs `run(..., max_iterations=10)`

### Где

`main_test.py` ~1827 печатает `Max iterations: 5`, а ~1841 передаёт `max_iterations=10`.

Тест accept-пути это уже знает:

```python
assert "Max iterations: 5" in out  # printed value; run() actually gets 10
```

### Как чинить

Одна константа на печать и на `run()`. После этого поправить assert в `test_accept_path_produces_every_artifact`.

На CI не влияет, путает чтение логов.

---

## Не баг: одинаковые параметры 10 раз и 10 reject

В логе CI на reject-пути:

- iter 0 (BASELINE): β=0.0910, ν=0.0553, μ=0.00850, peak height 51, accepted;
- iter 1–10: **один и тот же** набор β=0.0850, ν=0.0553, μ=0.00850, height 35, все REJECT;
- critic: «Height: expected higher, but got 35»;
- итог: Total 11, Accepted 1, Rejected 10, `No episodes were accepted`, `Max iterations reached`.

Это **сценарий теста**, не зациклившийся LLM. В логе колонка `ν` — тот же recovery rate, в коде поле `gamma`.

Оба упавших теста делают (helpers в `tests/support.py`):

```python
params_json(0.085, 0.0553, 0.0085)  # lower β while the expert wants a higher peak
```

`ScriptedClient` каждый раз отдаёт один и тот же JSON, без истории и без «поиска». Экспертный комментарий `"Need higher peak"` → `IntentParser` + детерминированный critic: height 35 < baseline 51 → REJECT. Настоящий генератор в этом прогоне не участвует.

Не тратить время на «генератор застрял», пока речь о CI.

---

## Другие баги, уже записанные в тестах (не из этого лога CI)

Их CI сейчас **не** валит: тесты наоборот закрепляют кривое поведение. Имеет смысл завести отдельные задачи, когда будете чинить по `# NOTE`.

| Тема | Где зафиксировано | Суть |
|------|-------------------|------|
| После 3 неудачных parse генератора `generated_params is None`, critic делает `new_params['beta']` | `tests/test_pipeline_graph.py::test_generator_that_never_yields_json_crashes_in_the_critic` | падает `TypeError: not subscriptable` вместо понятной ошибки parse |
| `route_after_history` пишет `state['iteration'] += 1` в **копию** стейта | `test_loop_until_max_iterations` | счётчик в роутере не действует; итерация растёт в генераторе. Сейчас тесты ждут «нет двойного инкремента» — если починить мутацию, проверить, что счётчик не скакнёт через 2 |
| `RetryParser.parse`: строка после цикла | `utils/RetryParser.py:42` | `raise OutputParserException("Failed to parse response")` недостижима: выход только через `return` или `raise` в `else`. Coverage 96%, missing 42 |

---

## Coverage из лога (не баги, не блокер CI)

Общее покрытие ~98%. Джоба не падает из‑за дырок. Строки из `term-missing`:

| Файл | Пропущено | Комментарий |
|------|-----------|-------------|
| `agents/DeterministicCriticAgent3.py` | 90, 101, 137, 148 | ветки critic |
| `agents/EpiParamGeneratorAgent.py` | 479 | редкий путь генератора |
| `agents/ParameterCriticAgent.py` | 384, 389, 412, 456 | retry/error LLM-critic |
| `main_test.py` | 278–286, 1339–1355, 1856–1857, 1910–2010, 2133–2135, 2723 | early-return «baseline not found», печать PINN validation, ветка без uncertainty, куски comparison |
| `utils/RetryParser.py` | 42 | мёртвый raise, см. выше |

Закрывать имеет смысл вместе с правкой соответствующего кода, а не отдельным «добором процентов».

---

## Чеклист работ

Код (сначала):

- [ ] `main_test.py` шаг 10: задать `t_final` в ветке без `pinn_verification`
- [ ] `main_test.py` шаг 10: не вызывать `int(t_train_split * 2.5)` при `None`
- [ ] `OptimizationPipeline.run`: передать `recursion_limit` в `graph.invoke` от `max_iterations`
- [ ] (по желанию) не строить «optimized» отчёт из последнего reject
- [ ] `PINN_const.py`: безопасное копирование в `TorchStandardScaler.fit`
- [ ] `create_summary_report`: ASCII вместо `✓✗✅` в таблице и SUMMARY
- [ ] `create_summary_report`: убрать/заменить `tight_layout` на GridSpec-фигуре
- [ ] согласовать печать Max iterations с аргументом `run()`

Тесты (после кода):

- [ ] переписать `test_more_than_five_rejections_exceed_the_recursion_limit`
- [ ] переписать/переименовать `test_no_acceptance_hits_the_langgraph_recursion_limit`
- [ ] обновить `# NOTE: current behavior` в этих тестах
- [ ] поправить assert `"Max iterations: 5"`, если сменится печать
- [ ] (по желанию) тест, что маленький явный `recursion_limit` всё ещё даёт `GraphRecursionError`

Не делать:

- [ ] не ждать в тестах `UnboundLocalError` как «правильное» поведение
- [ ] не чинить «застрявший генератор» по таблице iter 1–10 из CI
- [ ] не раздувать покрытие ради дырок из таблицы выше, пока нет правки кода
