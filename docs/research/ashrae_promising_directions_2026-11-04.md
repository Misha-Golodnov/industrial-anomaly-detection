# RP-1043: второй раунд перспективных направлений

Дата: 4 ноября 2026. Продолжение [первого исследования](ashrae_rp1043_research_2026-11-04.md). [Все результаты второго раунда](../../results/ashrae_directions_2026-11-04/report.md).

**Статус: исследовательская разработка.** Результаты прежнего test уже просмотрены. В этой серии нет нового нетронутого подтверждающего теста, независимого чиллера или гарантии SOTA. Новые выигрыши рассматриваются как кандидаты для дальнейшего подтверждения.

## 1. На какую литературу опирается новый код

| Направление | Первоисточник | Что именно перенесено |
|---|---|---|
| Физические показатели чиллера | [Bezyan et al., Feature Selection and Fault Detection Under Dynamic Conditions, 2026, Table 3](https://doi.org/10.3390/electronics15010208) | Формулы LMTD испарителя/конденсатора, эффективности переохлаждения/перегрева. Добавлены прозрачные flow/heat-transfer proxies. Не воспроизводится wrapper selection по известному fault. |
| Значение каналов и признаки слабых отказов | [Xia et al., Incipient Chiller Fault Diagnosis Using an Optimized LSSVM, 2021](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2021.755649/full), [Feasibility and improvement of FDD based on factory-installed sensors, 2019](https://www.sciencedirect.com/science/article/abs/pii/S1359431119347209) | `TRC_sub` — уже переохлаждение; подходы/перегрев и чувствительность к утечке. Supervised SVM здесь — обычный RBF-SVM, не оптимизированный GSA-LSSVM статьи. |
| Разделение частот и накопление свидетельства | [Lin, Tang, Tsai, LFD/FDSD, 2025](https://doi.org/10.1016/j.ijrefrig.2025.03.003) | Собственные признаки двух полос FFT и векторная MEWMA на завершённых окнах. **Не полная реплика FDSD:** нет Transformer–BiLSTM и опубликованного двухступенчатого MEWMA. |
| Временные псевдобатчи | [Bezyan et al., DBSCAN–KPCA–UPCA–EWMA, 2026](https://doi.org/10.1016/j.buildenv.2026.115033) | Проверка более длинного контекста, reset накопления по границам блока/пропускам. Предварительное выделение нормы DBSCAN–KPCA не требуется в чистом normal-only треке и не воспроизведено. |
| Regularized Gaussian geometry | [Ledoit–Wolf, 2004](https://www.ledoit.net/Well-conditioned2004.pdf), [Friedman, Hastie, Tibshirani, graphical lasso](https://pmc.ncbi.nlm.nih.gov/articles/PMC3019769/) | Сравнение автоматической усадки, усадки к диагонали и sparse precision. Параметр выбирается по likelihood **отложенной нормы**, без fault labels. |
| Уравнение состояния хладагента | [CoolProp High-Level API](https://coolprop.org/coolprop/HighLevelAPI.html) и [экспериментальное описание RP-1043 с R134a](https://www.ijmerr.com/2023/IJMERR-V12N4-223.pdf) | Отдельная абляция `P_measured - P_sat,R134a(T)` с явными предположениями °F/psig и atmospheric reference. |

Для LFD/FDSD доступны описание preprocessing, Transformer–BiLSTM и параметры обучения в индексируемых фрагментах первичной страницы; проверенная полноценная реализация статьи всё ещё отсутствует. Для VMD–KPCA перенесены именно физические формулы. **Сам VMD, его точный polynomial-KPCA pipeline и fault-assisted feature selection в этом раунде не обучались.**

## 2. Что было зафиксировано и посчитано

Сохранились исходные 32 CSV, allowlist из 26 каналов и семи fault families, один уникальный normal файл, envelope для работающего чиллера, stride 6, блок 180 строк, edge embargo 12, пять seeds и четыре протокола blocked/loto40/loto45/loto50. Никаких fault codes, timestamps, стендовых управляющих состояний или готового COP в моделях нет.

Новые основные серии:

- **680 обучений normal-only:** 17 методов × окна 12/48 × четыре протокола × пять seeds.
- **960 обучений diagnosis:** 12 комбинаций представления/классификатора × два окна × четыре протокола × пять seeds × seen SL1 / SL2–SL4 → SL1.
- **360 обучений covariance follow-up:** девять вариантов × два окна × четыре протокола × пять seeds. Эта серия запланирована **после просмотра** физического/спектрального раунда; это явно адаптивный этап разработки. Два исходных baseline повторены для контроля.
- **240 обучений scaling audit:** четыре новых варианта нормировки физических признаков и два повторных контроля × два окна × четыре протокола × пять seeds. Это ещё один адаптивный этап, проверяющий общий numerical floor для разномерных признаков.

Итого **2240 завершённых обучений основной серии**, включая повторные baseline; smoke checks в это число не входят. Веса/архитектуры не настраивались по fault test. Covariance grid выбирается только на normal reference; final calibration остаётся отдельной. Seeds не являются независимыми физическими экспериментами.

В диагностике normal-only reference/calibration дополнительно доступны signature-моделям для оценки нормального распределения и rejection gate. Обычные discriminative heads обучаются на fit-блоках. Это различие информационных бюджетов отражено в manifest, поэтому наличие gate не интерпретируется как чистое архитектурное преимущество.

## 3. Физические преобразования

Для испарителя и конденсатора:

\[
L_e=\frac{TWEI-TWEO}{\log((TWEI-TRE)/(TWEO-TRE))},\qquad
L_c=\frac{TWCO-TWCI}{\log((TRC-TWCI)/(TRC-TWCO))},
\]

\[
\epsilon_{sub,c}=\frac{TRC_{sub}}{TRC-TWCI},\qquad
\epsilon_{sh,c}=\frac{Tsh_{dis}}{TR_{dis}-TWCO}.
\]

`TRC_sub` не вычитается из TRC повторно. Температурные разности, approach temperatures, flows, pressure lift, oil differences и разности дублирующих температурных датчиков образуют fixed feature map. Не все эти признаки инвариантны к нагрузке; физически интерпретируемый признак не становится автоматически nuisance-free.

Для численной устойчивости логарифмического среднего при \(a,b>0\):

\[
m=(a+b)/2,\quad z=(a-b)/(a+b),\quad L=mz/\operatorname{atanh}(z),
\]

\[
L=m(1-z^2/3-4z^4/45+O(z^6)).
\]

При недопустимых endpoint signs возвращаются отдельный invalid flag и нулевой placeholder. Строка не исчезает из test. В исходной нормальной выборке W12 окна с invalid LMTD составили около 0.29% для испарителя и 1.57% для конденсатора. Это аргумент против механического навязывания steady-state закона всем переходным состояниям.

`FWE*ΔT_e`, `FWC*ΔT_c`, их отношения к LMTD и мощности — **proxies в исходных единицах**, не утверждения о вычисленном SI heat rate или истинном UA. Для деления предусмотрены явные floors 0.5 по температурной разности и 10 по flow-temperature proxy. Эти floors фиксированы, не оптимизированы по test.

В отдельной EOS-абляции:

\[
T_K=(T_F-32)5/9+273.15,\quad
r_P=P_{psig}-\left(P_{sat,R134a}(T_K)/6894.757293168-14.6959488\right).
\]

На норме 5/50/95% квантили residual psig приблизительно равны −0.045/0.095/0.183 для испарителя и −0.056/0.150/0.322 для конденсатора. Это поддерживает совместимость принятой шкалы, **но не доказывает независимость P/T измерений**: они могли быть связаны при формировании данных. EOS-абляция не улучшила основное обнаружение; дополнительные зависимые признаки способны изменить усадку и ухудшить метрику.

## 4. Почему помогла работа с covariance/precision

Gaussian novelty score:

\[
a(x)=(\phi(x)-\widehat\mu)^\top\widehat\Sigma^{-1}(\phi(x)-\widehat\mu).
\]

Усреднение окна меняет не только сигнал, но и covariance шума. При stationarity:

\[
\operatorname{Var}(\bar x_w)=w^{-2}\sum_{i=1}^w\sum_{j=1}^w\Gamma(i-j).
\]

Отсюда не следует независимость отсчётов или точное снижение дисперсии в \(w\) раз. Более длинное окно — проверяемая гипотеза для слабого устойчивого сдвига; цена — потеря временного разрешения. Из-за округлённого времени в этой копии выигрыш/проигрыш delay в секундах не заявляется.

Для diagonal-target shrinkage:

\[
\widehat\Sigma_\lambda=(1-\lambda)S+\lambda\operatorname{diag}(S)+\text{numerical ridge},
\]

\[
\widehat\lambda=\arg\min_{\lambda\in\Lambda}\frac12\left[\log\det\widehat\Sigma_\lambda+
\frac1{n_{ref}}\sum_{i\in ref}r_i^\top\widehat\Sigma_\lambda^{-1}r_i\right].
\]

Данные ref здесь только нормальные. Final threshold оценивается по другой части. В коде numerical ridge \(10^{-6}I\) добавлен к sample covariance, чтобы обрабатывать константные validity flags; это зафиксировано до итогового запуска. Этот критерий использует зависимые окна как development criterion, без гарантии независимого predictive likelihood.

Graphical lasso решает

\[
\min_{\Theta\succ0}\operatorname{tr}(S\Theta)-\log\det\Theta+
\lambda\sum_{i\ne j}|\Theta_{ij}|.
\]

Он задаёт sparse conditional-dependence graph, а не причинный граф чиллера. Ненулевые/нулевые связи не получают причинной интерпретации. Неконвергировавшие кандидаты исключались и записаны в selection.csv; всего 339 candidate fits были отброшены. Выбранные модели конечны. В этой серии sparse precision не превзошла dense shrinkage, несмотря на более содержательную структуру.

## 5. Основные результаты normal-only

На SL1, blocked, номинальный calibration budget 5%:

| Метод | Окно | AUROC | Recall | Normal FPR |
|---|---:|---:|---:|---:|
| Исходный Ledoit–Wolf | 12 | 0.9413 | 65.90% | 10.31% |
| Ledoit–Wolf, длинное окно | 48 | 0.9571 | 68.10% | 7.25% |
| Diagonal shrinkage, выбор по normal reference | 48 | 0.9669 | 66.80% | 6.51% |
| Фиксированная shrinkage 1e-4 | 48 | 0.9675 | 63.67% | 6.51% |

Фиксированная 1e-4 немного выше по AUROC, но не выбрана как универсальный метод по этому test. Normal-reference selection имеет самостоятельный алгоритм выбора. Разные achieved FPR не позволяют трактовать recall как сравнение при одинаковой фактической ошибке.

Длинные окна отбрасывают часть окон около границ и некорректных точек. Поэтому выполнен **дополнительный matched-endpoint анализ**: оставлены одинаковые calibration/test концы для W12 и W48, пороги пересчитаны по одинаковому calibration support. Для raw covariance CV blocked SL1 AUROC вырос с **0.9502 до 0.9669**; для Ledoit–Wolf — с **0.9477 до 0.9571**. Это поддерживает пользу длинного контекста, а не только удаления трудных test окон. Fit support при этом всё ещё различается: это не полностью контролируемое доказательство причинного механизма.

При Hold 50 °F задача остаётся нерешённой. W48 raw covariance CV: AUROC 0.9190, FPR 62.18%. Invariant covariance CV снижает FPR до 41.38% при AUROC 0.9213; это всё ещё неприемлемо как готовый монитор. Уменьшение FPR физической polynomial-моделью сопровождается потерей recall и AUROC; отчёт показывает их вместе.

GP, PCA-SPE, spectra, MEWMA и scan не дали универсального выигрыша. На коротких завершённых окнах спектральное представление добавляет шумные и избыточные координаты. Это вывод о конкретной реализации; он **не опровергает опубликованный Transformer–BiLSTM FDSD**.

Проведён дополнительный **аудит единиц масштабирования**. Общий floor 0.05 в исходных физических Gaussian-абляциях подавлял малые отношения, например inverse-COP proxy. В новых `thermo_unit_*` каждая неконстантная координата масштабируется по собственному normal-fit std, константная — по 1. Для \(\phi'_j=a_j\phi_j+b_j\), \(a_j>0\), нормированные координаты совпадают; инвариантность метрики при переводе мощности из kW в W проверена отдельным тестом. Это более корректная обработка смешанных единиц, однако лидера не изменило: blocked W48 thermo-unit LW AUROC 0.9024, прежний thermo LW 0.9154. Четыре новые абляции с LW/OAS/CV/EOS сохранены, неблагоприятные результаты не удалены.

GP variance correction здесь — эвристическая поправка на неопределённость conditional mean, а не подтверждённая calibrated predictive density. Провал этих небольших GP-моделей не доказывает непригодность probabilistic physical models вообще.

## 6. Diagnosis: нелинейные физические признаки и регуляризованная LDA

Для fixed nonlinear embedding \(z=\phi(X)\), class means \(\mu_k\) и regularized pooled within-class covariance \(\Sigma_w\) LDA принимает решение по

\[
\delta_k(z)=z^\top\Sigma_w^{-1}\mu_k-\tfrac12\mu_k^\top\Sigma_w^{-1}\mu_k+\log\pi_k.
\]

Использованы equal priors \(\pi_k=1/8\), `solver=lsqr`, автоматическая усадка covariance. Физические признаки — nonlinear coordinate map, а не hard physical loss. Для raw+thermo используется 104 mean/std признака; признаки и их scales оцениваются только на supervised fit.

На blocked seen SL1:

| Метод | Окно | Macro-F1 | Normal FPR |
|---|---:|---:|---:|
| Raw logistic regression | 12 | 0.9046 | 16.07% |
| Raw+thermo LDA | 12 | **0.9368** | **13.33%** |
| Raw logistic regression | 48 | 0.8951 | 20.43% |
| Raw+thermo LDA | 48 | **0.9499** | **9.91%** |

Это наиболее устойчивый новый supervised-кандидат: W48 seen SL1 macro-F1 на Hold 40/45/50 °F равны **0.9128/0.9709/0.9307**. Однако normal FPR составляет соответственно 8.93%/0.71%/19.20%; высокая общая F1 не скрывает проблему нормы.

**Граница применимости существенна:** при обучении SL2–SL4 → SL1 raw+thermo LDA резко ухудшается (blocked W48 macro-F1 0.4024). Нормализация тяжести и generalization по severity — отдельная задача. Предыдущая temporal CNN с blocked transfer macro-F1 0.7509 остаётся более сильным ориентиром для этого сценария. Вновь посчитанный raw logistic имеет другой порядок промежуточных float-преобразований; выводы нового раунда основаны на его одновременно пересчитанных baseline, а не на предположении о побайтной идентичности всех представлений.

## 7. Что получилось с сигнатурой неизвестной тяжести

Собственная математическая адаптация физической fault-signature идеи:

\[
r\mid f,a\sim N(a d_f,I),\quad a\ge0,
\]

\[
\widehat a_f=\max(r^\top d_f,0)/\|d_f\|^2,\quad
2\log\Lambda_f=\max(r^\top d_f,0)^2/\|d_f\|^2.
\]

Для фиксированной \(\|d_f\|=1\) направление fault prediction инвариантно к положительному масштабированию \(r\), пока объект проходит normal rejection gate. Gate калибруется по максимуму свидетельства на normal calibration. Матрица whitening и сигнатуры оценены, поэтому точный Gaussian null и target-domain FPR не заявляются.

Проверены ray на invariant/thermo, conditional thermo ray и sign-unrestricted rank-2 fault subspaces. В целом они не превзошли discriminative heads/CNN. На source blocked transfer они чувствительны к split и normal gate. Отдельные преимущества на Hold 50 °F не переносятся на все условия.

Проверка Gram матрицы strong-fault прототипов для blocked seed 0 показала максимальный off-diagonal cosine около 0.484: **не обнаружено оснований объяснять провал просто коллинеарностью всех средних сигнатур**. Меняющееся направление при другой severity, разброс внутри класса и ошибки normal model остаются возможными причинами, а не доказанными механизмами. [Сохранённая матрица](../../results/ashrae_directions_2026-11-04/signature_cosines.csv).

## 8. Следующие приоритеты после этого раунда

1. **Для seen-fault diagnosis:** raw+thermo LDA как основной новый кандидат; проверить label budget и строгий sensor budget, разделить fault signature и file/campaign identity на независимой записи.
2. **Для normal-only:** longer context + dense covariance с normal-reference выбором регуляризации. Сначала решить границы области нормальных условий и target calibration; нет основания заменять это ещё более сложной сетью ради одного среднего AUROC.
3. **Для unseen severity:** сохранить temporal CNN как baseline и проверить physics feature encoder с явным severity disentanglement; не переносить победу LDA из seen SL1 в этот сценарий.
4. **Для предметного сравнения:** полная реплика LFD/FDSD и VMD–KPCA требует точного доступа к preprocessing и архитектуре. Нынешние спектральные/физические адаптации не заменяют её.

Для подтверждающего заявления нужен новый зафиксированный test, independent campaigns и явный alarm budget. На той же просмотренной CSV-копии дополнительный подбор не может сам по себе установить SOTA.

## 9. Воспроизведение

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements-ashrae-extensions.txt
.venv/Scripts/python.exe scripts/run_ashrae_directions.py
.venv/Scripts/python.exe scripts/run_ashrae_diagnosis_directions.py
.venv/Scripts/python.exe scripts/run_ashrae_precision.py
.venv/Scripts/python.exe scripts/run_ashrae_physical_scaling.py
.venv/Scripts/python.exe scripts/analyze_ashrae_matched_windows.py
.venv/Scripts/python.exe scripts/summarize_ashrae_directions.py
.venv/Scripts/python.exe -m unittest discover -s tests
```

Сохранены manifests с hash кода, per-fault/per-class метрики, hyperparameter selections, все score arrays для matched анализа и графики. Большие score archives остаются локально и исключены из Git; они восстанавливаются runner-скриптами. Исходные отчёты первого раунда сохранены.
