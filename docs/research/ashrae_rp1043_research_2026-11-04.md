# ASHRAE RP-1043: исследование, математическая постановка и путь к проверяемому SOTA

**Дата:** 4 ноября 2026. **Два независимых трека:** обучение только на норме и диагностика известных классов неисправностей.

Подготовлены [обзор первичной литературы](ashrae_rp1043_literature_2026-11-04.md), [результаты normal-only](../../results/ashrae_2026-11-04/report.md), [результаты supervised](../../results/ashrae_2026-11-04/supervised/report.md), код загрузки/аудита и воспроизводимые эксперименты. **Достижение SOTA не установлено.** Ниже строго разделены наблюдения, реализованные методы и предлагаемое развитие.

## 1. Что проверено на данных

Скачана доступная CSV-копия из [авторского S-DAGMM репозитория](https://github.com/viettra-xai/S-DAGMM/tree/467961ef98d2c13dbff6e47490f38be1934ceef9/datasets/chiller/complete_dataset), commit `467961ef98d2c13dbff6e47490f38be1934ceef9`. Сохранены 32 файла, их SHA-256, размеры и аудит: [manifest](../../results/ashrae_2026-11-04/data_manifest.json), [audit](../../results/ashrae_2026-11-04/data_audit.csv). Сырые CSV находятся в `data/raw/ashrae_public`, исключённой из Git.

Это производная авторская копия, **не полный официальный архив RP-1043**. [Официальный проектный пакет](https://store.accuristech.com/ashrae/standards/rp-1043-fault-detection-and-diagnostic-fdd-requirements-and-evaluation-tools-for-chillers?product_id=1716217) распространяется отдельно. Оплата и получение платного пакета не выполнялись; открытая копия не позволяет предполагать права на его перераспространение.

Обнаружены следующие свойства именно скачанной копии:

- Каждый CSV: **5191 строка × 66 колонок**. Семь семейств неисправностей, четыре уровня, плюс четыре файла нормы.
- `normal10/20/30/40.csv` **побайтно одинаковы**. В эксперименты входит только `normal10.csv`; четыре файла не считаются четырьмя независимыми нормальными прогонами.
- В каждом файле **3771 повторяющаяся временная метка**. `Time` округлён; восстановить точные интервалы только из него нельзя. В коде сохраняется порядок строк. Задержки в секундах не вычисляются.
- Есть очень малые ненулевые числа порядка `1e-45`, выбросы вроде `TWCI = -99.4`, служебные `Unit Status`, `Active Fault` и стендовые управляющие каналы. Не доказано, что все особенности возникли при одном конкретном способе конвертации; это наблюдения, а не реконструкция истории файлов.
- Метка family/severity задаётся именем файла; **момент физического введения неисправности не восстановлен**. Приписывать каждому такому файлу известный fault onset нельзя.
- На каждый fault/severity приходится одна запись. Даже блоковое разделение не отделяет эффект неисправности от особенностей её конкретной записи; для этого нужны независимые повторные campaigns/runs. Это особенно существенно для supervised seen-SL1 сравнения.

Исторический RP-1043 — лабораторный centrifugal water-cooled chiller порядка 90 refrigeration tons / 316 kW. Разные публикации используют 6, 7 или 8 типов, отфильтрованные steady-state данные либо динамику, разные подвыборки и sampling. Поэтому **описание исходного проекта не заменяет паспорт конкретной копии**. Базовый первоисточник: [Comstock, Braun, Groll, 2001](https://doi.org/10.1080/10789669.2001.10391274).

## 2. Две разные статистические задачи

Пусть \(x_{r,t}\in\mathbb R^d\) — измерения в эксперименте \(r\), \(c_{r,t}\) — внешние условия и уставки, \(f\in\{0,\ldots,7\}\) — класс состояния, \(s\in\{1,2,3,4\}\) — выраженность неисправности.

### 2.1. Normal-only novelty detection

Обучение использует только \(f=0\). Детектор строит \(a_t=a(x_{t-w+1:t},c_{t-w+1:t})\), и правило тревоги

\[
\tau=\inf\{t:a_t>q\}.
\]

Целевая эксплуатационная задача при фиксированном горизонте \(H\):

\[
\max_a\;\min_{f,s,c}\Pr_{f,s,c}(\tau\le H),
\qquad
\sup_{c\in\mathcal C_{\rm admissible}}\Pr_{0,c}(\tau\le H)\le\alpha.
\]

Это формулировка исследовательской цели, **не свойство уже обученных моделей**. Minimax по всем возможным неисправностям без ограничений на альтернативу невозможен: распределение «неисправности» может совпасть с нормальным.

### 2.2. Supervised diagnosis

В обучении известны \(f\), иногда \(s\). Классификатор

\[
\widehat p_\psi(f\mid X,c),\qquad
\widehat f=\arg\max_f\widehat p_\psi(f\mid X,c)
\]

оценивается по macro-F1 и balanced accuracy восьми классов. Дополнительно сообщаются normal FPR и вероятность хотя бы распознать fault как non-normal.

Нужно разделять: **новое проявление известного fault class на SL1**, **новый fault class**, **новый режим** и **новый чиллер**. Обучение SL2–SL4 → проверка SL1 не доказывает обнаружение нового класса. Эту границу иллюстрирует [работа Jin et al. об incipient faults и MC-dropout](https://arxiv.org/abs/1902.06366).

Для balanced classification можно использовать

\[
\mathcal L_{\rm CE}=-\frac1n\sum_i w_{f_i}\log\widehat p_\psi(f_i\mid X_i,c_i),
\quad w_k=\frac{n}{K n_k}.
\]

Такой вес меняет целевое распределение. Полученные вероятности без дополнительной калибровки не являются вероятностями отказа при реальной эксплуатационной редкости.

## 3. Что литература действительно предлагает

Полная карта с источниками и статусом воспроизведения — в [отдельном обзоре](ashrae_rp1043_literature_2026-11-04.md). Ключевые конкурирующие механизмы:

1. **Условные физические остатки.** Сравнивать состояние с нормальным эталоном при той же нагрузке и внешних температурах; использовать направленные физические индексы. Историческая основа — [enhanced sensitivity strategy](https://www.sciencedirect.com/science/article/abs/pii/S1359431111004078).
2. **Динамика и спектр.** [LFD/FDSD, 2025](https://doi.org/10.1016/j.ijrefrig.2025.03.003) — предметный normal-only comparator; [VMD–KPCA, 2026](https://doi.org/10.3390/electronics15010208) — динамическая обработка, но wrapper feature selection использует известный fault. В одном строгом normal-only рейтинге эти версии нельзя смешивать без маркировки.
3. **Неразмеченная загрязнённая история.** [DBSCAN–KPCA–UPCA–EWMA, 2026](https://doi.org/10.1016/j.buildenv.2026.115033) — актуальный прямой comparator. Предположение «доминирующее поведение нормально» требует проверки; оно не эквивалентно известной чистой норме.
4. **Представление и плотность.** [NeuTraL AD](https://proceedings.mlr.press/v139/qiu21a.html), [DAGMM](https://bzong.github.io/doc/iclr18-dagmm.pdf), [conditional flow](https://doi.org/10.1016/j.isatra.2023.09.002), [MTGFlow](https://arxiv.org/abs/2208.02108). Низкая reconstruction error или высокая likelihood не гарантируют исправность при сдвиге домена.
5. **Supervised/few-label diagnosis.** SVM/ансамбли, [S-DAGMM](https://doi.org/10.1016/j.enbuild.2022.111893), MC-dropout, generative augmentation. Их преимущество при наличии fault labels не переносится на normal-only постановку.

Большой Transformer, flow или physics-informed loss сами по себе не достаточны для новизны. Нужна сопоставимая проверка на SL1, переносе режима и ложных тревогах. Порог и способ получения alarm rule являются частью метода.

## 4. Протокол уже выполненного пилота

### 4.1. Признаки и границы данных

Разрешены два context канала `TWE_set`, `TWCI` и 24 измеряемых response канала: температуры контуров и хладагента, мощности/токи, расходы, давления и масляные измерения. Полный allowlist: [ashrae.py](../../src/industrial_novelty/data/ashrae.py). Это лабораторный набор; доступность этих каналов на промышленной BMS отдельно не установлена.

Исключены `Time`, `Unit Status`, `Active Fault`, стендовые valve states, COP, kW/Ton и готовые energy-balance индексы. Исключение задаётся allowlist до просмотра метрик. Расходы `FWC/FWE` остаются откликами: conditioning на них может убрать reduced-flow неисправность из остатка.

Фиксированный sensor-based operating envelope: конечные значения; температуры между −10 и 200 в исходной Fahrenheit шкале; `kW > 5`, `FWC > 10`, `FWE > 10`. Это отсечение неработающего/явно некорректного измерения, **не steady-state detector**. Низкорасходные состояния вне envelope не покрываются выводами.

Окно 12 строк, шаг 6; искусственный блок 180 строк; на обоих краях блока отброшено по 12 строк. Окно не пересекает границы блоков, недопустимые строки и номинальные температурные полосы. StandardScaler каналов оценивается только по fit. Внутри части окна перекрываются; между частями общих исходных строк нет.

### 4.2. Разбиения

- `blocked`: стратификация временных блоков по температурным полосам 40/45/50 °F; непересекающиеся fit, reference, calibration, test. Нормальный test — блоки одной исходной нормальной траектории, а не новый эксперимент.
- `loto40/45/50`: целая температурная полоса удержана из обучения и калибровки. Любой блок, затрагивающий target-полосу, исключён из source fit/reference/calibration. В normal и fault test берутся только окна target-полосы.
- 40/50 °F — крайние режимы для этого фактора; 45 °F — промежуточный. Эти проверки **не являются leave-one-out по всем 27 режимам**: независимые метки нагрузки и полного factorial schedule не восстановлены.
- Seeds 0–4 заданы заранее. Параметры фиксированы, поиска по fault test нет. Seed SD отражает зависимость от разбиения/инициализации, не доверительный интервал по независимым установкам.

В normal-only треке все fit/reference/calibration — норма. Fault файлы только оцениваются. В supervised треке fault labels разрешены при обучении: `seen_sl1` обучается на fit-блоках SL1; `transfer_sl1` — на fit-блоках SL2–SL4. Test в обоих случаях — отдельные блоки SL1, normal test одинаков по конструкции. Это разные информационные бюджеты.

### 4.3. Что реально обучено

**Novelty, 240 fits:** Mahalanobis с Ledoit–Wolf; Isolation Forest; One-Class SVM; novelty LOF; KPCA с RKHS residual; DPCA; CVA; conditional RBF; conditional RBF с tangent projection; AE на средних окон; conditional affine flow; MLP-адаптация NeuTraL AD. Для трёх нейронных вариантов 60 эпох, фиксированные небольшие архитектуры.

**Diagnosis, отдельная серия:** balanced logistic regression; RBF-SVM; Random Forest; Extra Trees; histogram gradient boosting; temporal CNN. Два сценария SL1, четыре протокола, пять seeds. Neural CNN — 30 эпох, baseline architecture, не воспроизведение конкретной статьи TCN.

LFD/FDSD, полный S-DAGMM, MTGFlow, AFNF, Anomaly Transformer, TranAD, VMD–KPCA и DBSCAN–KPCA–UPCA–EWMA **не воспроизведены**. Наши представители семейств не названы репликациями этих моделей. Поиск гиперпараметров и полноценные ablation sweeps остаются следующим этапом на зафиксированном новом test.

### 4.4. Метрики и наблюдения normal-only

Сообщены per-fault/per-severity AUROC, AP с искусственным prior 50/50, recall и отдельный нормальный FPR. Макроусреднение сначала по семи fault families, затем по seeds. Point adjustment отсутствует; его искажения разобраны в [Kim et al., AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/view/20680).

На blocked SL1:

| Метод | Macro AUROC | Recall при номинальном calibration budget 5% | Фактический normal FPR |
|---|---:|---:|---:|
| Mahalanobis | 0.9413 | 65.90% | 10.31% |
| Conditional RBF + tangent | 0.9311 | 60.72% | 7.39% |
| Conditional RBF | 0.9224 | 59.53% | 7.55% |
| Conditional flow | 0.8572 | 52.57% | 6.41% |

Это не recall при одинаковом **достигнутом** FPR. Подтянуть каждый threshold по normal test для красивого равенства FPR означало бы использовать test для калибровки. Такие значения можно показывать только отдельно как ROC operating-characteristic analysis, не как готовую эксплуатационную настройку.

На Hold 50 °F conditional RBF, tangent-версия и flow дают FPR 100%. Mahalanobis сохраняет AUROC 0.8913, но FPR 56.07%. Следовательно, **ни одна из проверенных моделей не решила надёжный мониторинг во всём диапазоне**. Развитие должно устранять эту проблему, а не только улучшать средний AUROC.

Слабая утечка хладагента остаётся сложной: у Mahalanobis blocked SL1 AUROC 0.7968, recall 23.06%. Слабые CF/FWE значительно легче. Нельзя доказывать чувствительность к ранним отказам только общим средним по всем severities.

## 5. Реализованный условный остаток

Разделим \(x=(c,y)\), \(c\in\mathbb R^2\), \(y\in\mathbb R^{24}\). На среднем завершённого окна строится

\[
\widehat\mu(c)=\phi(c)^\top B,\quad
\phi(c)=\big[1,c^\top,e^{-\gamma\|c-u_1\|^2},\ldots,e^{-\gamma\|c-u_M\|^2}\big]^\top,
\]

\[
\widehat B=(\Phi^\top\Phi+\lambda D)^{-1}\Phi^\top Y.
\]

В коде \(M\le64\), \(\lambda=1\), gamma — inverse median nonzero squared distance между fit-centers. Intercept почти не штрафуется. Независимая reference-часть нормы определяет bias остатка и shrinkage covariance:

\[
\widehat\Sigma=(1-\rho)S+\rho\frac{\operatorname{tr}S}{d}I,
\qquad
r=\widehat\Sigma^{-1/2}(y-\widehat\mu(c)-\widehat b),\quad a_{\rm full}=\|r\|^2.
\]

Reference здесь отделён от fit, чтобы не оценивать неопределённость только по оптимистичным resubstitution residuals. Это не решает temporal dependence и out-of-support bias.

Tangent-абляция вычисляет Jacobian \(J=\partial\widehat\mu/\partial c\) центральными разностями, затем \(A=\widehat\Sigma^{-1/2}J\). С \(P_A=AA^\dagger\):

\[
a_\eta=r^\top(I-P_A)r+\eta r^\top P_A r,
\qquad \eta=0.1.
\]

Ранг определяется SVD; повторяющиеся колонки Jacobian не должны удваивать вычитаемую энергию. Это проверяется тестом. При \(\eta=1\) получаем полный остаток, при \(\eta=0\) — полную локальную invariance к nuisance direction. Наше небольшое улучшение не доказывает глобальную invariance и не устраняет экстраполяцию.

## 6. Более сильная математическая постановка для следующей модели

Предлагается исследовать **физически осмысленные условные остатки, локальную идентифицируемость неисправности и калибровку накопленного свидетельства**. Это направление-кандидат, а не уже проверенная end-to-end архитектура и не заявление приоритетной новизны.

### 6.1. Физика: что именно должно быть устойчивым

После проверки единиц и схемы измерений:

\[
Q_e=\dot m_e c_p(T_{e,in}-T_{e,out}),\quad
Q_c=\dot m_c c_p(T_{c,out}-T_{c,in}),
\]

\[
Q_c=Q_e+\eta_m P_{\rm electric}-Q_{\rm loss}.
\]

Для динамики в баланс нужно добавить член накопления энергии \(dU/dt\). Механическое равенство стационарного баланса на start-up создаёт ложные тревоги. \(\eta_m\), losses, теплоёмкости и систематические sensor biases нельзя автоматически считать известными.

**Процессная неисправность обычно тоже подчиняется сохранению энергии.** Поэтому один heat-balance residual не обнаруживает все CF/RL/RO. Нужны параметры эффективности/теплопередачи относительно нормального режима:

\[
UA_e=\frac{Q_e}{\operatorname{LMTD}_e},\quad
UA_c=\frac{Q_c}{\operatorname{LMTD}_c},\quad
COP=\frac{Q_e}{P},\quad
\Delta P=P_c-P_e,
\]

\[
\operatorname{LMTD}(a,b)=\frac{a-b}{\log(a/b)},\qquad a,b>0,
\quad \lim_{b\to a}\operatorname{LMTD}(a,b)=a.
\]

Нужно сверить refrigerant states, saturation pressure relations, знак approach temperatures и допустимость LMTD на каждом режиме. Для near-zero gradients нужна численно устойчивая формула; для недопустимых состояний — явный статус, а не клиппинг до удобного числа.

Физический baseline строит \(h(x)\), а затем \(h(x)-\mathbb E_0[h(x)\mid c]\). Это позволяет обнаруживать изменение \(UA\) даже когда баланс сохраняется. Пилот **не включает эти hard physical features**, поскольку единицы и достоверность derived columns скачанной копии не подтверждены по оригинальному архиву.

### 6.2. Условная вероятностная модель вместо безусловного распределения

Для \(y_t\) и известной до момента решения истории \(\mathcal H_{t-1}\):

\[
p_0(y_t\mid c_t,\mathcal H_{t-1}),\qquad
z_t=f_\theta(y_t;c_t,\mathcal H_{t-1}),
\]

\[
\log p_0(y_t\mid c_t,\mathcal H_{t-1})
=\log p_Z(z_t)+\log|\det D_yf_\theta|.
\]

Условный flow нужен, если Gaussian residual model не описывает распределение достаточно хорошо. Сначала следует проверить условный ridge/GP и heteroscedastic Gaussian; увеличение глубины при нескольких сотнях зависимых окон может ухудшить перенос.

Физику можно включать в параметризацию \(\mu_\theta\), regularized state estimator либо soft penalty на нормальных fit данных. Коэффициенты выбираются на normal development data. Жёстко навязать ошибочное уравнение опаснее, чем иметь простой статистический baseline.

Три принципиальных ограничения:

1. `Cooling Tons`, измеренная мощность и расходы могут быть descendants неисправности. Condition на них может убрать нужный fault signal. Контекст должен задаваться внешними условиями/уставками, либо нужна отдельная причинная модель.
2. Наличие context как входа не доказывает invariance. Вне support обучающих условий likelihood и residuals могут быть ненадёжными; это наблюдалось в пилоте.
3. Слишком гибкий history predictor способен подстроиться под длительный fault и перестать его видеть. Нужны долговременный нормальный эталон и отдельный анализ persistent faults; online adaptation нельзя включать без контроля загрязнения.

### 6.3. Идентифицируемость после исключения nuisance

Рассмотрим локальную Gaussian-модель:

\[
y=\mu_0+J_c\delta c+J_\theta\delta\theta+\varepsilon,
\qquad \varepsilon\sim N(0,\Sigma),
\]

где \(\delta c\) — ошибка/неучтённое возмущение условий, \(\delta\theta\) — физическая неисправность. Если \(c\) наблюдается безошибочно и модель корректна, дополнительная nuisance projection может быть не нужна; её польза относится к остаточной неопределённости контекста.

После whitening положим

\[
r=\Sigma^{-1/2}(y-\mu_0),\quad A=\Sigma^{-1/2}J_c,
\quad B=\Sigma^{-1/2}J_\theta,
\quad \Pi=I-AA^\dagger.
\]

Профильная минимизация \(\min_{\delta c}\|r-A\delta c\|^2\) равна \(r^\top\Pi r\). Эффективный score и информация об отказе:

\[
u=B^\top\Pi r,\qquad
I_{\rm eff}=B^\top\Pi B
=I_{\theta\theta}-I_{\theta c}I_{cc}^{\dagger}I_{c\theta}.
\]

Последнее — Schur complement Fisher information при стандартных локальных условиях. Для многонаправленной альтернативы:

\[
T_{\rm eff}=u^\top I_{\rm eff}^{\dagger}u.
\]

**Следствие 1.** Если \(Bv\in\operatorname{col}(A)\), то \(v^\top I_{\rm eff}v=0\). Это направление неисправности неотличимо от nuisance после проекции. Универсально обещать «убрали режим и сохранили все faults» нельзя.

**Следствие 2.** В oracle Gaussian-модели с фиксированными \(A,B,\Sigma\), под \(H_0\), \(T_{\rm eff}\sim\chi^2_{\operatorname{rank}(\Pi B)}\); под локальной альтернативой noncentrality равна \(\delta\theta^\top I_{\rm eff}\delta\theta\). Это позволяет сравнивать наборы датчиков через минимальное собственное число \(I_{\rm eff}\).

**Граница доказательства.** После оценки Jacobian/ковариации нейросетью по зависимым данным точный chi-square порог не обоснован. Калибровка нужна отдельно. Проекция в пилоте использует \(J_c\), но **не содержит** физическую матрицу \(J_\theta\) и не вычисляет полный efficient score.

Для известной направленной физической сигнатуры \(b_f\) полезен matched statistic

\[
S_f=\frac{b_f^\top\Pi r}{\sqrt{b_f^\top\Pi b_f}},
\]

если знаменатель положителен. Знаки/сигнатуры следует выводить из модели и оценивать только по разрешённому обучающему набору. В normal-only треке нельзя оценить их по закрытому fault test.

### 6.4. Почему накопление помогает слабым fault

Если нормализованный остаток имеет небольшой устойчивый сдвиг \(\delta\), то для независимых шагов Gaussian log-likelihood ratio даёт

\[
\log\Lambda_{k:t}(\delta)=\delta^\top\sum_{j=k}^t r_j
-\tfrac12(t-k+1)\|\delta\|^2.
\]

Профилирование по неизвестному \(\delta\) приводит к scan/GLR:

\[
G_t=\max_{\ell\in\mathcal L}\frac{1}{2\ell}
\left\|\sum_{j=t-\ell+1}^t r_j\right\|^2.
\]

При temporal dependence \(\ell I\) следует заменить ковариацией суммы

\[
\Psi_\ell=\operatorname{Var}_0\left(\sum_{j=t-\ell+1}^t r_j\right),\qquad
G_t=\max_{\ell\in\mathcal L}\tfrac12 S_{t,\ell}^\top\Psi_\ell^{\dagger}S_{t,\ell}.
\]

Оценка \(\Psi_\ell\) требует regularization и достаточной normal history; её можно строить по независимым нормальным сериям или при явно проверенных stationarity/mixing assumptions. Порог должен учитывать поиск по нескольким \(\ell\) и моментам времени. Нельзя использовать отдельный \(\chi^2\) threshold каждого окна и объявлять run-wise контроль.

Постоянный fault onset в скачанной копии неизвестен, поэтому выигрыш GLR по detection delay здесь **не проверен**. EWMA/GLR пока не включены в результат этого нового пилота.

### 6.5. Контроль вероятности тревоги за горизонт

Пусть detector и все преобразования зафиксированы без final calibration. Имеются \(m\) **обменных нормальных траекторий одинакового горизонта** и одна новая траектория. Внутри каждой допустима произвольная зависимость. Для заранее выбранного online statistic \(G_t\) определим

\[
R_i=\max_{t\le H}G_{i,t},\qquad
k=\lceil(m+1)(1-\alpha)\rceil,
\quad q=R_{(k)},
\]

а при \(k>m\) положим \(q=+\infty\). Тогда для строгого правила \(G_t>q\):

\[
\Pr_0(\exists t\le H:G_{{\rm new},t}>q)\le\alpha.
\]

**Доказательство.** Событие тревоги равно \(R_{\rm new}>R_{(k)}\). При обменности ранг нового максимума среди \(m+1\) максимумов равномерен при отсутствии ties; strict comparison при ties консервативен. Вероятность верхних \(m+1-k\) рангов не больше \(\alpha\). Независимость окон не использовалась. Это стандартная split-conformal rank логика; см. [Bates et al.](https://arxiv.org/abs/2104.08279).

**Требования:** одна схема горизонта/окон/перезапуска статистики; calibration и test из одного распределения траекторий; никакого выбора лучшего детектора по этим же calibration maxima. Если требуется выбрать модель, сначала development selection, затем новая final calibration. Максимумы коротких и длинных прогонов не взаимозаменяемы.

Числа разрешения: для конечного порога при \(\alpha=1\%\) нужно минимум 99 calibration trajectories, при 5% — 19. Это маргинальная гарантия. Для более сильного требования «с уверенностью 95% условный FPR при данном calibration set не выше 1%» простой max-threshold требует примерно

\[
m\ge\frac{\log(0.05)}{\log(0.99)}\approx298.07,
\quad\text{то есть }299
\]

независимых калибровочных единиц при обычной continuous/i.i.d. постановке; это отдельная tolerance-bound задача. Повторное использование одной нормы под четырьмя именами размер выборки не увеличивает.

**В скачанной копии условия theorem не выполнены:** одна нормальная trajectory разрезана на блоки; при LOTO также есть covariate shift. Наш ранговый threshold по окнам — лишь воспроизводимая эмпирическая процедура. В LOTO при 1% даже числа calibration windows недостаточно для конечного порога. Существующий [run-wise calibration модуль](../../src/industrial_novelty/eval/calibration.py) пригодится после получения подходящих исходных серий, но переименование блоков в `run` не создаёт гарантию.

### 6.6. Что меняется при сдвиге домена

Если trajectory distributions \(P\) и \(Q\) отличаются и \(\operatorname{TV}(P,Q)\le\varepsilon\), для фиксированного calibration rule

\[
\Pr_Q(\text{alarm})\le\alpha+\varepsilon
\]

после усреднения по calibration из \(P\). Это прямое свойство total variation, не способ получить \(\varepsilon\) из воздуха. Для unsupported температурного режима такой bound обычно неизвестен.

Практическое решение: отдельный статус **«режим вне области обучения»**, выделенная normal target calibration при доступности и раздельный отчёт по supported/unsupported conditions. Отказ от решения не должен улучшать показатели скрытно: сообщать coverage, false alarms и missed faults по всем случаям, включая abstentions.

Не следует перемножать обычные marginal conformal p-values зависимых окон и обещать anytime-valid e-process. Для этого нужна conditional superuniformity/корректный likelihood-ratio martingale либо отдельная доказанная construction. Конечногоризонтная калибровка максимумов математически проще и честнее для этого проекта.

## 7. Что нужно для научной статьи и обоснованного SOTA

### 7.1. Подтверждающий benchmark

Нужен оригинальный data dictionary, точные temporal/run IDs и mapping 27 operating conditions; желательно несколько независимых normal runs. Если повторов нет в самом RP-1043, ограничить выводы внутристендовым benchmark и добавить независимый chiller dataset/реальный объект. Один стенд не доказывает перенос между чиллерами.

Разделить заранее:

1. Same-condition held-out runs, отдельно steady и transient.
2. Leave-one-operating-condition/group-out без windows, общих между частями.
3. Supervised unseen SL1 и leave-one-fault-family-out как **разные** сценарии.
4. Contaminated training 0/1/5/10% при фиксированных известных условиях; это отдельный трек.
5. Field sensor budget, sensor dropout/noise и физически допустимый дрейф.

Временной embargo выбирается с учётом autocorrelation и support всех фильтров, не только ширины окна. Любой VMD/FFT/normalization должен обрабатываться в пределах разрешённой части; centered filtering через границу split создаёт утечку. Online detector использует только историю до текущего решения.

### 7.2. Критерий успеха

Primary novelty endpoint: macro detection probability на SL1 при заранее заданном **run/horizon false-alarm budget**; delay с censoring необнаруженных fault runs. AUROC, balanced AP и latency — secondary. Average delay только среди detected runs создаёт selection bias.

Primary diagnosis endpoint: macro-F1 по восьми классам на unseen runs/conditions; per-class recall и confusion matrix; отдельные normal FPR, calibration и неизвестные классы. Энергетический эффект/maintenance cost можно добавить после получения достоверной шкалы тяжести.

Сравнение должно быть paired по одним экспериментальным единицам. Bootstrap/test по независимым runs или обоснованным clusters; не считать перекрывающиеся окна независимыми. Пять seeds на одном test — не пять новых datasets. Для нескольких сравнений заранее определить primary comparisons и поправку на множественность. Результаты текущего пилота уже просмотрены: их test нельзя переиспользовать как нетронутый confirmatory test после настройки.

### 7.3. Проверяемые гипотезы и абляции

| Гипотеза | Сравнение | Что опровергает гипотезу |
|---|---|---|
| Условный физический эталон устойчивее статистического среднего | Mahalanobis против физического residual Gaussian при одном alarm budget | Нет выигрыша SL1 или рост FPR на held-out regime |
| Nuisance projection сохраняет различимую fault информацию | full residual / tangent / regularized tangent; оценка \(I_{eff}\) | Fault сигнатуры совпали с nuisance subspace или пропала чувствительность RL/RO |
| Flow нужен сверх Gaussian | Один encoder/context/features, Gaussian против flow | Нет устойчивого выигрыша после normal-only настройки |
| Накопление улучшает incipient detection | Instant / EWMA / multiscale GLR | Дольше delay либо больше horizon false alarms при равной калибровке |
| Физические residual features улучшают diagnosis при нехватке меток | Одинаковый SVM/linear head на raw features и residual features | Нет улучшения macro-F1 на unseen SL1/conditions |
| Полученный эффект обобщается | Independent chiller / новый campaign | Выигрыш остаётся только на одном преобразованном RP-1043 наборе |

Составная модель, которая улучшает только AUROC на уже просмотренной CSV-копии, не достаточна для SOTA. Научно сильный результат здесь — измеримое улучшение на слабых fault при явной области применимости, анализе неидентифицируемых направлений и подтверждённом контроле тревог.

## 8. Артефакты и воспроизведение

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements-ashrae.txt
uv pip install --python .venv/Scripts/python.exe -r requirements-neural.txt
.venv/Scripts/python.exe scripts/prepare_ashrae_public.py
.venv/Scripts/python.exe scripts/run_ashrae_benchmark.py
.venv/Scripts/python.exe scripts/summarize_ashrae_benchmark.py
.venv/Scripts/python.exe scripts/run_ashrae_supervised.py
.venv/Scripts/python.exe scripts/summarize_ashrae_supervised.py
.venv/Scripts/python.exe -m unittest discover -s tests
```

CPU, фиксированные seeds и deterministic torch; параметры и hash кода сохранены. Poetry lock исходного проекта не изменён; дополнительные зависимости вынесены отдельно. Предыдущие эксперименты TEP и их результаты сохранены.

Код: [загрузка и аудит](../../scripts/prepare_ashrae_public.py), [схема/разбиения](../../src/industrial_novelty/data/ashrae.py), [условные модели](../../src/industrial_novelty/models/chiller.py), [novelty runner](../../scripts/run_ashrae_benchmark.py), [diagnosis runner](../../scripts/run_ashrae_supervised.py), [проверки границ и математики](../../tests/test_ashrae.py).
