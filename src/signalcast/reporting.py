"""Build publication artifacts only from saved experimental evidence."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data import load_panel
from .figures import plot_real_cases, plot_detection, case_explanation, DETECTOR_NAMES, RISK_NAMES

DISPLAY={"naive":"Последний месяц","seasonal_naive":"Прошлый год","seasonal_growth":"Сезонность + рост",
         "panel_shrink":"Панельное сглаживание","ridge_panel":"Ridge + календарь","catboost":"CatBoost",
         "lightgbm":"LightGBM","lightgbm_news":"LightGBM + новости","prophet_default":"Prophet default",
         "prophet_monthly":"Prophet monthly","chronos_raw":"Chronos исходный","chronos_seasonal":"Chronos + сезонность",
         "signal_ensemble":"SIGNAL","signal_adaptive":"Выбор двух экспертов",
         "pooled_yoy_damped":"Общий годовой рост","seasonal_ratio":"Сезонные отношения","damped_trend":"Затухающий тренд"}
COLORS={"signal_adaptive":"#7da998","signal_ensemble":"#007d66","prophet_default":"#ba634b","prophet_monthly":"#dbad77",
        "chronos_seasonal":"#4c6bc1","lightgbm":"#67a698","catboost":"#63858c"}


def read_predictions(out):
    files=sorted(Path(out).glob("predictions_h*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files],ignore_index=True) if files else pd.read_parquet(Path(out)/"predictions.parquet")


def md_table(df):
    return df.to_markdown(index=False,floatfmt=".3f")


def build_report(cfg):
    out=Path(cfg["results_dir"]);docs=Path("docs");fig=docs/"figures";fig.mkdir(exist_ok=True,parents=True)
    metrics=pd.read_csv(out/"metrics.csv");paired=pd.read_csv(out/"metrics_paired_prophet.csv")
    audit=json.loads((out/"data_audit.json").read_text(encoding="utf-8"))
    pred=read_predictions(out)
    category=cfg["primary_category"]
    main_model="signal_ensemble"
    test=paired[(paired.split=="test")&(paired.category==category)]
    full=metrics[(metrics.split=="test")&(metrics.category==category)]
    table=test.pivot(index="model",columns="horizon",values="mae").sort_values(1)
    table.index=[DISPLAY.get(x,x) for x in table.index]
    table=table.reset_index().rename(columns={"index":"Модель"})
    gains=[]
    for h in cfg["horizons"]:
        g=test[test.horizon==h].set_index("model")
        a=g.loc[main_model];b=g.loc["prophet_default"];c=g.loc["prophet_monthly"]
        gains.append({"horizon":h,"mae_signal":float(a.mae),"mae_prophet_default":float(b.mae),
                      "mae_prophet_monthly":float(c.mae),"improvement_default_percent":float(100*(1-a.mae/b.mae)),
                      "improvement_monthly_percent":float(100*(1-a.mae/c.mae)),"n":int(a.n),"target_months":int(a.target_months)})
    (out/"headline.json").write_text(json.dumps(gains,ensure_ascii=False,indent=2),encoding="utf-8")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                         "figure.facecolor":"#fbfbf7","axes.facecolor":"#fbfbf7","axes.titleweight":"bold","savefig.facecolor":"#fbfbf7"})
    # Paired forecast comparison: shared records, no mixing of sample sizes.
    names=["seasonal_naive","seasonal_growth","lightgbm","lightgbm_news","pooled_yoy_damped",
           "chronos_seasonal","prophet_default","prophet_monthly","signal_ensemble","signal_adaptive"]
    f,axes=plt.subplots(2,2,figsize=(12,7.2))
    for ax,h in zip(axes.ravel(),cfg["horizons"]):
        g=test[(test.horizon==h)&test.model.isin(names)].sort_values("mae",ascending=False)
        ax.barh([DISPLAY.get(m,m) for m in g.model],g.mae,color=[COLORS.get(m,"#bac8cb") for m in g.model])
        ax.set_title(f"Горизонт {h} мес. | {int(g.iloc[0].target_months)} целевых мес.",loc="left")
        ax.set_xlabel("MAE, руб. (меньше - лучше)");ax.grid(axis="x",alpha=.15);ax.set_axisbelow(True)
    f.tight_layout();f.savefig(fig/"forecast_comparison.png",dpi=170);plt.close(f)
    panel=load_panel(Path(cfg["data_dir"])/"consumption.parquet")
    economy=[]
    for j,c in enumerate(panel.categories):
        complete=np.isfinite(panel.values[:,j,:]).all(1)
        values=panel.values[complete,j,:]
        annual_change=100*(values[:,12:].mean(1)/values[:,:12].mean(1)-1)
        economy.append({"category":c,"complete_municipalities":int(complete.sum()),
                        "median_annual_change_percent":float(np.median(annual_change)),
                        "p10_annual_change_percent":float(np.quantile(annual_change,.1)),
                        "p90_annual_change_percent":float(np.quantile(annual_change,.9)),
                        "median_december_november_2024_percent":float(np.median(100*(values[:,-1]/values[:,-2]-1)))})
    economy=pd.DataFrame(economy)
    economy.to_csv(out/"economic_summary.csv",index=False)
    f,ax=plt.subplots(figsize=(10,3.5))
    for j,c in enumerate(panel.categories):
        y=np.nanmedian(panel.values[:,j,:],axis=0)
        ax.plot(panel.dates,100*y/y[0],label=c,lw=2)
    ax.set_ylabel("Медиана по МО, январь 2023 = 100");ax.legend(ncol=3,fontsize=8,frameon=False)
    ax.grid(alpha=.15);f.tight_layout();f.savefig(fig/"category_paths.png",dpi=170);plt.close(f)
    # Select cases by error rank; retain gaps and every observed point.
    p=pred[(pred.model==main_model)&(pred.horizon==1)&(pred.category==category)&(pred.split=="test")]
    cases=plot_real_cases(panel,p,category,cfg["test_start"],fig/"real_cases.png")
    pd.DataFrame(cases).to_csv(out/"case_studies.csv",index=False)
    detection=pd.read_csv(out/"detection_metrics.csv")
    det=detection.groupby("model")[["recall","precision","f1","false_alarms_per_100","median_delay_detected"]].mean().reset_index()
    plot_detection(detection,fig/"detection.png")
    risk=pd.read_csv(out/"early_warning_metrics.csv")
    risk_summary=risk.groupby("model")[["average_precision","roc_auc","brier","prevalence"]].mean().reset_index()
    risk_summary=risk_summary.set_index("model").loc[list(RISK_NAMES)].reset_index()
    news_metrics=full[full.model.isin(["lightgbm","lightgbm_news"])].pivot(index="horizon",columns="model",values="mae")
    news_metrics["news_improvement_percent"]=100*(1-news_metrics.lightgbm_news/news_metrics.lightgbm)
    selected=json.loads((out/"selected_detector.json").read_text(encoding="utf-8")) if (out/"selected_detector.json").exists() else {}
    text=f'''# Результаты воспроизводимого эксперимента

Эксперимент: 01.10.2026. Обновление представления результатов: 02.10.2026. Числа автоматически рассчитаны из `{cfg['results_dir']}`.

## Прогноз: парное сравнение с Prophet

Основная цель: «{category}». В таблице MAE в рублях, меньше - лучше. Все строки внутри горизонта оцениваются на одних и тех же наблюдениях заранее выбранных 512 МО. Наличие пропусков уменьшает число доступных пар.

{md_table(table)}

Размеры выборки и относительное изменение MAE:

{md_table(pd.DataFrame(gains))}

Положительное improvement означает меньшую ошибку SIGNAL; отрицательное означает проигрыш. Годовой горизонт имеет только один старт и один целевой месяц. Нельзя интерпретировать его как многолетнюю проверку.

Основной вариант - исходный ансамбль `signal_ensemble`. Дополнительный `signal_adaptive` не улучшил его MAE на горизонтах 1,3,6, хотя оказался лучше на единственном годовом эпизоде. Это пример пользы усреднения различных ошибок: выбор двух экспертов по короткой прошлой истории может быть нестабильным. Основной вариант оставлен после сравнительного исследования; отдельного нового теста после этого выбора нет. Числа являются измерениями каждого зафиксированного метода в опубликованном backtest, а не независимой оценкой процедуры итогового выбора.

![Сравнение прогнозов](figures/forecast_comparison.png)

## Основной алгоритм на всей доступной панели

{md_table(full[full.model==main_model][['horizon','n','mae','r2','wape','target_months']])}

Результаты всех категорий и development-периода: `results/full/metrics.csv`. Интервалы парных разностей и доля выигранных МО: `paired_comparisons.json`. Оценка независимой временной неопределённости ограничена тремя тестовыми месяцами. Интервал для h=12 не вычисляется.

## Категории и реальные примеры

![Динамика категорий](figures/category_paths.png)

На графике индексируется медиана расходов каждой категории. Это описание изменения номинальных расходов, а не эффект определённой политики. Различия категорий обосновывают их отдельное моделирование.

{md_table(economy)}

Для этой описательной таблицы используются полные ряды каждой категории. Годовое изменение - отношение среднего расхода за 2024 год к среднему за 2023 внутри одного МО; затем берутся медиана и перцентили между МО. Это невзвешенная характеристика территорий, а не темп роста потребления всего населения России. Размах 10-90% показывает неоднородность локальной динамики. Декабрьское изменение к ноябрю помогает увидеть сезонность, которую простая экстраполяция тренда пропускает. Инфляция и переход к безналичным платежам отдельно не идентифицированы.

![Реальные примеры](figures/real_cases.png)

{md_table(pd.DataFrame(cases).rename(columns={"territory_id":"МО","selection_rule":"Правило выбора","observations":"Месяцев с фактом","history_months":"Месяцев в панели","test_months":"Оценённых прогнозов","test_mae_h1":"MAE, ₽"}))}

{case_explanation(cases)} На графике факт показан точками и линиями; разрывы соответствуют отсутствующим наблюдениям. Штриховка обозначает пропуски, зелёная область - период оценки. Масштабы осей Y различаются.

Примеры выбраны по рангу средней абсолютной ошибки одношагового прогноза: медианный, 90-й перцентиль и максимальный. Выбор для иллюстрации использует test и не является отдельной проверкой качества. Территории обозначены исходными id; справочник названий в данных отсутствует.

## Абляция реальных новостей

{md_table(news_metrics.reset_index())}

Сравниваются одинаковые LightGBM с/без доступных на дату старта новостных признаков. Улучшение может быть отрицательным. Это проверка небольшого корпуса официальных федеральных публикаций; перенос результата на широкий поток местных новостей не доказан.

## Детекторы: только полусинтетические события

Рабочий метод, выбранный на отдельной калибровочной группе: `{selected.get('model','not selected')}`. В таблице средние по 27 сценариям (3 амплитуды × 3 типа × 3 повтора) на других МО.

{md_table(det)}

![Чувствительность детекторов](figures/detection.png)

Recall и precision относятся к внесённым изменениям. Контрольные ряды остаются реальными и могут содержать собственные изменения. Частоты ложных тревог близки, но не идентичны; сравнение не выдаётся за тест при строго равной FAR. Задержка посчитана только для найденных событий.

## Предупреждение на месяц вперёд

{md_table(risk_summary)}

Это макросреднее по трём месяцам, а не метрики одного объединённого пула. Цель - крупное относительное изменение следующего месяца, **статистический proxy**, не подтверждённый экономический шок. Average precision сравнивается с распространённостью события; Brier - ошибка вероятностного прогноза.

Файл `preannounced_events.json` отдельно содержит календарные предупреждения из публикаций до будущего заседания. Эти записи знают дату события, но не его будущий исход.

## Аудит

* Наблюдений: {audit['rows_observed']:,}; уникальных МО: {audit['municipalities']}; категорий: {len(audit['categories'])}.
* Месяцев: {audit['months']}; полных МО для детекторов: {audit['complete_territories']}.
* Пропущенных ячеек панели: {audit['missing_cells']:,}; пропущенные факты не оцениваются.
* SHA-256 consumption.parquet: `{audit['sha256']}`.
* СберИндекс, CC BY-SA 4.0. Новости: Пресс-служба Банка России. Источники и ограничения подробно перечислены в [методологии](METHODOLOGY.md).

Оценки относятся к этому снимку данных и описанному протоколу. Ограничения временной проверки и интерпретации сигналов приведены в [методологии](METHODOLOGY.md).
'''
    (docs/"RESULTS.md").write_text(text,encoding="utf-8")
    presentation(cfg,gains,test,full,det,risk_summary,news_metrics,cases,audit,selected,main_model,economy)


def presentation(cfg,gains,test,full,det,risk,news_metrics,cases,audit,selected,main_model,economy):
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import Paragraph,Table,TableStyle
    from reportlab.lib.styles import ParagraphStyle
    fontdir=Path(matplotlib.get_data_path())/"fonts/ttf"
    pdfmetrics.registerFont(TTFont("DVS",str(fontdir/"DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DVS-Bold",str(fontdir/"DejaVuSans-Bold.ttf")))
    W,H=960,540
    canvas=Canvas("docs/PRESENTATION.pdf",pagesize=(W,H),invariant=1)
    canvas.setTitle("SIGNAL - прогноз расходов и ранние сигналы изменений")
    canvas.setAuthor("SIGNAL project contributors")
    navy="#19323b";green="#007d66";muted="#60747a";page_number=0
    def para(text,x,y,width,size=15,color=navy,bold=False,leading=None):
        st=ParagraphStyle("p",fontName="DVS-Bold" if bold else "DVS",fontSize=size,leading=leading or size*1.45,textColor=HexColor(color))
        p=Paragraph(text,st);_,height=p.wrap(width,H);p.drawOn(canvas,x,y-height);return y-height
    def page(title,subtitle=None):
        nonlocal page_number
        if page_number:canvas.showPage()
        page_number+=1;canvas.setFillColor(HexColor("#fbfbf7"));canvas.rect(0,0,W,H,fill=1,stroke=0)
        canvas.setFillColor(HexColor(green));canvas.rect(42,H-37,34,4,fill=1,stroke=0)
        para("SIGNAL / СБЕРИНДЕКС 2026",86,H-27,600,9,muted)
        para(title,42,H-65,880,26,bold=True,leading=31)
        if subtitle:para(subtitle,42,H-111,880,11,muted)
        canvas.setStrokeColor(HexColor("#d7dfda"));canvas.line(42,38,W-42,38)
        para("Воспроизводимый эксперимент · данные 2023-2024 · подготовлено 01.10.2026",42,27,820,8,muted)
        para(f"{page_number:02}",886,27,35,8,muted)
    def picture(name,x,y,width,height):
        canvas.drawImage(str(Path("docs/figures")/name),x,y,width=width,height=height,preserveAspectRatio=True,anchor="c",mask="auto")
    def table(headers,rows,x,y,width,font=10,padding=8):
        style=ParagraphStyle("cell",fontName="DVS",fontSize=font,leading=font*1.3,textColor=HexColor(navy))
        data=[[Paragraph(str(v),style) for v in r] for r in [headers]+rows]
        widths=[width/len(headers)]*len(headers)
        if len(headers)>2:widths=[width*.34]+[width*.66/(len(headers)-1)]*(len(headers)-1)
        t=Table(data,colWidths=widths)
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),HexColor("#dceee7")),("VALIGN",(0,0),(-1,-1),"TOP"),
                              ("BOTTOMPADDING",(0,0),(-1,-1),padding),("TOPPADDING",(0,0),(-1,-1),padding),
                              ("LINEBELOW",(0,0),(-1,-1),.4,HexColor("#d7dfda"))]))
        _,hh=t.wrap(width,H);t.drawOn(canvas,x,y-hh);return y-hh
    page("Прогнозировать расходы. Раньше замечать изменения.","Прогноз расходов муниципалитетов и отдельная оценка риска изменений")
    para("24 месяца наблюдений.<br/>Общие закономерности тысяч территорий.",42,370,800,31,bold=True,leading=42)
    for k,g in enumerate(gains):
        x=42+k*224
        para(f"{g['horizon']} мес.",x,245,205,13,muted)
        para(f"{g['mae_signal']:,.0f} ₽".replace(","," "),x,216,205,27,green,True)
        direction="ниже" if g['improvement_default_percent']>=0 else "выше"
        para(f"MAE SIGNAL<br/>На {abs(g['improvement_default_percent']):.1f}% {direction}, чем Prophet default",x,173,205,11)
    para("Парное сравнение на фиксированной выборке 512 МО. Для горизонта 12 месяцев - только один проверяемый эпизод.",42,94,865,11,muted)
    page("Что именно прогнозируется","Средние безналичные расходы жителей МО в текущих рублях; основной показатель - «Все категории»")
    para(f"{audit['municipalities']} МО<br/>{audit['rows_observed']:,} строк<br/>6 категорий<br/>24 месяца".replace(","," "),42,362,270,24,bold=True,leading=33)
    para("Не все МО наблюдаются каждый месяц. Пропуски исключены из оценки. Пять подкатегорий не дают полный итог. Территории обозначены исходными id.",42,211,270,13)
    picture("category_paths.png",333,107,588,286)
    page("Экономическая динамика неоднородна","Средний расход 2024 года к среднему за 2023 год; распределение изменений по 2 016 МО с полной историей")
    table(["Категория","Медиана","10-й перцентиль","90-й перцентиль"],
          [[r.category,f"{r.median_annual_change_percent:+.1f}%",f"{r.p10_annual_change_percent:+.1f}%",f"{r.p90_annual_change_percent:+.1f}%"] for r in economy.itertuples()],42,379,870,12)
    para("Темпы различаются: маркетплейсы растут быстрее остальных категорий. Это номинальные расходы; вклад цен, числа покупок и перехода к безналичной оплате здесь не разделён.",42,120,860,13)
    page("Как исключается доступ к будущим данным","В каждой точке расчёта используются только уже наблюдавшиеся расходы и опубликованные новости")
    steps=[("01", "История до даты T", "Расходы, категории и только опубликованные новости."),
           ("02", "Прогнозы моделей", "Сезонные модели, бустинг и Chronos на 1/3/6/12 мес."),
           ("03", "Известные ошибки", "Веса меняются после появления факта и оценки ошибки."),
           ("04", "Проверка и сигналы", "MAE и R²; изменения ряда; риск следующего месяца.")]
    for i,(num,title,body) in enumerate(steps):
        x=42+i*224;para(num,x,363,195,28,green,True);para(title,x,310,195,17,bold=True);para(body,x,257,194,13)
    para("Тестовые факты: октябрь-декабрь 2024. При последовательном обновлении прошлый тестовый месяц становится допустимой историей для следующего прогноза.",42,112,860,12,muted)
    page("Результат на одинаковых наблюдениях","MAE в рублях, меньше - лучше. default: стандартный Prophet; monthly: явно заданная годовая сезонность")
    compare=test.pivot(index="model",columns="horizon",values="mae")
    chosen=[main_model,"prophet_default","prophet_monthly","seasonal_naive","seasonal_growth",
            "panel_shrink","chronos_seasonal","lightgbm","lightgbm_news","signal_adaptive"]
    table(["Модель","1 мес.","3 мес.","6 мес.","12 мес."],
          [[DISPLAY.get(m,m)]+[f"{compare.loc[m,h]:.0f}" for h in cfg["horizons"]] for m in chosen],42,389,870,11,5)
    para("Общие наблюдения выборки 512 МО. SIGNAL лучше показанных моделей на 1/3/6 мес.; на годовом эпизоде лучше LightGBM с новостями. Параметры baseline организатора неизвестны.",42,82,860,11,muted)
    page("Точность на всей панели и границы проверки","Результаты SIGNAL на всех доступных МО; отдельные размеры выборки по каждому горизонту")
    g=full[full.model==main_model].sort_values("horizon")
    table(["Горизонт, мес.","Наблюдений","MAE, ₽","R²","WAPE"],[[int(r.horizon),int(r.n),f"{r.mae:.1f}",f"{r.r2:.3f}",f"{100*r.wape:.2f}%"] for r in g.itertuples()],42,375,870,12)
    para("Здесь больше МО, чем в сравнении с Prophet, поэтому MAE отличается. Для 1/3/6 месяцев проверены три целевых месяца; для 12 месяцев - один. Высокий R² частично отражает различия уровней расходов между МО.",42,161,860,14)
    page("Новости помогают не на каждом горизонте","Вся доступная панель; настройки LightGBM одинаковы. Отличаются только признаки опубликованных новостей")
    table(["Горизонт, мес.","Без новостей, MAE ₽","С новостями, MAE ₽","Снижение MAE"],[[h,f"{r.lightgbm:.1f}",f"{r.lightgbm_news:.1f}",f"{r.news_improvement_percent:+.2f}%"] for h,r in news_metrics.iterrows()],42,378,870,12)
    para("Плюс означает улучшение, минус - ухудшение. Использованы 17 релизов Банка России с датами публикации. Эти федеральные события общие для всех МО; большой размер панели не заменяет длинную историю новостей.",42,166,860,13)
    page("Четыре метода обнаружения изменений","Изменения внесены в реальные ряды. Калибровка: 672 МО; проверка: другие 1 344 МО. Будущие значения не используются")
    picture("detection.png",40,146,880,265)
    para(f"Выбран по F1 на калибровке: байесовский BOCPD. Изменения: ±5%, ±10%, ±20%. Событие засчитывается, если обнаружено в месяц начала или в следующие два месяца.",42,106,860,12)
    page("Детекторы: чувствительность и ложные тревоги","Среднее по 27 сценариям: 3 типа изменения × 3 амплитуды × 3 повтора. Только искусственно внесённые события")
    table(["Метод","Полнота","Точность","F1","Тревог / 100"],[[DETECTOR_NAMES[r.model],f"{r.recall:.3f}",f"{r.precision:.3f}",f"{r.f1:.3f}",f"{r.false_alarms_per_100:.2f}"] for r in det.itertuples()],42,374,870,11)
    para("Полнота - доля найденных внесённых событий; точность - доля тревог, совпавших с такими событиями. Последний столбец: тревоги на 100 контрольных ряд-месяцев. Реальные экономические шоки не размечены; сигнал требует проверки аналитиком.",42,169,860,13)
    page("Риск крупного изменения следующего месяца","Средние метрики по октябрю-декабрю 2024. Это прогноз до события, отдельно от обнаружения уже начавшегося скачка")
    table(["Вход модели","AP ↑","ROC AUC ↑","Brier ↓"],[[RISK_NAMES[r.model],f"{r.average_precision:.3f}",f"{r.roc_auc:.3f}",f"{r.brier:.4f}"] for r in risk.itertuples()],42,374,870,12)
    para("Метка: крупное изменение относительно общей динамики МО; порог задан по 2023 году. AP оценивает поиск редких событий, Brier - ошибку вероятности. Прирост от новостей мал; экономический шок этой меткой не подтверждается.",42,204,860,13)
    para("Календарный сигнал: 07.06.2024 Банк России сообщил о возможном повышении ставки на следующем заседании 26.07.2024. Это предупреждение о риске, а не известный заранее исход.",42,114,860,12,green)
    page("Реальные примеры: точность и пропуски","Прогноз на один месяц. У каждого примера показано число доступных фактов и оценённых прогнозов; масштабы осей Y различаются")
    picture("real_cases.png",30,137,900,263)
    para(case_explanation(cases),42,127,860,12)
    page("Результат, который можно проверить","Код, данные с лицензией, конфигурации, сохранённые прогнозы и методологический отчёт входят в проект")
    para("Один протокол - все модели",42,365,420,22,bold=True)
    para("Сравнение на общих наблюдениях; веса по известным ошибкам; раздельные группы для выбора и проверки детекторов; сравнение с новостями и без них; прогнозы по каждому МО.",42,315,410,16)
    para("Границы применимости",510,365,410,22,bold=True)
    para("Всего 24 месяца, один годовой эпизод. Нет дат публикации расходов и разметки шоков. Пересечение с предобучением Chronos неизвестно. Нового теста после выбора модели нет.",510,315,405,16)
    para("Источник: СберИндекс, CC BY-SA 4.0. Новости: Пресс-служба Банка России. Методология, ссылки, точные версии и команды воспроизведения - в README и docs/METHODOLOGY.md.",42,132,860,13,muted)
    canvas.save()
