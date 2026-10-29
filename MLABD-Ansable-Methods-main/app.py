import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Калькулятор стоимости недвижимости (Zingat)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 1.6rem; font-weight: 600; margin-bottom: 0.5rem; color: #1e293b; }
    .sub-text { color: #64748b; font-size: 0.95rem; margin-bottom: 1.5rem; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; color: #0f172a; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_pipeline():
    try:
        return joblib.load('real_estate_model_pipeline.joblib')
    except FileNotFoundError:
        st.error("Файл модели 'real_estate_model_pipeline.joblib' не найден в текущей директории.")
        return None
    except Exception as e:
        st.error(f"Ошибка при инициализации модели: {e}")
        return None

artifact = load_model_pipeline()
if artifact is None:
    st.stop()

model = artifact['model']
preprocessor = artifact['preprocessor']
cat_cols = artifact['cat_cols']

ohe = preprocessor.named_transformers_['cat']
categories_map = {col: sorted(list(cats)) for col, cats in zip(cat_cols, ohe.categories_)}

# Контрольные метрики модели на тестовой выборке.
MODEL_MAE = 110771.32   # в TL.
MODEL_MAPE = 24.54      # в %.
MODEL_R2 = 0.76         # log scale.

with st.sidebar:
    st.markdown("### Zingat Engine")
    st.caption("Модуль автоматизированной оценки жилой недвижимости (Турция).")
    st.divider()
    
    st.markdown("**Характеристики текущей модели:**")
    st.write(f"• Алгоритм: `HistGradientBoosting`")
    st.write(f"• Качество $R^2$: `{MODEL_R2:.2f}`")
    st.write(f"• Средняя погрешность (MAPE): `{MODEL_MAPE:.1f}%`")
    st.write(f"• Погрешность MAE: `±{MODEL_MAE:,.0f} TL`")
    st.divider()
    
    st.caption("Обучающая выборка: 280 180 объектов сегмента продажи (Satılık), номинированных в турецких лирах (TRY).")

# НАВИГАЦИЯ ПО РАЗДЕЛАМ
tabs = st.tabs(["Расчет стоимости", "Аналитика и метрики", "О проекте"])

# 4.1 ФОРМА ПРОГНОЗА СТОИМОСТИ
with tabs[0]:
    st.markdown('<div class="main-header">Расчет рыночной цены квартиры</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">Введите физические и адресные параметры объекта для получения оценки на базе градиентного бустинга.</div>', unsafe_allow_html=True)

    with st.form("valuation_form"):
        col_loc, col_flat, col_bld = st.columns(3)

        with col_loc:
            st.markdown("**1. Локация и тип фонда**")
            cities = categories_map.get('city', ['Istanbul', 'Other'])
            city = st.selectbox("Город", cities, index=cities.index('Istanbul') if 'Istanbul' in cities else 0)

            districts = categories_map.get('district_grouped', ['Other'])
            district = st.selectbox("Район", districts, index=districts.index('Kadikoy') if 'Kadikoy' in districts else 0)

            sub_types = categories_map.get('sub_type', ['Daire'])
            sub_type = st.selectbox("Тип объекта", sub_types, index=0)

            heatings = categories_map.get('heating_type', ['Yok'])
            heating = st.selectbox("Отопление", heatings, index=heatings.index('Kombi (Doğalgaz)') if 'Kombi (Doğalgaz)' in heatings else 0)

        with col_flat:
            st.markdown("**2. Площадь и планировка**")
            size = st.number_input("Общая площадь, м²", min_value=15.0, max_value=800.0, value=90.0, step=5.0)
            bedrooms = st.number_input("Количество спален", min_value=1, max_value=8, value=2, step=1)
            living_rooms = st.number_input("Количество гостиных (залов)", min_value=0, max_value=4, value=1, step=1)
            st.caption(f"Формат планировки: {bedrooms}+{living_rooms} (всего комнат: {bedrooms + living_rooms})")

        with col_bld:
            st.markdown("**3. Здание и экспозиция**")
            age_dict = {
                "Новостройка (0 лет)": 0.0,
                "1–5 лет": 3.0,
                "6–10 лет": 8.0,
                "11–15 лет": 13.0,
                "16–20 лет": 18.0,
                "21–25 лет": 23.0,
                "26–30 лет": 28.0,
                "31–35 лет": 33.0,
                "36–40 лет": 38.0,
                "Более 40 лет": 45.0
            }
            age_selected = st.selectbox("Возраст дома", list(age_dict.keys()), index=1)
            age_num = age_dict[age_selected]

            total_floors = st.number_input("Этажность дома", min_value=1, max_value=50, value=5, step=1)
            floor_num = st.number_input("Этаж объекта", min_value=-2, max_value=50, value=2, step=1)
            tom = st.slider("Срок нахождения в листинге (дней)", min_value=1, max_value=180, value=30)

        btn_calc = st.form_submit_button("Рассчитать стоимость", use_container_width=True)

    if btn_calc:
        # Проверка базовой логики
        if floor_num > total_floors:
            st.error(f"Некорректный ввод: этаж квартиры ({floor_num}) превышает общую этажность строения ({total_floors}).")
        else:
            # Расчет производных признаков
            total_rooms = bedrooms + living_rooms
            avg_room_size = size / max(total_rooms, 1)
            relative_floor = np.clip(floor_num / max(total_floors, 1), -1.0, 1.5)

            input_data = pd.DataFrame([{
                'sub_type': sub_type,
                'tom': float(tom),
                'total_floor_count': float(total_floors),
                'size': float(size),
                'heating_type': heating,
                'bedrooms': bedrooms,
                'living_rooms': living_rooms,
                'total_rooms': total_rooms,
                'avg_room_size': avg_room_size,
                'building_age_num': age_num,
                'floor_num': float(floor_num),
                'relative_floor': relative_floor,
                'city': city,
                'district_grouped': district
            }])

            # Вычисление прогноза
            transformed_input = preprocessor.transform(input_data)
            pred_log = model.predict(transformed_input)
            pred_price = float(np.expm1(pred_log[0]))

            # Расчет интервала
            range_min = max(0.0, pred_price - MODEL_MAE)
            range_max = pred_price + MODEL_MAE
            sqm_price = pred_price / size

            st.divider()
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                st.metric("Расчетная цена", f"{pred_price:,.0f} TL")
            with col_res2:
                st.metric("Цена за 1 м²", f"{sqm_price:,.0f} TL/м²")
            with col_res3:
                st.metric("Диапазон оценки (±MAE)", f"{range_min:,.0f} – {range_max:,.0f} TL")

            st.caption(f"Точечная оценка сформирована на основе регрессионной модели. Доверительный интервал рассчитан по средней абсолютной погрешности теста (±{MODEL_MAE:,.0f} TL, MAPE {MODEL_MAPE:.1f}%).")

# 4.2 ДАШБОРД
with tabs[1]:
    st.markdown('<div class="main-header">Аналитический дашборд</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">Сравнение регрессионных моделей и структура факторов ценообразования.</div>', unsafe_allow_html=True)

    dash_tabs = st.tabs(["Метрики моделей", "Факторы стоимости", "Влияние отопления"])

    with dash_tabs[0]:
        st.markdown("**Результаты валидации алгоритмов на контрольной выборке (Test):**")
        metrics_df = pd.DataFrame([
            {"Модель": "HistGradientBoosting", "R² (log)": 0.76, "MAPE (%)": 24.54, "MAE (TL)": 110771, "MedAE (TL)": 46330, "Инференс": "< 1 мс"},
            {"Модель": "Random Forest", "R² (log)": 0.74, "MAPE (%)": 25.71, "MAE (TL)": 111482, "MedAE (TL)": 49749, "Инференс": "~ 15 мс"},
            {"Модель": "Decision Tree", "R² (log)": 0.73, "MAPE (%)": 26.42, "MAE (TL)": 115305, "MedAE (TL)": 49241, "Инференс": "< 1 мс"},
            {"Модель": "Ridge (L2 Linear)", "R² (log)": 0.70, "MAPE (%)": 28.50, "MAE (TL)": 127247, "MedAE (TL)": 51980, "Инференс": "< 1 мс"}
        ])
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        fig_m = px.bar(
            metrics_df, 
            x="Модель", 
            y="MAPE (%)", 
            text="MAPE (%)",
            color_discrete_sequence=["#2563eb"],
            title="Относительная ошибка моделей (MAPE, %)"
        )
        fig_m.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig_m.update_layout(yaxis_range=[0, 35], template="plotly_white", height=380)
        st.plotly_chart(fig_m, use_container_width=True)

    with dash_tabs[1]:
        st.markdown("**Топ-10 факторов ценообразования (Random Forest Feature Importance):**")
        fi_df = pd.DataFrame([
            {"Признак": "Площадь (size)", "Вес": 0.312},
            {"Признак": "Тип: Квартира (sub_type_Daire)", "Вес": 0.104},
            {"Признак": "Город: Стамбул (city_Istanbul)", "Вес": 0.091},
            {"Признак": "Сумма комнат (total_rooms)", "Вес": 0.072},
            {"Признак": "Этажность дома (total_floor_count)", "Вес": 0.035},
            {"Признак": "Возраст строения (building_age_num)", "Вес": 0.032},
            {"Признак": "Район: Кадыкёй (Kadikoy)", "Вес": 0.030},
            {"Признак": "Город: Измир (city_Izmir)", "Вес": 0.022},
            {"Признак": "Район: Бодрум (Bodrum)", "Вес": 0.020},
            {"Признак": "Срок на рынке (tom)", "Вес": 0.018}
        ]).sort_values(by="Вес", ascending=True)

        fig_fi = px.bar(
            fi_df, 
            x="Вес", 
            y="Признак", 
            orientation='h', 
            color_discrete_sequence=["#334155"],
            title="Относительный вклад признаков в прогноз"
        )
        fig_fi.update_layout(template="plotly_white", height=420)
        st.plotly_chart(fig_fi, use_container_width=True)

    with dash_tabs[2]:
        st.markdown("**Медианная стоимость жилья в разрезе систем отопления (по данным Zingat):**")
        heat_df = pd.DataFrame([
            {"Отопление": "Теплый пол (Yerden Isıtma)", "Медиана (TL)": 350000},
            {"Отопление": "Центральное с теплосчетчиком", "Медиана (TL)": 255000},
            {"Отопление": "Кондиционер (Klima)", "Медиана (TL)": 245000},
            {"Отопление": "Центральное (Merkezi)", "Медиана (TL)": 190000},
            {"Отопление": "Газовый котел (Kombi)", "Медиана (TL)": 185000},
            {"Отопление": "Без отопления (Yok)", "Медиана (TL)": 185000},
            {"Отопление": "Печь на угле (Soba)", "Медиана (TL)": 135000}
        ]).sort_values(by="Медиана (TL)", ascending=False)

        fig_heat = px.bar(
            heat_df, 
            x="Отопление", 
            y="Медиана (TL)", 
            color_discrete_sequence=["#d97706"],
            text="Медиана (TL)"
        )
        fig_heat.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig_heat.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig_heat, use_container_width=True)

# 4.3 СПРАВКА И ДОКУМЕНТАЦИЯ
with tabs[2]:
    st.markdown('<div class="main-header">Справочная информация о сервисе</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Назначение приложения
    Веб-приложение разработано для оценки стоимости жилой недвижимости на рынке Турции по выборке объявлений портала **Zingat**. Сервис позволяет получить расчетную цену объекта на основе регрессионной модели, обученной на очищенных исторических данных.

    ---
    ### Перечень входных характеристик
    * **Город и Район:** Определяют территориальную базовую ставку за квадратный метр. Районы с низким объемом объявлений сгруппированы в категорию `Other`.
    * **Площадь (м²):** Общая площадь объекта. В рамках предварительной обработки модель ограничена диапазоном от 15 до 800 м².
    * **Спальни и Гостиные:** Формат планировки. На их основе автоматически вычисляются суммарное число комнат и удельная площадь на комнату.
    * **Возраст дома:** Числовое представление интервалов возраста, установленных первоисточником.
    * **Этаж и Этажность:** Числовые параметры, на базе которых вычисляется относительный уровень расположения квартиры в здании.
    * **Тип отопления:** Категориальный признак, выступающий индикатором класса инженерного оснащения объекта.

    ---
    ### Ограничения и допущения
    1. **Целевой сегмент:** Модель обучена строго на данных **продажи жилья (`Satılık`)**. Оценка арендных ставок (`Kiralık`) данным модулем не поддерживается.
    2. **Валюта:** Все расчеты производятся в турецких лирах (TRY).
    3. **Неучтенные факторы:** Модель не учитывает качественные характеристики интерьера (дизайнерский ремонт), видовые характеристики, юридические обременения и класс жилого комплекса, что может вызывать отклонения для объектов индивидуальной застройки и премиум-сегмента.

    ---
    ### Технический стек
    * **Модель:** `HistGradientBoostingRegressor` (библиотека `scikit-learn`).
    * **Препроцессинг:** `ColumnTransformer` (`RobustScaler` для числовых признаков, `OneHotEncoder` для категориальных).
    * **Интерфейс:** `Streamlit`, интерактивные диаграммы `Plotly`.
    * **Версия сборки:** 1.0.0
    """)