import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import AutoMinorLocator

def show_hist_boxplot(df, feature, max_value=None, bins=49, title=None):
    """
    Строит гистограмму и горизонтальную диаграмму размаха для выбранного столбца.

    Важно:
    - Q1, медиана, Q3 и среднее считаются в исходной шкале.
    - Границы выбросов считаются через log1p(data) + правило 1.5 IQR.
    - Boxplot строится вручную через ax.bxp(), чтобы усы соответствовали
      логарифмическому правилу, а не стандартному IQR в исходной шкале.

    Параметры:
    df : pd.DataFrame
        Таблица с данными.

    feature : str
        Название столбца.

    max_value : int
        Верхняя граница отображаемого диапазона по оси X.

    bins : int
        Количество корзин для гистограммы.

    title : str
        Название признака для заголовков графиков.
    """

    data = df[feature].dropna()

    if len(data) == 0:
        raise ValueError(f"В столбце {feature} нет данных после удаления NaN.")

    if (data < 0).any():
        raise ValueError(
            "Метод log1p + IQR подходит только для неотрицательных значений. "
            "В данных есть отрицательные значения."
        )

    if title is None:
        title = feature
        
    if max_value is None:
        max_value = data.max()    

    # Количество знаков после запятой для подписей
    max_abs_value = abs(data.max())

    if max_abs_value > 1000:
        rnd = 0
    elif max_abs_value > 100:
        rnd = 1
    elif max_abs_value > 10:
        rnd = 2
    elif max_abs_value > 1:
        rnd = 3
    else:
        rnd = 4

    # -------------------------
    # Расчёт статистик
    # -------------------------
    q1, median, q3 = np.percentile(data, [25, 50, 75])
    mean_value = data.mean()

    # Границы выбросов через log1p + 1.5 IQR
    log_data = np.log1p(data)

    q1_log = log_data.quantile(0.25)
    q3_log = log_data.quantile(0.75)
    iqr_log = q3_log - q1_log

    lower_log = q1_log - 1.5 * iqr_log
    upper_log = q3_log + 1.5 * iqr_log

    lower_bound = np.expm1(lower_log)
    upper_bound = np.expm1(upper_log)

    lower_whisker = max(data.min(), lower_bound)
    upper_whisker = min(data.max(), upper_bound)

    outliers = data[
        (data < lower_whisker) |
        (data > upper_whisker)
    ]

    # -------------------------
    # Фигура
    # -------------------------
    fig = plt.figure(figsize=(8, 10))
    gs = fig.add_gridspec(2, height_ratios=[1, 1])

    fig.text(
        0.5,
        1,
        f"Распределение значений {title}",
        ha="center",
        fontsize=14
    )

    # -------------------------
    # Гистограмма
    # -------------------------
    ax1 = fig.add_subplot(gs[0, 0])

    sns.histplot(
        x=data,
        bins=bins,
        kde=True,
        color="#a1c9f4",
        alpha=0.6,
        edgecolor="black",
        ax=ax1
    )

    ax1.set_title(f"Гистограмма для {title}")
    ax1.set_xlabel(feature)
    ax1.set_ylabel("Количество")

    ax1.set_xlim(0, max_value)
    ax1.set_xticks(np.arange(0, max_value + 1, 5000))
    ax1.xaxis.set_minor_locator(AutoMinorLocator(5))

    ax1.grid(True, which="major", axis="x", linewidth=0.8)
    ax1.grid(True, which="minor", axis="x", linewidth=0.4, alpha=0.4)
    ax1.grid(True, which="major", axis="y", alpha=0.3)

    # -------------------------
    # Горизонтальная диаграмма размаха
    # -------------------------
    ax2 = fig.add_subplot(gs[1, 0])

    stats = [{
        "med": median,
        "q1": q1,
        "q3": q3,
        "whislo": lower_whisker,
        "whishi": upper_whisker,
        "fliers": outliers.values
    }]

    ax2.bxp(
        stats,
        vert=False,
        positions=[0],
        widths=0.2,
        showfliers=True,
        patch_artist=True,
        boxprops={
            "facecolor": "#b5d9ff",
            "edgecolor": "gray"
        },
        medianprops={
            "color": "black",
            "linewidth": 1.2
        },
        whiskerprops={
            "color": "gray"
        },
        capprops={
            "color": "gray"
        },
        flierprops={
            "marker": "o",
            "markerfacecolor": "none",
            "markeredgecolor": "gray",
            "markersize": 5,
            "linestyle": "none"
        }
    )

    ax2.set_title(f"Диаграмма размаха для {title}")
    ax2.set_xlabel(feature)

    ax2.set_xlim(0, max_value)
    ax2.set_ylim(-0.65, 0.65)
    ax2.set_yticks([])

    ax2.set_xticks(np.arange(0, max_value + 1, 5000))
    ax2.xaxis.set_minor_locator(AutoMinorLocator(5))

    ax2.grid(True, which="major", axis="x", linewidth=0.8)
    ax2.grid(True, which="minor", axis="x", linewidth=0.4, alpha=0.4)

    # -------------------------
    # Подписи статистик
    # -------------------------
    values_dict = {
        "Мин": lower_whisker,
        "Q1": q1,
        "Медиана": median,
        "Q3": q3,
        "Макс": upper_whisker
    }

    seen = {}

    for label, value in values_dict.items():
        rounded_value = round(value, rnd)

        if rounded_value in seen:
            seen[rounded_value].append(label)
        else:
            seen[rounded_value] = [label]

    positions = []

    for value, labels in seen.items():
        label_text = ",".join(labels) + f": {value:.{rnd}f}"

        shift = -0.45

        for pos, pos_shift in positions:
            if abs(value - pos) < 0.01 * (data.max() - data.min()):
                shift = pos_shift + 0.6

        positions.append((value, shift))

        ax2.annotate(
            label_text,
            xy=(value, 0),
            xytext=(value, shift),
            textcoords="data",
            ha="center",
            fontsize=9,
            rotation=90
        )

    # Среднее значение
    ax2.annotate(
        f"Среднее: {mean_value:.{rnd}f}",
        xy=(mean_value, 0),
        xytext=(mean_value, 0.15),
        textcoords="data",
        ha="center",
        fontsize=9,
        color="blue",
        rotation=90
    )

    # Количество выбросов
    ax2.annotate(
        f"Количество выбросов: {len(outliers)}",
        xy=(0.1, -0.5),
        xytext=(1000, 0.55),
        textcoords="data",
        ha="left",
        fontsize=10,
        color="purple"
    )

    plt.tight_layout()
    plt.show()
    
def format_k(value):
    """
    Переводит значения в тысячи для подписей на тепловой карте
    """
    
    if pd.isna(value):
        return ""

    value_k = value / 1000

    if abs(value_k) >= 10:
        return f"{value_k:.0f}"
    else:
        return f"{value_k:.1f}"