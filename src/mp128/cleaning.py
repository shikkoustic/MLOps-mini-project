import pandas as pd


def clean_train(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the four data-quality issues recorded for MP128."""

    # Work on a copy so the original DataFrame is not modified.
    df = df.copy()

    # 1. Fix wind speed unit mix-up in Residential areas.
    residential = df["station_zone"] == "Residential"
    df.loc[residential, "wind_kmph"] = (
        df.loc[residential, "wind_kmph"] / 1000
    )

    # 2. Replace the stubble_fires sentinel with the column median.
    df["stubble_fires"] = df["stubble_fires"].replace(9999, float("nan"))
    median_value = df["stubble_fires"].median()
    df["stubble_fires"] = df["stubble_fires"].fillna(median_value)

    # 3. Correct confirmed 10x typos in the target using the IQR upper fence.
    target = df["pm25_next_day"]
    q1 = target.quantile(0.25)
    q3 = target.quantile(0.75)
    iqr = q3 - q1

    if iqr > 0:
        upper_fence = q3 + 3 * iqr
        outliers = target[target > upper_fence]
        normal_max = target[target <= upper_fence].max()

        confirmed = outliers[(outliers / 10) <= normal_max * 1.1]
        df.loc[confirmed.index, "pm25_next_day"] /= 10

    # 4. Normalize day_type spelling variants.
    df["day_type"] = (
        df["day_type"]
        .str.lower()
        .str.strip()
        .str.rstrip(".")
    )

    return df

