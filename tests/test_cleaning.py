import pandas as pd

from mp128.cleaning import clean_train


def test_clean_train_fixes_data_issues():
    df = pd.DataFrame({
        "station_zone": ["Residential"] * 8,
        "wind_kmph": [12000, 12000, 12000, 12000,
                      12000, 12000, 12000, 12000],
        "stubble_fires": [10, 20, 9999, 30, 40, 50, 60, 70],
        "pm25_next_day": [40, 42, 44, 46, 48, 50, 52, 500],
        "day_type": [
            "Festival", " FESTIVAL.", "weekday",
            "WEEKDAY.", "Weekend", " weekend.",
            "festival ", "Weekday"
        ]
    })

    cleaned = clean_train(df)

    assert set(cleaned["day_type"].unique()) == {
        "festival", "weekday", "weekend"
    }
    assert cleaned["day_type"].nunique() == 3
    assert cleaned.loc[2, "stubble_fires"] == 40
    assert 9999 not in cleaned["stubble_fires"].values
    assert cleaned.loc[7, "pm25_next_day"] == 50


def test_clean_train_corrects_wind_speed():
    df = pd.DataFrame({
        "station_zone": ["Residential", "Residential", "Commercial"],
        "wind_kmph": [12000, 30000, 22],
        "stubble_fires": [10, 20, 30],
        "pm25_next_day": [40, 42, 44],
        "day_type": ["weekday", "weekend", "festival"]
    })

    cleaned = clean_train(df)

    assert cleaned.loc[0, "wind_kmph"] == 12
    assert cleaned.loc[1, "wind_kmph"] == 30
    assert cleaned.loc[2, "wind_kmph"] == 22
    assert cleaned["wind_kmph"].max() <= 35

