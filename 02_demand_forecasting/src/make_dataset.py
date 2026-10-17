"""
Hourly bike-sharing demand (UCI, 2011-2012) pulled from OpenML (dataset id 42712).

The OpenML copy keeps the chronological row order but drops the calendar date, so
the timestamp is rebuilt by walking the hour column and incrementing the day every
time the hour wraps around. The reconstruction is then validated against the
dataset's own weekday column - if it disagrees we fail loudly instead of
silently training on a broken time index.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CSV_PATH = DATA_DIR / "bike_sharing_hourly.csv"
OPENML_ID = 42712
START = pd.Timestamp("2011-01-01")


def _download() -> pd.DataFrame:
    from sklearn.datasets import fetch_openml

    print("  [data] fetching OpenML dataset %d (bike sharing, hourly)" % OPENML_ID)
    bunch = fetch_openml(data_id=OPENML_ID, as_frame=True)
    df = bunch.frame.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _rebuild_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Map each calendar day in the file onto the real 2011-2012 calendar.

    Rows arrive in chronological order and the hour column wraps at every new day,
    which gives us day groups. Whole days are missing from the file, so the groups
    are matched forward against the real calendar using the (year, month, weekday)
    triple that ships with the data - that pins every group to exactly one date.
    """
    hour = df["hour"].astype(int).to_numpy()
    weekday = df["weekday"].astype(int).to_numpy()
    group = np.zeros(len(hour), dtype=int)
    g = 0
    for i in range(1, len(hour)):
        # a new day starts when the hour wraps, or when the weekday changes without
        # a wrap - that happens on 2012-10-29, where hurricane Sandy knocked out
        # the service mid-afternoon and the next recorded hour is already the 30th
        if hour[i] <= hour[i - 1] or weekday[i] != weekday[i - 1]:
            g += 1
        group[i] = g

    keys = (
        df.assign(_g=group)
        .groupby("_g")[["year", "month", "weekday"]]
        .first()
        .astype(int)
    )

    calendar = pd.DataFrame({"date": pd.date_range(START, "2012-12-31", freq="D")})
    calendar["year"] = calendar.date.dt.year - START.year
    calendar["month"] = calendar.date.dt.month
    # the dataset encodes weekday with 0 = Sunday
    calendar["weekday"] = (calendar.date.dt.dayofweek + 1) % 7

    dates, ptr = [], 0
    for _, k in keys.iterrows():
        while ptr < len(calendar) and not (
            calendar.year[ptr] == k.year
            and calendar.month[ptr] == k.month
            and calendar.weekday[ptr] == k.weekday
        ):
            ptr += 1
        if ptr >= len(calendar):
            raise AssertionError("could not align a day group to the calendar")
        dates.append(calendar.date[ptr])
        ptr += 1

    out = df.copy()
    out["timestamp"] = pd.Series(dates).iloc[group].to_numpy() + pd.to_timedelta(hour, unit="h")

    weekday_ok = ((out.timestamp.dt.dayofweek + 1) % 7 == out.weekday.astype(int)).mean()
    month_ok = (out.timestamp.dt.month == out.month.astype(int)).mean()
    dup = int(out.timestamp.duplicated().sum())
    print("  [data] timestamp check: weekday %.3f, month %.3f, duplicates %d"
          % (weekday_ok, month_ok, dup))
    if min(weekday_ok, month_ok) < 1.0 or dup:
        raise AssertionError("timestamp reconstruction disagrees with the shipped columns")
    return out.sort_values("timestamp").reset_index(drop=True)


def load() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CSV_PATH.exists():
        print("  [data] cached %s" % CSV_PATH.name)
        return pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
    df = _rebuild_timestamp(_download())
    df.to_csv(CSV_PATH, index=False)
    print("  [data] %d hourly observations %s -> %s"
          % (len(df), (str(df.timestamp.min().date()) + " .. " + str(df.timestamp.max().date())),
             CSV_PATH.name))
    return df


if __name__ == "__main__":
    d = load()
    print(d.head())
    print(d.dtypes)
