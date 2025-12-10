import pandas as pd
import numpy as np
from dateutil import parser

def preprocess_single(sample: dict) -> pd.DataFrame:
        df = pd.DataFrame([sample])

    # Processing mapping
    processing_mapping = {
        "Double Anaerobic Washed": "Washed / Wet",
        "Semi Washed": "Washed / Wet",
        "Honey,Mossto": "Pulped natural / honey",
        "Double Carbonic Maceration / Natural": "Natural / Dry",
        "Wet Hulling": "Washed / Wet",
        "Anaerobico 1000h": "Washed / Wet",
        "SEMI-LAVADO": "Natural / Dry",
        np.nan: "Washed / Wet"
    }
    if 'Processing Method' in df.columns:
        df['Processing Method'] = df['Processing Method'].replace(processing_mapping)

    variety_mapping = {
        "Santander": "Other",
        "Typica Gesha": "Other",
        "Catucai": "Catuai",
        "Yellow Catuai": "Catuai",
        "unknow": "unknown",
        np.nan: "Other"
    }
    if 'Variety' in df.columns:
        df['Variety'] = df['Variety'].replace(variety_mapping)

    def clean_altitude(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, str):
            v = val.replace(" ", "").replace("masl", "").replace("m", "")
            if '-' in v:
                try:
                    s, e = v.split('-')
                    return (float(s) + float(e)) / 2
                except:
                    return np.nan
            try:
                return float(v)
            except:
                return np.nan
        try:
            return float(val)
        except:
            return np.nan

    if 'Altitude' in df.columns:
        df['Altitude'] = df['Altitude'].apply(clean_altitude)

    if 'Coffee Age' not in df.columns:
        if 'Harvest Year' in df.columns and 'Expiration' in df.columns:
            try:
                df['Harvest Year'] = pd.to_datetime(df['Harvest Year'].astype(str).str.split('/').str[0].str.strip(), errors='coerce', format='%Y')
                df['Expiration'] = df['Expiration'].apply(lambda x: parser.parse(str(x)) if pd.notnull(x) else pd.NaT)
                df['Coffee Age'] = (df['Expiration'] - df['Harvest Year']).dt.days
            except:
                df['Coffee Age'] = 0
        else:
            df['Coffee Age'] = 0

    for c in df.select_dtypes(include=['float64','int64']).columns:
        df[c] = df[c].fillna(0)

    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].fillna('unknown')

    return df
