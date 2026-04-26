import os
from entsoe import entsoe
import pandas as pd
api_key = os.getenv("EPEXPREDICTOR_ENTSOE_API_KEY")
if api_key:
    client = entsoe.EntsoePandasClient(api_key=api_key)
    try:
        from datetime import datetime, timedelta
        start = pd.Timestamp(datetime.now() - timedelta(days=1), tz='UTC')
        end = pd.Timestamp(datetime.now() + timedelta(days=2), tz='UTC')
        a01 = client.query_load_forecast("FI", start=start, end=end) # A01 is default
        print("A01:")
        print(a01.head())
        print("Freq:", a01.index.freq)
    except Exception as e:
        print("err A01", e)
else:
    print("NO API KEY")
