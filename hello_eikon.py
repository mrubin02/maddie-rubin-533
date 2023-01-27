import os
import eikon as ek
import fetch_refinitiv_data
eikon_api = os.getenv('EIKON_API')

ek.set_app_key(eikon_api)

df = ek.get_timeseries(["MSFT.O"],
                       start_date = "2016-01-01",
                       end_date = "2016-01-10")

df2 = ek.get_timeseries(["IVV"],
                       start_date = "2016-01-01",
                       end_date = "2016-01-10")
fetch_refinitiv_data

print(df)