# Imports
from datetime import datetime
import eikon as ek
import pandas as pd
import os

###### Before running this script:
# 1) Create an app key within Refinitiv
# 2) Create an environmental variable on your computer that stores your key.
#      (I named mine 'EIKON_API')
######

# 3) Use your app key in this Python session w the following line:
ek.set_app_key(os.getenv('EIKON_API'))


# 4) Use get_timeseries() to get historical prices for 2 stocks and an ETF
#    --> if this command fails... don't forget to check and make sure you have
#        Refinitiv Workstation open and running on your computer :)
get_timeseries_output = ek.get_timeseries(
    rics=["IVV"],
    start_date="2023-01-25",
    end_date=datetime.now().strftime("%Y-%m-%d")
)
get_timeseries_output.to_csv('get_timeseries_output.csv', index=False)

