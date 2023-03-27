import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date, timedelta
import math as m

blotter = pd.read_csv("blotter.csv")
def ledger(blotter):
    ledger_dict = {
        "trade_id" : [],
        "asset" : [],
        "dt_enter" : [],
        "dt_exit" : [],
        "success" : [],
        "n" : [],
        "rtn" : []
    }

    ledger_dict['trade_id'] = list(set(blotter['trade_id']))
    enter = blotter.loc[(blotter['trip'] == "ENTER") & ((blotter['status'] == 'SUBMITTED') | (blotter['status'] == 'LIVE'))]
    enter = enter.drop(labels=2605, axis=0)
    exit = blotter.loc[((blotter['trip'] == 'ENTER') & ((blotter['status'] == 'CANCELLED') | (blotter['status'] == 'LIVE'))) | ((blotter['trip'] == "EXIT") & ((blotter['status'] == 'FILLED') | (blotter['status']=='LIVE')))]
    success = [(exit['status'] == 'CANCELLED') * 0 + ((exit['status'] == 'FILLED') & (exit['type'] == 'MKT')) * -1 + ((exit['status'] == 'FILLED') & (exit['type'] == 'LMT')) * 1 + (exit['status'] == 'LIVE') * 3]
    exit_dates = exit['date']

    ledger_dict['asset'] = list(enter['asset'])
    ledger_dict['dt_enter'] = list(enter['date'])
    ledger_dict['success'] = np.array(success)[0]
    ledger_dict['dt_exit'] = exit_dates

    ent_dates = [datetime.strptime(date,'%m/%d/%y').date() for date in ledger_dict['dt_enter']]
    ext_dates = [datetime.strptime(date,'%m/%d/%y').date() for date in ledger_dict['dt_exit']]

    ledger_dict['n'] = np.add(np.busday_count(ent_dates, ext_dates), 1)

    entry_prices = np.array(enter['price'])
    exit_prices = np.array(exit['price'])

    print(entry_prices)

    returns = np.divide(np.log(np.divide(exit_prices, entry_prices)), ledger_dict['n'])
    ledger_dict['rtn'] = returns
    ledger = pd.DataFrame(ledger_dict)
    ledger.loc[ledger_dict['success'] == 0, ['dt_exit', 'n', 'rtn']] = ['', 3, '']
    ledger.loc[ledger_dict['success'] == 3, ['dt_exit', 'n', 'rtn', 'success']]  = ['', '', '', '']
    print(ledger)
    ledger.to_csv('ledger.csv', index=False)

ledger(blotter)


