import pandas as pd
from datetime import datetime
import numpy as np
import os
import refinitiv.dataplatform.eikon as ek
import refinitiv.data as rd

#####################################################

ek.set_app_key(os.getenv('EIKON_API'))

start_date_str = '2023-01-30'
end_date_str = '2023-02-08'

ivv_prc, ivv_prc_err = ek.get_data(
    instruments = ["IVV"],
    fields = [
        'TR.OPENPRICE(Adjusted=0)',
        'TR.HIGHPRICE(Adjusted=0)',
        'TR.LOWPRICE(Adjusted=0)',
        'TR.CLOSEPRICE(Adjusted=0)',
        'TR.PriceCloseDate'
    ],
    parameters = {
        'SDate': start_date_str,
        'EDate': end_date_str,
        'Frq': 'D'
    }
)

ivv_prc['Date'] = pd.to_datetime(ivv_prc['Date']).dt.date
ivv_prc.drop(columns='Instrument', inplace=True)

##### Get the next business day from Refinitiv!!!!!!!
rd.open_session()

next_business_day = rd.dates_and_calendars.add_periods(
    start_date= ivv_prc['Date'].iloc[-1].strftime("%Y-%m-%d"),
    period="1D",
    calendars=["USA"],
    date_moving_convention="NextBusinessDay",
)

rd.close_session()
######################################################

# Parameters:
alpha1 = -0.01
n1 = 3

# submitted entry orders
submitted_entry_orders = pd.DataFrame({
    "trade_id": range(1, ivv_prc.shape[0]),
    "date": list(pd.to_datetime(ivv_prc["Date"].iloc[1:]).dt.date),
    "asset": "IVV",
    "trip": 'ENTER',
    "action": "BUY",
    "type": "LMT",
    "price": round(
        ivv_prc['Close Price'].iloc[:-1] * (1 + alpha1),
        2
    ),
    'status': 'SUBMITTED'
})

# if the lowest traded price is still higher than the price you bid, then the
# order never filled and was cancelled.
with np.errstate(invalid='ignore'):
    cancelled_entry_orders = submitted_entry_orders[
        np.greater(
            ivv_prc['Low Price'].iloc[1:][::-1].rolling(3).min()[::-1].to_numpy(),
            submitted_entry_orders['price'].to_numpy()
        )
    ].copy()
cancelled_entry_orders.reset_index(drop=True, inplace=True)
cancelled_entry_orders['status'] = 'CANCELLED'
cancelled_entry_orders['date'] = pd.DataFrame(
    {'cancel_date': submitted_entry_orders['date'].iloc[(n1-1):].to_numpy()},
    index=submitted_entry_orders['date'].iloc[:(1-n1)].to_numpy()
).loc[cancelled_entry_orders['date']]['cancel_date'].to_list()
print(cancelled_entry_orders)

filled_entry_orders = submitted_entry_orders[
    submitted_entry_orders['trade_id'].isin(
        list(
            set(submitted_entry_orders['trade_id']) - set(
                cancelled_entry_orders['trade_id']
            )
        )
    )
].copy()
filled_entry_orders.reset_index(drop=True, inplace=True)
filled_entry_orders['status'] = 'FILLED'
for i in range(0, len(filled_entry_orders)):

    idx1 = np.flatnonzero(
        ivv_prc['Date'] == filled_entry_orders['date'].iloc[i]
    )[0]

    ivv_slice = ivv_prc.iloc[idx1:(idx1+n1)]['Low Price']

    fill_inds = ivv_slice <= filled_entry_orders['price'].iloc[i]

    if (len(fill_inds) < n1) & (not any(fill_inds)):
        filled_entry_orders.at[i, 'status'] = 'LIVE'
    else:
        filled_entry_orders.at[i, 'date'] = ivv_prc['Date'].iloc[
            fill_inds.idxmax()
        ]

live_entry_orders = pd.DataFrame({
    "trade_id": ivv_prc.shape[0],
    "date": pd.to_datetime(next_business_day).date(),
    "asset": "IVV",
    "trip": 'ENTER',
    "action": "BUY",
    "type": "LMT",
    "price": round(ivv_prc['Close Price'].iloc[-1] * (1 + alpha1), 2),
    'status': 'LIVE'
},
    index=[0]
)

if any(filled_entry_orders['status'] =='LIVE'):
    live_entry_orders = pd.concat([
        filled_entry_orders[filled_entry_orders['status'] == 'LIVE'],
        live_entry_orders
    ])
    live_entry_orders['date'] = pd.to_datetime(next_business_day).date()

filled_entry_orders = filled_entry_orders[
    filled_entry_orders['status'] == 'FILLED'
    ]

entry_orders = pd.concat(
    [
        submitted_entry_orders,
        cancelled_entry_orders,
        filled_entry_orders,
        live_entry_orders
    ]
).sort_values(["date", 'trade_id'])

print("submitted_entry_orders:")
print(submitted_entry_orders)

print("cancelled_entry_orders:")
print(cancelled_entry_orders)

print("filled_entry_orders:")
print(filled_entry_orders)

print("live_entry_orders:")
print(live_entry_orders)

print("entry_orders:")
print(entry_orders)