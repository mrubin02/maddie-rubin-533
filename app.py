import pandas as pd
import refinitiv.data as rd
from dash import Dash, html, dcc, dash_table, Input, Output, State
import refinitiv.dataplatform.eikon as ek
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date, timedelta
import plotly.express as px
import dash_bootstrap_components as dbc
import os
import base64



#####################################################
app = Dash(__name__)
server = app.server
app = Dash(external_stylesheets=[dbc.themes.MORPH])

controls = dbc.Card(
    [
        dbc.Row(html.Button('QUERY Refinitiv', id='run-query', n_clicks=0)),
        html.Br(),
        dbc.Row([
            dbc.Table(
                    html.Tbody([
                        html.Tr([
                            html.Td(html.Th("Asset:")),
                            html.Td(
                                dbc.Input(
                                    id='benchmark-id',
                                    type='text',
                                    value="IVV"
                                )
                            )
                        ]),
                        html.Tr([
                            html.Td(html.Th("Date: ")),
                            html.Td(
                                dcc.DatePickerRange(
                                                id='date-id',
                                                min_date_allowed = date(2015, 1, 1),
                                                max_date_allowed = datetime.now() - timedelta(1),
                                                start_date = date(2023, 2, 1),
                                                end_date = datetime.now().date() - timedelta(1)
                                )
                            )
                        ]),
                        html.Tr([
                            html.Td(html.Th("α1:"), style = {'text-align' : 'center'}),
                            html.Td(
                                dbc.Input(
                                    id='alpha1-id',
                                    type='number',
                                    value="-0.01"
                                )
                            )
                        ]),
                        html.Tr([
                            html.Td(html.Th("n1:"), style = {'text-align' : 'center'}),
                            html.Td(
                                dbc.Input(
                                    id='n1-id',
                                    type='number',
                                    value="3"
                                )
                            )
                        ]),
                        html.Tr([
                            html.Td(html.Th("α2:"), style = {'text-align' : 'center'}),
                            html.Td(
                                dbc.Input(
                                    id='alpha2-id',
                                    type='number',
                                    value="0.01"
                                )
                            )
                        ]),
                        html.Tr([
                            html.Td(html.Th("n2:"), style = {'text-align' : 'center'}),
                            html.Td(
                                dbc.Input(
                                    id='n2-id',
                                    type='number',
                                    value="5"
                                )
                            )
                        ])
                    ]),
                bordered=True
            )
        ]
        )
    ],
    body=True
)
app.layout = dbc.Container([
    html.Br(),
    dbc.Row([html.Center("JANEANE", style={'font-family': 'Arial Rounded MT Bold',
            'display' : 'inline-block', 'font-size' : '80px'}), html.Center("a financial experience")]),
    dbc.Row(html.Center('by maddie rubin, sam seelig, and melissa king'), justify="center"),
    html.Br(),
    html.Br(),
    dbc.Row([dbc.Col(html.Img(src='assets/janeane.jpg', width = '300', height = '400')), dbc.Col(controls, md = 4, style = {'text-align' : 'center'}),
             dbc.Col(html.Img(src='assets/janeane2.jpg', width = '300', height = '400'))], align = "center", justify = 'center', style = {'text-align':'center'}),
    html.Br(),

    dbc.Row(html.Center("REACTIVE GRAPH", style={'font-family': 'Arial Rounded MT Bold',
                                                 'display': 'inline-block', "font-size": "30px"}), justify="center"),
    html.Br(),
    dbc.Row(html.Img(src='assets/graph.png')),
    html.Br(),

    dbc.Row(html.Center("BLOTTER", style={'font-family': 'Arial Rounded MT Bold',
            'display' : 'inline-block', "font-size" : "30px"}), justify = "center"),

    html.Br(),
    dash_table.DataTable(id = 'blotter', page_action='none',
        style_table={'height': '300px', 'overflowY': 'auto'}),

    html.Br(),
])

ek.set_app_key(os.getenv('EIKON_API'))

def next_n_biz_day(n, in_date):
    rd.open_session()
    total_dates = pd.DataFrame()
    date = in_date
    dates = []
    dates += [in_date]
    for i in range(n-1):
        day1 = rd.dates_and_calendars.add_periods(
            start_date=date,
            period="1D",
            calendars=["USA"],
            date_moving_convention="NextBusinessDay",
        )
        date = str(day1)
        dates += [day1]
    rd.close_session()
    return dates

@app.callback(
    Output('blotter', "data"),
    Input("run-query", "n_clicks"),
    [State('date-id', 'start_date'), State('date-id', 'end_date'), State('alpha1-id', 'value'), State('n1-id', 'value'),
     State("alpha2-id", 'value'), State('n2-id', 'value'), State('benchmark-id', 'value')],
)
def query_refinitiv(n_clicks, start_date, end_date, alpha_1, n_1, alpha_2, n_2, asset):
    ivv_prc, ivv_prc_err = ek.get_data(
        instruments = [asset],
        fields = [
            'TR.OPENPRICE(Adjusted=0)',
            'TR.HIGHPRICE(Adjusted=0)',
            'TR.LOWPRICE(Adjusted=0)',
            'TR.CLOSEPRICE(Adjusted=0)',
            'TR.PriceCloseDate'
        ],
        parameters = {
            'SDate': start_date,
            'EDate': end_date,
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
    alpha1 = float(alpha_1)
    n1 = int(n_1)
    alpha2 = float(alpha_2)
    n2 = int(n_2)

    # submitted entry orders
    submitted_entry_orders = pd.DataFrame({
        "trade_id": range(1, ivv_prc.shape[0]),
        "date": list(pd.to_datetime(ivv_prc["Date"].iloc[1:]).dt.date),
        "asset": asset,
        "trip": 'ENTER',
        "action": "BUY",
        "type": "LMT",
        "price": ivv_prc['Close Price'].iloc[:-1] * (1 + alpha1),
        'status': 'SUBMITTED'
    })
    print(submitted_entry_orders)

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

    if any(filled_entry_orders['status'] =='LIVE'):
        live_entry_orders = pd.concat(
            [
                pd.DataFrame({
                    "trade_id": ivv_prc.shape[0],
                    "date": pd.to_datetime(next_business_day).date(),
                    "asset": asset,
                    "trip": 'ENTER',
                    "action": "BUY",
                    "type": "LMT",
                    "price": ivv_prc['Close Price'].iloc[-1] * (1 + alpha1),
                    'status': 'LIVE'
                },
                    index=[0]
                ),
                filled_entry_orders[filled_entry_orders['status'] == 'LIVE']
            ]
        )
    else:
        live_entry_orders = pd.DataFrame({
            "trade_id": ivv_prc.shape[0],
            "date": pd.to_datetime(next_business_day).date(),
            "asset": asset,
            "trip": 'ENTER',
            "action": "BUY",
            "type": "LMT",
            "price": ivv_prc['Close Price'].iloc[-1] * (1 + alpha1),
            'status': 'LIVE'
        },
            index=[0]
        )


    filled_entry_orders = filled_entry_orders[
        filled_entry_orders['status'] == 'FILLED'
        ]

    print(filled_entry_orders)
    print(live_entry_orders)

    submitted_exit_orders = filled_entry_orders.copy()
    submitted_exit_orders['status'] = 'SUBMITTED'
    submitted_exit_orders['trip'] = 'EXIT'
    submitted_exit_orders['action'] = 'SELL'
    submitted_exit_orders['price'] = filled_entry_orders['price'] * (1+alpha2)

    print(submitted_exit_orders)

    ## filled and cancelled exit limit orders

    cancelled_dict =  {
        "trade_id" : [],
        "date": [],
        "asset": asset,
        "trip": 'EXIT',
        "action": "SELL",
        "type": "LMT",
        "price": [],
        'status': 'CANCELLED'

    }
    filled_dict = {
        "trade_id": [],
        "date": [],
        "asset": asset,
        "trip": 'EXIT',
        "action": "SELL",
        "type": "LMT",
        "price": [],
        'status': 'FILLED'
    }

    live_dict = {
        "trade_id": [],
        "date": [],
        "asset": asset,
        "trip": 'EXIT',
        "action": "SELL",
        "type": "LMT",
        "price": [],
        'status': 'LIVE'
    }

    mkt_dict = {
        "trade_id": [],
        "date": [],
        "asset": asset,
        "trip": 'EXIT',
        "action": "SELL",
        "type": "MKT",
        "price": [],
        'status': 'SUBMITTED'
    }

    sub_dict = submitted_exit_orders['date'].to_numpy()
    for k in submitted_exit_orders.to_dict('records'):
        prices = []
        dates = next_n_biz_day(n2, k['date'])
        for j in dates:
            for i in ivv_prc['Date']:
                if i == j:
                    if j == dates[0]:
                        prices += [ivv_prc[ivv_prc['Date'] == i]['Close Price'].to_numpy()[0]]
                    else:
                        prices += [ivv_prc[ivv_prc['Date'] == i]['High Price'].to_numpy()[0]]
                    break
        price_adjust = [p for p in prices if p >= k['price']]
        if (len(price_adjust) > 0):
            filled_dict['trade_id'] += [k['trade_id']]
            filled_dict['date'] += [datetime.strptime(str(dates[prices.index(price_adjust[0])])[0:10], "%Y-%m-%d").date()]
            filled_dict['price'] += [k['price']]
        else:
            if len(prices) < n2:
                live_dict['trade_id'] +=[k['trade_id']]
                live_dict['date'] += [pd.to_datetime(next_business_day).date()]
                live_dict['price'] += [ivv_prc['Close Price'].iloc[-1] * (1 + alpha2)]
            else:
                cancelled_dict['trade_id'] += [k['trade_id']]
                cancelled_dict['date'] += [datetime.strptime(str(dates[len(dates)-1])[0:10], "%Y-%m-%d").date()]
                cancelled_dict['price'] += [k['price']]
                mkt_dict['trade_id'] += [k['trade_id']]
                mkt_dict['date'] += [datetime.strptime(str(dates[len(dates)-1])[0:10], "%Y-%m-%d").date()]
                mkt_dict['price'] += [ivv_prc[ivv_prc['Date'] == dates[len(dates)-1]]['Close Price'].to_numpy()[0]]

    mkt_filled_dict = mkt_dict.copy()
    mkt_filled_dict['status'] = 'FILLED'

    filled_exit_orders = pd.DataFrame(filled_dict)
    cancelled_exit_orders = pd.DataFrame(cancelled_dict)
    live_exit_orders = pd.DataFrame(live_dict)
    submitted_mkt_exit_orders = pd.DataFrame(mkt_dict)
    filled_mkt_exit_orders = pd.DataFrame(mkt_filled_dict)

    print(filled_exit_orders)
    print(cancelled_exit_orders)
    print(live_exit_orders)
    print(submitted_mkt_exit_orders)
    print(filled_mkt_exit_orders)


    blotter = pd.concat(
        [
            submitted_entry_orders,
            cancelled_entry_orders,
            filled_entry_orders,
            live_entry_orders,
            submitted_exit_orders,
            filled_exit_orders,
            live_exit_orders,
            cancelled_exit_orders,
            submitted_mkt_exit_orders,
            filled_mkt_exit_orders
        ]
    ).sort_values(["trade_id", 'trip', 'date'])

    print(blotter)
    return blotter.to_dict('records')import pandas as pd
import refinitiv.data as rd
from dash import Dash, html, dcc, dash_table, Input, Output, State
import refinitiv.dataplatform.eikon as ek
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date, timedelta
import plotly.express as px
import dash_bootstrap_components as dbc
import os
import base64



#####################################################
app = Dash(__name__)
server = app.server
app = Dash(external_stylesheets=[dbc.themes.MORPH])

controls = dbc.Card(
    [
        dbc.Row(html.Button('QUERY Refinitiv', id='run-query', n_clicks=0)),
        html.Br(),
        dbc.Row([
            dbc.Table(
                    html.Tbody([
                        html.Tr([
                            html.Td(html.Th("Asset:")),
                            html.Td(
                                dbc.Input(
                                    id='benchmark-id',
                                    type='text',
                                    value="IVV"
                                )
                            )
                        ]),
                        html.Tr([
                            html.Td(html.Th("Date: ")),
                            html.Td(
                                dcc.DatePickerRange(
                                                id='date-id',
                                                min_date_allowed = date(2015, 1, 1),
                                                max_date_allowed = datetime.now() - timedelta(1),
                                                start_date = date(2023, 2, 1),
                                                end_date = datetime.now().date() - timedelta(1)
                                )
                            )
                        ]),
                        html.Tr([
                            html.Td(html.Th("α1:"), style = {'text-align' : 'center'}),
                            html.Td(
                                dbc.Input(
                                    id='alpha1-id',
                                    type='number',
                                    value="-0.01"
                                )
                            )
                        ]),
                        html.Tr([
                            html.Td(html.Th("n1:"), style = {'text-align' : 'center'}),
                            html.Td(
                                dbc.Input(
                                    id='n1-id',
                                    type='number',
                                    value="3"
                                )
                            )
                        ]),
                        html.Tr([
                            html.Td(html.Th("α2:"), style = {'text-align' : 'center'}),
                            html.Td(
                                dbc.Input(
                                    id='alpha2-id',
                                    type='number',
                                    value="0.01"
                                )
                            )
                        ]),
                        html.Tr([
                            html.Td(html.Th("n2:"), style = {'text-align' : 'center'}),
                            html.Td(
                                dbc.Input(
                                    id='n2-id',
                                    type='number',
                                    value="5"
                                )
                            )
                        ])
                    ]),
                bordered=True
            )
        ]
        )
    ],
    body=True
)
app.layout = dbc.Container([
    html.Br(),
    dbc.Row([html.Center("JANEANE", style={'font-family': 'Arial Rounded MT Bold',
            'display' : 'inline-block', 'font-size' : '80px'}), html.Center("a financial experience")]),
    dbc.Row(html.Center('by maddie rubin, sam seelig, and melissa king'), justify="center"),
    html.Br(),
    html.Br(),
    dbc.Row([dbc.Col(html.Img(src='assets/janeane.jpg', width = '300', height = '400')), dbc.Col(controls, md = 4, style = {'text-align' : 'center'}),
             dbc.Col(html.Img(src='assets/janeane2.jpg', width = '300', height = '400'))], align = "center", justify = 'center', style = {'text-align':'center'}),
    html.Br(),

    dbc.Row(html.Center("REACTIVE GRAPH", style={'font-family': 'Arial Rounded MT Bold',
                                                 'display': 'inline-block', "font-size": "30px"}), justify="center"),
    html.Br(),
    dbc.Row(html.Img(src='assets/graph.png')),
    html.Br(),

    dbc.Row(html.Center("BLOTTER", style={'font-family': 'Arial Rounded MT Bold',
            'display' : 'inline-block', "font-size" : "30px"}), justify = "center"),

    html.Br(),
    dash_table.DataTable(id = 'blotter', page_action='none',
        style_table={'height': '300px', 'overflowY': 'auto'}),

    html.Br(),
])

ek.set_app_key(os.getenv('EIKON_API'))

def next_n_biz_day(n, in_date):
    rd.open_session()
    total_dates = pd.DataFrame()
    date = in_date
    dates = []
    dates += [in_date]
    for i in range(n-1):
        day1 = rd.dates_and_calendars.add_periods(
            start_date=date,
            period="1D",
            calendars=["USA"],
            date_moving_convention="NextBusinessDay",
        )
        date = str(day1)
        dates += [day1]
    rd.close_session()
    return dates

@app.callback(
    Output('blotter', "data"),
    Input("run-query", "n_clicks"),
    [State('date-id', 'start_date'), State('date-id', 'end_date'), State('alpha1-id', 'value'), State('n1-id', 'value'),
     State("alpha2-id", 'value'), State('n2-id', 'value'), State('benchmark-id', 'value')],
)
def query_refinitiv(n_clicks, start_date, end_date, alpha_1, n_1, alpha_2, n_2, asset):
    ivv_prc, ivv_prc_err = ek.get_data(
        instruments = [asset],
        fields = [
            'TR.OPENPRICE(Adjusted=0)',
            'TR.HIGHPRICE(Adjusted=0)',
            'TR.LOWPRICE(Adjusted=0)',
            'TR.CLOSEPRICE(Adjusted=0)',
            'TR.PriceCloseDate'
        ],
        parameters = {
            'SDate': start_date,
            'EDate': end_date,
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
    alpha1 = float(alpha_1)
    n1 = int(n_1)
    alpha2 = float(alpha_2)
    n2 = int(n_2)

    # submitted entry orders
    submitted_entry_orders = pd.DataFrame({
        "trade_id": range(1, ivv_prc.shape[0]),
        "date": list(pd.to_datetime(ivv_prc["Date"].iloc[1:]).dt.date),
        "asset": asset,
        "trip": 'ENTER',
        "action": "BUY",
        "type": "LMT",
        "price": ivv_prc['Close Price'].iloc[:-1] * (1 + alpha1),
        'status': 'SUBMITTED'
    })
    print(submitted_entry_orders)

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

    if any(filled_entry_orders['status'] =='LIVE'):
        live_entry_orders = pd.concat(
            [
                pd.DataFrame({
                    "trade_id": ivv_prc.shape[0],
                    "date": pd.to_datetime(next_business_day).date(),
                    "asset": asset,
                    "trip": 'ENTER',
                    "action": "BUY",
                    "type": "LMT",
                    "price": ivv_prc['Close Price'].iloc[-1] * (1 + alpha1),
                    'status': 'LIVE'
                },
                    index=[0]
                ),
                filled_entry_orders[filled_entry_orders['status'] == 'LIVE']
            ]
        )
    else:
        live_entry_orders = pd.DataFrame({
            "trade_id": ivv_prc.shape[0],
            "date": pd.to_datetime(next_business_day).date(),
            "asset": asset,
            "trip": 'ENTER',
            "action": "BUY",
            "type": "LMT",
            "price": ivv_prc['Close Price'].iloc[-1] * (1 + alpha1),
            'status': 'LIVE'
        },
            index=[0]
        )


    filled_entry_orders = filled_entry_orders[
        filled_entry_orders['status'] == 'FILLED'
        ]

    print(filled_entry_orders)
    print(live_entry_orders)

    submitted_exit_orders = filled_entry_orders.copy()
    submitted_exit_orders['status'] = 'SUBMITTED'
    submitted_exit_orders['trip'] = 'EXIT'
    submitted_exit_orders['action'] = 'SELL'
    submitted_exit_orders['price'] = filled_entry_orders['price'] * (1+alpha2)

    print(submitted_exit_orders)

    ## filled and cancelled exit limit orders

    cancelled_dict =  {
        "trade_id" : [],
        "date": [],
        "asset": asset,
        "trip": 'EXIT',
        "action": "SELL",
        "type": "LMT",
        "price": [],
        'status': 'CANCELLED'

    }
    filled_dict = {
        "trade_id": [],
        "date": [],
        "asset": asset,
        "trip": 'EXIT',
        "action": "SELL",
        "type": "LMT",
        "price": [],
        'status': 'FILLED'
    }

    live_dict = {
        "trade_id": [],
        "date": [],
        "asset": asset,
        "trip": 'EXIT',
        "action": "SELL",
        "type": "LMT",
        "price": [],
        'status': 'LIVE'
    }

    mkt_dict = {
        "trade_id": [],
        "date": [],
        "asset": asset,
        "trip": 'EXIT',
        "action": "SELL",
        "type": "MKT",
        "price": [],
        'status': 'SUBMITTED'
    }

    sub_dict = submitted_exit_orders['date'].to_numpy()
    for k in submitted_exit_orders.to_dict('records'):
        prices = []
        dates = next_n_biz_day(n2, k['date'])
        for j in dates:
            for i in ivv_prc['Date']:
                if i == j:
                    if j == dates[0]:
                        prices += [ivv_prc[ivv_prc['Date'] == i]['Close Price'].to_numpy()[0]]
                    else:
                        prices += [ivv_prc[ivv_prc['Date'] == i]['High Price'].to_numpy()[0]]
                    break
        price_adjust = [p for p in prices if p >= k['price']]
        if (len(price_adjust) > 0):
            filled_dict['trade_id'] += [k['trade_id']]
            filled_dict['date'] += [datetime.strptime(str(dates[prices.index(price_adjust[0])])[0:10], "%Y-%m-%d").date()]
            filled_dict['price'] += [k['price']]
        else:
            if len(prices) < n2:
                live_dict['trade_id'] +=[k['trade_id']]
                live_dict['date'] += [pd.to_datetime(next_business_day).date()]
                live_dict['price'] += [ivv_prc['Close Price'].iloc[-1] * (1 + alpha2)]
            else:
                cancelled_dict['trade_id'] += [k['trade_id']]
                cancelled_dict['date'] += [datetime.strptime(str(dates[len(dates)-1])[0:10], "%Y-%m-%d").date()]
                cancelled_dict['price'] += [k['price']]
                mkt_dict['trade_id'] += [k['trade_id']]
                mkt_dict['date'] += [datetime.strptime(str(dates[len(dates)-1])[0:10], "%Y-%m-%d").date()]
                mkt_dict['price'] += [ivv_prc[ivv_prc['Date'] == dates[len(dates)-1]]['Close Price'].to_numpy()[0]]

    mkt_filled_dict = mkt_dict.copy()
    mkt_filled_dict['status'] = 'FILLED'

    filled_exit_orders = pd.DataFrame(filled_dict)
    cancelled_exit_orders = pd.DataFrame(cancelled_dict)
    live_exit_orders = pd.DataFrame(live_dict)
    submitted_mkt_exit_orders = pd.DataFrame(mkt_dict)
    filled_mkt_exit_orders = pd.DataFrame(mkt_filled_dict)

    print(filled_exit_orders)
    print(cancelled_exit_orders)
    print(live_exit_orders)
    print(submitted_mkt_exit_orders)
    print(filled_mkt_exit_orders)


    blotter = pd.concat(
        [
            submitted_entry_orders,
            cancelled_entry_orders,
            filled_entry_orders,
            live_entry_orders,
            submitted_exit_orders,
            filled_exit_orders,
            live_exit_orders,
            cancelled_exit_orders,
            submitted_mkt_exit_orders,
            filled_mkt_exit_orders
        ]
    ).sort_values(["trade_id", 'trip', 'date'])

    print(blotter)
    return blotter.to_dict('records')

if __name__ == '__main__':
    app.run_server(debug=True)

if __name__ == '__main__':
    app.run_server(debug=True)