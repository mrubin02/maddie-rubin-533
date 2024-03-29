import pandas as pd
import refinitiv.data as rd
from dash import Dash, html, dcc, dash_table, Input, Output, State
import refinitiv.dataplatform.eikon as ek
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date, timedelta
import dash_bootstrap_components as dbc
import os
import base64


#####################################################
app = Dash(__name__)
server = app.server
app = Dash(external_stylesheets=[dbc.themes.MORPH])

controls = dbc.Card(
    [
        dbc.Row(html.Button('Get Data', id='run-query', n_clicks=0)),
        html.Br(),
        dbc.Row([
            dbc.Table(
                    html.Tbody([
                        html.Tr([
                            html.Td(html.Th("Benchmark:")),
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
                                                min_date_allowed = date(2020, 1, 3),
                                                max_date_allowed = date(2023, 3, 20),
                                                start_date = date(2020, 1, 3),
                                                end_date = date(2023, 3, 20),
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

controls2 = dbc.Card(
    [
        dbc.Row(html.Button('Set Blotter Parameters', id='params', n_clicks=0)),
        html.Br(),
        dbc.Row([
            dbc.Table(
                    html.Tbody([
                        html.Tr([
                            html.Td(html.Th("Alpha1:")),
                            html.Td(
                                dbc.Input(
                                    id='alpha1-id',
                                    type='text',
                                    value="-0.01"
                                )
                            )
                        ]),
                        html.Tr([
                            html.Td(html.Th("n1:")),
                            html.Td(
                                dbc.Input(
                                    id='n1-id',
                                    type='text',
                                    value="3"
                                )
                            )
                        ]),

                        html.Tr([
                            html.Td(html.Th("Alpha2:")),
                            html.Td(
                                dbc.Input(
                                    id='alpha2-id',
                                    type='text',
                                    value="0.01"
                                )
                            )
                        ]),
                        html.Tr([
                            html.Td(html.Th("n2:")),
                            html.Td(
                                dbc.Input(
                                    id='n2-id',
                                    type='text',
                                    value="5"
                                )
                            )
                        ]),

                    ]),
                bordered=True
            )
        ]
        )
    ],
    body=True
)

controls3 = dbc.Card(
    [
        dbc.Row(html.Button('Set Lookback Window', id='lookback-bt', n_clicks=0)),
        html.Br(),
        dbc.Row([
            dbc.Table(
                    html.Tbody([
                        html.Tr([
                            html.Td(html.Th("Lookback Size:")),
                            html.Td(
                                dbc.Input(
                                    id='lookback',
                                    type='text',
                                    value="60"
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

controls4 = dbc.Card(
    [
        dbc.Row(html.Button('Set b (Hoeffding Lower Bound)', id='b_bt', n_clicks=0)),
        html.Br(),
        dbc.Row([
            dbc.Table(
                    html.Tbody([
                        html.Tr([
                            html.Td(html.Th("b:")),
                            html.Td(
                                dbc.Input(
                                    id='b',
                                    type='text',
                                    value="-0.3"
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
    dbc.Row(html.Center('by maddie rubin, sam seelig, melissa king, and julia leodori'), justify="center"),
    html.Br(),
    html.Br(),
    dbc.Row([dbc.Col(html.Img(src='assets/janeane.jpg', width = '300', height = '400')), dbc.Col(controls, md = 4, style = {'text-align' : 'center'}),
             dbc.Col(html.Img(src='assets/janeane2.jpg', width = '300', height = '400'))], align = "center", justify = 'center', style = {'text-align':'center'}),
    html.Br(),

    html.Br(),

    dbc.Row(html.Center("REFINITIV DATA", style={'font-family': 'Arial Rounded MT Bold',
                                          'display': 'inline-block', "font-size": "30px"}), justify="center"),
    html.Br(),
    dash_table.DataTable(
        id = "history-tbl",
        page_action='none',
        style_table={'height': '300px', 'overflowY': 'auto'}
    ),

    html.Br(),
    dbc.Row([dbc.Col(),
             dbc.Col(controls2, md=4, style={'text-align': 'center'}),
             dbc.Col()], align="center", justify='center',
            style={'text-align': 'center'}),

    html.Br(),
    html.Br(),
    dbc.Row(html.Center("BLOTTER", style={'font-family': 'Arial Rounded MT Bold',
            'display' : 'inline-block', "font-size" : "30px"}), justify = "center"),

    html.Br(),
    dash_table.DataTable(id = 'blotter', page_action='none',
        style_table={'height': '300px', 'overflowY': 'auto'}),

    html.Br(),

    dbc.Row(html.Center("LEDGER 1", style={'font-family': 'Arial Rounded MT Bold',
            'display' : 'inline-block', "font-size" : "30px"}), justify = "center"),

    html.Br(),
    dash_table.DataTable(id = 'ledger1', page_action='none',
        style_table={'height': '300px', 'overflowY': 'auto'}),

    html.Br(),

    html.Br(),
    dbc.Row([dbc.Col(),
             dbc.Col(controls3, md=4, style={'text-align': 'center'}),
             dbc.Col()], align="center", justify='center',
            style={'text-align': 'center'}),

    html.Br(),

    dbc.Row(html.Center("PREDICTED SUCCESS", style={'font-family': 'Arial Rounded MT Bold',
            'display' : 'inline-block', "font-size" : "30px"}), justify = "center"),

    html.Br(),
    dash_table.DataTable(id = 'ledger2', page_action='none',
        style_table={'height': '300px', 'overflowY': 'auto'}),

    html.Br(),

    dbc.Row(html.Center("RETURNS TABLE", style={'font-family': 'Arial Rounded MT Bold',
            'display' : 'inline-block', "font-size" : "30px"}), justify = "center"),
    html.Br(),
    dash_table.DataTable(id = 'ledger3', page_action='none',
        style_table={'height': '300px', 'overflowY': 'auto'}),

    html.Br(),



    dbc.Row(html.Center("ALPHA BETA PLOT", style={'font-family': 'Arial Rounded MT Bold',
            'display' : 'inline-block', "font-size" : "30px"}), justify = "center"),

    html.Center([
        html.H2("BETA:", style={'textAlign':'center', 'display': 'inline-block', 'font-family': 'Arial Rounded MT Bold'}),
        html.H2(id="alpha", style={'display': 'inline-block','font-family': 'Arial Rounded MT Bold', 'margin-left': '13px'}),
        html.H2("ALPHA:", style={'display': 'inline-block', 'font-family': 'Arial Rounded MT Bold', 'margin-left' : "20px"}),
        html.H2(id='beta', style={'display': 'inline-block', 'font-family': 'Arial Rounded MT Bold', 'margin-left' : '13px'})
    ]),
    html.Br(),

    dcc.Graph(id="ab-plot"),

    html.Br(),
    html.Br(),

    dbc.Row([dbc.Col(),
             dbc.Col(controls4, md=4, style={'text-align': 'center'}),
             dbc.Col()], align="center", justify='center',
            style={'text-align': 'center'}),

    html.Br(),

    html.Center([html.H2("HOEFFDING:", style={'font-family': 'Arial Rounded MT Bold',
            'display' : 'inline-block', "font-size" : "30px"}),
               html.H2(id="fin", style={'display': 'inline-block','font-family': 'Arial Rounded MT Bold', 'margin-left': '13px'})]),

    html.Br(),

    dbc.Row(html.Center("REACTIVE GRAPH", style={'font-family': 'Arial Rounded MT Bold',
                                                 'display': 'inline-block', "font-size": "30px"}), justify="center"),
    html.Br(),

    dbc.Row(html.Img(src='assets/graph.png')),

    html.Br()

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
    Output("history-tbl", "data"),
    Input("run-query", "n_clicks"),
    [State('benchmark-id', 'value'), State('date-id', 'start_date'), State('date-id', 'end_date')],
    prevent_initial_call=True
)
def query_refinitiv(n_clicks, asset, start_date_id, end_date_id):
    prices, prc_err = ek.get_data(
        instruments=asset,
        fields=[
            'TR.OPENPRICE(Adjusted=0)',
            'TR.HIGHPRICE(Adjusted=0)',
            'TR.LOWPRICE(Adjusted=0)',
            'TR.CLOSEPRICE(Adjusted=0)',
            'TR.PriceCloseDate'
        ],
        parameters={
            'SDate': start_date_id,
            'EDate': end_date_id,
            'Frq': 'D'
        }
    )

    divs, div_err = ek.get_data(
        instruments=asset,
        fields=[
            'TR.DivExDate',
            'TR.DivUnadjustedGross',
            'TR.DivType',
            'TR.DivPaymentType'
        ],
        parameters={
            'SDate': start_date_id,
            'EDate': end_date_id,
            'Frq': 'D'
        }
    )

    splits, splits_err = ek.get_data(
        instruments=asset,
        fields=['TR.CAEffectiveDate', 'TR.CAAdjustmentFactor'],
        parameters={
            "CAEventType": "SSP",
            'SDate': start_date_id,
            'EDate': end_date_id,
            'Frq': 'D'
        }
    )

    prices.dropna(inplace=True)
    prices['Date'] = pd.to_datetime(prices['Date']).dt.date

    divs.rename(
        columns={
            'Dividend Ex Date': 'Date',
            'Gross Dividend Amount': 'div_amt',
            'Dividend Type': 'div_type',
            'Dividend Payment Type': 'pay_type'
        },
        inplace=True
    )
    # divs.dropna(inplace=True)
    divs['Date'] = pd.to_datetime(divs['Date']).dt.date
    divs = divs[(divs.Date.notnull()) & (divs.div_amt > 0)]

    divs = divs.groupby(['Instrument', 'Date'], as_index=False).agg({
        'div_amt': 'sum',
        'div_type': lambda x: ", ".join(x),
        'pay_type': lambda x: ", ".join(x)
    })

    splits.rename(
        columns={
            'Capital Change Effective Date': 'Date',
            'Adjustment Factor': 'split_rto'
        },
        inplace=True
    )
    splits.dropna(inplace=True)
    splits['Date'] = pd.to_datetime(splits['Date']).dt.date

    unadjusted_price_history = pd.merge(
        prices, divs[['Instrument', 'Date', 'div_amt']],
        how='outer',
        on=['Date', 'Instrument']
    )
    unadjusted_price_history['div_amt'].fillna(0, inplace=True)

    unadjusted_price_history = pd.merge(
        unadjusted_price_history, splits,
        how='outer',
        on=['Date', 'Instrument']
    )
    unadjusted_price_history['split_rto'].fillna(1, inplace=True)
    unadjusted_price_history.drop_duplicates(inplace=True)

    return unadjusted_price_history.to_dict('records')

@app.callback(
    Output('blotter', "data"),
    [Input("params", "n_clicks"), Input('history-tbl', 'data')],
    [State('benchmark-id', 'value'), State('alpha1-id', 'value'), State('alpha2-id', 'value'), State('n1-id', 'value'), State('n2-id', 'value')],
    prevent_initial_call=True
)
def make_blotter(n_clicks, history, asset, alpha_1, alpha_2, n_1, n_2):
    ivv_prc = pd.DataFrame(history)
    ivv_prc['Date'] = pd.to_datetime(ivv_prc['Date']).dt.date

    ##### Get the next business day from Refinitiv!!!!!!!
    rd.open_session()

    next_business_day = rd.dates_and_calendars.add_periods(
        start_date=ivv_prc['Date'].iloc[-1].strftime("%Y-%m-%d"),
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
                ivv_prc['Low Price'].iloc[1:][::-1].rolling(n1).min()[
                ::-1].to_numpy(),
                submitted_entry_orders['price'].to_numpy()
            )
        ].copy()
    cancelled_entry_orders.reset_index(drop=True, inplace=True)
    cancelled_entry_orders['status'] = 'CANCELLED'
    cancelled_entry_orders['date'] = pd.DataFrame(
        {'cancel_date': submitted_entry_orders['date'].iloc[(n1 - 1):].to_numpy()},
        index=submitted_entry_orders['date'].iloc[:(1 - n1)].to_numpy()
    ).loc[cancelled_entry_orders['date']]['cancel_date'].to_list()

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

        ivv_slice = ivv_prc.iloc[idx1:(idx1 + n1)]['Low Price']

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

    if any(filled_entry_orders['status'] == 'LIVE'):
        live_entry_orders = pd.concat([
            filled_entry_orders[filled_entry_orders['status'] == 'LIVE'],
            live_entry_orders
        ])
        # "today" is the next business day after the last closing price
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


    # for every filled entry order, there must exist a submitted exit order:
    submitted_exit_orders = filled_entry_orders.copy()
    submitted_exit_orders['trip'] = 'EXIT'
    submitted_exit_orders['action'] = 'SELL'
    submitted_exit_orders['price'] = submitted_exit_orders['price'] * (1 + alpha2)
    submitted_exit_orders['status'] = 'SUBMITTED'

    # Figure out what happened to each exit order we submitted
    exit_order_fates = submitted_exit_orders.copy()
    exit_mkt_orders = pd.DataFrame(columns=exit_order_fates.columns)

    for index, exit_order in submitted_exit_orders.iterrows():

        # was it filled the day it was submitted?
        if float(
                ivv_prc.loc[ivv_prc['Date'] == exit_order['date'], 'Close Price']
        ) >= exit_order['price']:
            exit_order_fates.at[index, 'status'] = 'FILLED'
            continue

        window_prices = ivv_prc[ivv_prc['Date'] > exit_order['date']].head(n2)

        # was it submitted on the very last day for which we have data?
        if window_prices.size == 0:
            exit_order_fates.at[index, 'date'] = pd.to_datetime(
                next_business_day).date()

            exit_order_fates.at[index, 'status'] = 'LIVE'
            continue

        filled_ind, *asdf = np.where(
            window_prices['High Price'] >= exit_order['price']
        )

        if filled_ind.size == 0:

            if window_prices.shape[0] < n2:
                exit_order_fates.at[index, 'date'] = pd.to_datetime(
                    next_business_day).date()

                exit_order_fates.at[index, 'status'] = 'LIVE'
                continue

            exit_order_fates.at[index, 'date'] = window_prices['Date'].iloc[
                window_prices.shape[0] - 1
                ]
            exit_order_fates.at[index, 'status'] = 'CANCELLED'
            exit_mkt_orders = pd.concat([
                exit_mkt_orders,
                pd.DataFrame({
                    'trade_id': exit_order['trade_id'],
                    'date': window_prices['Date'].tail(1),
                    'asset': exit_order['asset'],
                    'trip': exit_order['trip'],
                    'action': exit_order['action'],
                    'type': "MKT",
                    'price': window_prices['Close Price'].tail(1),
                    'status': 'FILLED'
                })
            ])
            continue

        exit_order_fates.at[index, 'date'] = window_prices['Date'].iloc[
            min(filled_ind)
        ]
        exit_order_fates.at[index, 'status'] = 'FILLED'

    blotter = pd.concat(
        [entry_orders, submitted_exit_orders, exit_order_fates, exit_mkt_orders]
    ).sort_values(['trade_id', "date", 'trip']).reset_index(drop=True)

    return blotter.to_dict('records')


@app.callback(
    Output('ledger1', "data"),
    [Input('blotter', 'data')],
    prevent_initial_call=True
)
def make_ledger(blotter):

    blotter = pd.DataFrame(blotter)
    ledger_dict = {
        "trade_id": [],
        "asset": [],
        "dt_enter": [],
        "dt_exit": [],
        "success": [],
        "n": [],
        "rtn": []
    }

    ledger_dict['trade_id'] = list(set(blotter['trade_id']))
    enter = blotter.loc[
        (blotter['trip'] == "ENTER") & ((blotter['status'] == 'SUBMITTED') | (blotter['status'] == 'LIVE'))]
    enter = enter.drop_duplicates(subset = 'trade_id', keep = 'first')
    exit = blotter.loc[
        ((blotter['trip'] == 'ENTER') & ((blotter['status'] == 'CANCELLED') | (blotter['status'] == 'LIVE'))) | (
                    (blotter['trip'] == "EXIT") & ((blotter['status'] == 'FILLED') | (blotter['status'] == 'LIVE')))]
    success = [(exit['status'] == 'CANCELLED') * 0 + ((exit['status'] == 'FILLED') & (exit['type'] == 'MKT')) * -1 + (
                (exit['status'] == 'FILLED') & (exit['type'] == 'LMT')) * 1 + (exit['status'] == 'LIVE') * 3]
    exit_dates = exit['date']

    ledger_dict['asset'] = list(enter['asset'])
    ledger_dict['dt_enter'] = list(enter['date'])
    ledger_dict['success'] = np.array(success)[0]
    ledger_dict['dt_exit'] = exit_dates

    ent_dates = [datetime.strptime(date, '%Y-%m-%d').date() for date in ledger_dict['dt_enter']]
    ext_dates = [datetime.strptime(date, '%Y-%m-%d').date() for date in ledger_dict['dt_exit']]

    ledger_dict['n'] = np.add(np.busday_count(ent_dates, ext_dates), 1)

    entry_prices = np.array(enter['price'])
    exit_prices = np.array(exit['price'])

    returns = np.divide(np.log(np.divide(exit_prices, entry_prices)), ledger_dict['n'])
    ledger_dict['rtn'] = returns
    ledger = pd.DataFrame(ledger_dict)
    ledger.loc[ledger_dict['success'] == 0, ['dt_exit', 'n', 'rtn']] = ['', 3, '']
    ledger.loc[ledger_dict['success'] == 3, ['dt_exit', 'n', 'rtn', 'success']] = ['', '', '', '']
    return ledger.to_dict('records')

import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
@app.callback(
    Output('ledger2', "data"),
    [Input('ledger1', 'data'), Input('lookback-bt', 'n_clicks')],
    State('lookback', 'value'),
    prevent_initial_call=True
)
def perceptron(ledg, n_clicks, lookback):
    ledger = pd.DataFrame(ledg)
    lookback = int(lookback)
    data = pd.read_csv('hw4.csv')
    features = data[['ECRPUS 1Y Index', 'SPXSFRCS Index', 'FDTRFTRL Index']]
    predictions = {'date' : [], 'prediction': []}
    # Make a training set and let's try it out on two upcoming trades.
    # Choose a subset of data:
    for i,n in enumerate(ledg[:-int(lookback) - 3]):
        X = features.iloc[int(i):int(i)+lookback]
        x_test = features.iloc[[i+lookback]]
        y = np.asarray(ledger['success'][i:i+lookback], dtype="|S6")

        sc = StandardScaler()
        sc.fit(X)
        X_std = sc.transform(X)
        x_test_std = sc.transform(x_test)

        pca = PCA(n_components=1)
        X_train = pca.fit_transform(X_std)
        X_test = pca.transform(x_test_std)
        explained_variance = pca.explained_variance_ratio_

        ppn = Perceptron(eta0=0.1)
        ppn.fit(X_train, y)

        y_pred = ppn.predict(X_test)[0]
        predictions['date'] += [datetime.strptime(data['Date'][i+lookback], "%m/%d/%y").strftime("%Y-%m-%d")]
        predictions['prediction'] += [int(y_pred)]
    predictions = pd.DataFrame(predictions)
    predictions = ledger.merge(predictions, how = 'inner', left_on='dt_enter', right_on = 'date')
    predictions = predictions[['trade_id', 'asset', 'dt_enter', 'rtn', 'success', 'prediction']]
    same = np.where(np.equal(predictions['prediction'],predictions['success']))
    return predictions.to_dict('records')

import plotly.express as px
@app.callback(
    Output('ledger3', 'data'),
    [Input('ledger2', 'data'), Input('history-tbl', 'data')],
    prevent_initial_call=True
)
def ab_plot(ledger2, unadjusted_price_history):
    ledger2= pd.DataFrame(ledger2)

    dt_prc_div_splt = pd.DataFrame(unadjusted_price_history)

    # Define what columns contain the Identifier, date, price, div, & split info
    ins_col = 'Instrument'
    dte_col = 'Date'
    prc_col = 'Close Price'
    div_col = 'div_amt'
    spt_col = 'split_rto'

    dt_prc_div_splt[dte_col] = pd.to_datetime(dt_prc_div_splt[dte_col])
    dt_prc_div_splt = dt_prc_div_splt.sort_values([ins_col, dte_col])[
        [ins_col, dte_col, prc_col, div_col, spt_col]].groupby(ins_col)
    numerator = dt_prc_div_splt[[dte_col, ins_col, prc_col, div_col]].tail(-1)
    denominator = dt_prc_div_splt[[prc_col, spt_col]].head(-1)

    act_rtns = pd.DataFrame({
        'Date': numerator[dte_col].reset_index(drop=True),
        'Instrument': numerator[ins_col].reset_index(drop=True),
        'ivv_rtn_w_perceptron': np.log(
            (numerator[prc_col] + numerator[div_col]).reset_index(drop=True) / (
                    denominator[prc_col] * denominator[spt_col]
            ).reset_index(drop=True)
        )
    })
    print(act_rtns)
    start_date = ledger2['dt_enter'][1]
    end_date = ledger2['dt_enter'].to_numpy()[-1]
    start_index = act_rtns[act_rtns['Date']==start_date].index.values[0]
    end_index = act_rtns[act_rtns['Date']==end_date].index.values[0]
    act_rtns = act_rtns.iloc[start_index:end_index+1]
    act_rtns = act_rtns.reset_index()
    ledger2 = ledger2[1:].reset_index()
    pred_returns = act_rtns[ledger2['prediction']==1]
    Er = np.matmul(ledger2['prediction'], act_rtns['ivv_rtn_w_perceptron'])
    print(Er)
    pred_returns['Date'] = pred_returns['Date'].dt.strftime("%Y-%m-%d")
    ledger2 = ledger2[ledger2['rtn'] != '']
    ledger3 = ledger2.merge(pred_returns, how='inner', left_on = 'dt_enter', right_on = 'Date')
    ledger3['rtn'] = np.array(ledger3['rtn'], dtype = float)
    ledger3['ivv_rtn_w_perceptron'] = np.array(ledger3['ivv_rtn_w_perceptron'], dtype = float)
    ledger3 = ledger3[['Date', 'ivv_rtn_w_perceptron', 'rtn']]
    return ledger3.to_dict('records')

@app.callback(
    Output('ab-plot', 'figure'),
    Input('ledger3', 'data'),
    prevent_initial_call=True)
def ab(ledger3):
    ledger3 = pd.DataFrame(ledger3)
    plot = px.scatter(ledger3[['ivv_rtn_w_perceptron', 'rtn']], y='ivv_rtn_w_perceptron', x='rtn', trendline='ols')
    return plot

@app.callback(
    Output('alpha', 'children'),
    Output('beta', 'children'),
    Input('ab-plot', 'figure'),
    prevent_initial_call = True
)
def calc_alpha_beta(ab_plot):
    line = ab_plot['data'][1]['hovertemplate']
    index1 = line.find(" = ") + 2
    index2 = line.find(' * ')
    index3 = line.find(" + ") + 2
    index4 = line.find("<br>R")
    alpha = line[index1:index2]
    beta = line[index3:index4]
    return(alpha, beta)

@app.callback(
    Output('fin', 'children'),
    [Input('ledger3', 'data'), Input('b_bt', 'n_clicks')],
    [State('b', 'value'), State('alpha2-id', 'value')],
    prevent_initial_call = True
)
def hoeffding_test(returns, n_clicks, b, a):

    returns = pd.DataFrame(returns)
    original_sample = returns['rtn'].values
    mod_sample = returns['ivv_rtn_w_perceptron'].values
    # make sure geometric
    print(np.prod(original_sample))
    print(1/len(original_sample))
    print(np.prod(mod_sample))
    print(1/len(mod_sample))
    original_mean = np.power(np.prod(original_sample), 1/len(original_sample))
    mod_mean = np.power(np.prod(mod_sample), 1 / len(mod_sample))
    diff_means = original_mean - mod_mean

    # hoeffding, add in divide (b-a)^2

    n = len(original_sample)
    epsilon = abs(diff_means)
    prob = 2 * np.exp(-2 * n * epsilon ** 2)
    fin = prob / ((float(b) - float(a)) ** 2)

    # Compare probability to threshold
    # use threshold as historical low maybe
    # what else should this return?
    # are we overfitting or not? that is what hoeffdings tells us

    return str(fin) + "%"



if __name__ == '__main__':
    app.run_server(debug=True)