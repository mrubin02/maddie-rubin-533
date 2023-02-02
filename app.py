from dash import Dash, html, dcc, dash_table, Input, Output, State
import refinitiv.dataplatform.eikon as ek
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import os

ek.set_app_key(os.getenv('EIKON_API'))

#dt_prc_div_splt = pd.read_csv('unadjusted_price_history.csv')

app = Dash(__name__)
app.layout = html.Div([
    html.Div([
        html.H2('Benchmark:',
                style={ 'font-family': 'Arial Rounded MT Bold',
                       'display' : 'inline-block'}),
        dcc.Input(id = 'benchmark-id',
                  type = 'text',
                  value="IVV",
                  style={
                        "margin-left" : "15px",
                         'font-family': 'Arial Rounded MT Bold',
                        'font-size' : '25px',
                         'width' : '284px'})]),
    html.Div([
        html.H2('Asset:',
                style={'font-family': 'Arial Rounded MT Bold',
                       'display' : 'inline-block'}),
        dcc.Input(id='asset-id',
                  type='text',
                  value="AAPL.O",
                  style={
                      'font-family': 'Arial Rounded MT Bold',
                         'margin-left' : '82px',
                         'font-size' : '25px',
                         'width' : '284px'})]),
    html.Div([
        html.H2('Date Range:',
                style={'font-family': 'Arial Rounded MT Bold',
                       'display' : 'inline-block'}),
        dcc.DatePickerRange(
            id = 'date-id',
            max_date_allowed=datetime.now().strftime("%Y-%m-%d"),
            style={'background-color' : 'LavenderBlush',
                    'margin-left' : '13px',
                   'font-family': 'Arial Rounded MT Bold',
                   'width' : '286px'}
        )]),

    html.Br(),

    html.Button('QUERY Refinitiv',
                id = 'run-query',
                n_clicks = 0,
                style={'background-color' : 'pink',
                    'font-size' : '25px',
                    'margin-left': '155px',
                    'font-family': 'Arial Rounded MT Bold',
                       'width' : '286px'}),
    html.Br(),
    html.Br(),
    html.H2('Raw Data from Refinitiv',
            style={'font-family': 'Arial Rounded MT Bold'}),

    dash_table.DataTable(
        id = "history-tbl",
        page_action='none',
        style_table={'height': '300px', 'overflowY': 'auto'}
    ),
    html.H2('Historical Returns', style={'font-family': 'Arial Rounded MT Bold'}),
    dash_table.DataTable(
        id = "returns-tbl",
        page_action='none',
        style_table={'height': '300px', 'overflowY': 'auto'}
    ),
    html.H2('Alpha & Beta Scatter Plot', style={'font-family': 'Arial Rounded MT Bold'}),
    html.Div([
                html.H2('Plot Range:',
                        style={'font-family': 'Arial Rounded MT Bold',
                               'display' : 'inline-block'}),
                dcc.DatePickerRange(
                    id = 'plot-date-id',
                    max_date_allowed=datetime.now().strftime("%Y-%m-%d"),
                    style={'background-color' : 'LavenderBlush',
                            'margin-left' : '13px',
                           'font-family': 'Arial Rounded MT Bold',
                           'width' : '286px'}
                )]),
    html.Div([
        html.H2("Beta:", style={'textAlign':'center', 'display': 'inline-block', 'font-family': 'Arial Rounded MT Bold'}),
        html.H2(id="alpha", style={'display': 'inline-block','font-family': 'Arial Rounded MT Bold', 'margin-left': '13px'}),
        html.H2("Alpha:", style={'display': 'inline-block', 'font-family': 'Arial Rounded MT Bold', 'margin-left' : "20px"}),
        html.H2(id='beta', style={'display': 'inline-block', 'font-family': 'Arial Rounded MT Bold', 'margin-left' : '13px'})
    ]),
    dcc.Graph(id="ab-plot"),
    html.P(id='summary-text', children="")],
    style={'background-color' : 'PowderBlue'})
@app.callback(
    Output("history-tbl", "data"),
    Output('plot-date-id', 'min_date_allowed'),
    Output('plot-date-id', 'start_date'),
    Output('plot-date-id', 'end_date'),
    Output('plot-date-id', 'max_date_allowed'),
    Input("run-query", "n_clicks"),
    [State('benchmark-id', 'value'), State('asset-id', 'value'), State('date-id', 'start_date'), State('date-id', 'end_date')],
    prevent_initial_call=True
)
def query_refinitiv(n_clicks, benchmark_id, asset_id, start_date_id, end_date_id):
    assets = [benchmark_id, asset_id]
    prices, prc_err = ek.get_data(
        instruments=assets,
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
        instruments=assets,
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
        instruments=assets,
        fields=['TR.CAEffectiveDate', 'TR.CAAdjustmentFactor'],
        parameters={
            "CAEventType": "SSP",
            'SDate': start_date_id,
            'EDate': end_date_id,
            'Frq': 'D'
        }
    )

    prices.rename(
        columns={
            'Open Price': 'open',
            'High Price': 'high',
            'Low Price': 'low',
            'Close Price': 'close'
        },
        inplace=True
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
    #divs.dropna(inplace=True)
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

    if unadjusted_price_history.isnull().values.any():
        raise Exception('missing values detected!')

    return(unadjusted_price_history.to_dict('records'), start_date_id, start_date_id, end_date_id, end_date_id)

@app.callback(
    Output("returns-tbl", "data"),
    Input("history-tbl", "data"),
    prevent_initial_call = True
)
def calculate_returns(history_tbl):

    dt_prc_div_splt = pd.DataFrame(history_tbl)

    # Define what columns contain the Identifier, date, price, div, & split info
    ins_col = 'Instrument'
    dte_col = 'Date'
    prc_col = 'close'
    div_col = 'div_amt'
    spt_col = 'split_rto'

    dt_prc_div_splt[dte_col] = pd.to_datetime(dt_prc_div_splt[dte_col])
    dt_prc_div_splt = dt_prc_div_splt.sort_values([ins_col, dte_col])[
        [ins_col, dte_col, prc_col, div_col, spt_col]].groupby(ins_col)
    numerator = dt_prc_div_splt[[dte_col, ins_col, prc_col, div_col]].tail(-1)
    denominator = dt_prc_div_splt[[prc_col, spt_col]].head(-1)

    return(
        pd.DataFrame({
        'Date': numerator[dte_col].reset_index(drop=True),
        'Instrument': numerator[ins_col].reset_index(drop=True),
        'rtn': np.log(
            (numerator[prc_col] + numerator[div_col]).reset_index(drop=True) / (
                    denominator[prc_col] * denominator[spt_col]
            ).reset_index(drop=True)
        )
    }).pivot_table(
            values='rtn', index = 'Date', columns='Instrument'
        ).to_dict('records')
    )

@app.callback(
    Output("ab-plot", "figure"),
    [Input("returns-tbl", "data"), Input('plot-date-id', 'start_date'), Input('plot-date-id', 'end_date')],
    [State('benchmark-id', 'value'), State('history-tbl', 'data'), State('asset-id', 'value'), State('date-id', 'start_date'), State('date-id', 'end_date')],
    prevent_initial_call = True
)
def render_ab_plot(returns, plot_start, plot_end, benchmark_id, history, asset_id, start_date, end_date):

    dates = [i['Date'] for i in history]
    try:
        end_ind = dates.index(plot_end)
    except:
        try:
            end_ind = dates.index(plot_end[0:-1] + str(int(plot_end[-1]) - 3))
        except:
            end_ind = dates.index(plot_end[0:-1] + str(int(plot_end[-1]) + 1))

    try:
        start_ind = dates.index(plot_start)
    except:
        try:
            start_ind = dates.index(plot_start[0:-1] + str(int(plot_start[-1]) + 3))
        except:
            start_ind = dates.index(plot_start[0:-1] + str(int(plot_start[-1]) - 1))

    plotrets = returns[start_ind:end_ind+1]
    return(
        px.scatter(plotrets, x=benchmark_id, y=asset_id, trendline='ols')
    )

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

if __name__ == '__main__':
    app.run_server(debug=True)