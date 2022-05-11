import tushare as ts  # 获取股票信息
import dash # 设计网站
from datetime import datetime, timedelta
import dash_core_components as dcc #设计网站采用的核心插件库
import dash_bootstrap_components as dbc #设计网站采用的美化页面
import dash_html_components as html  #设计网站采用的核心插件库
from dash.dependencies import Input, Output, State # 完成用户点击响应
import plotly.graph_objects as go # 画出趋势图
from dash.exceptions import PreventUpdate
import pandas as pd
# ID definitions
id_button_update_a_share_list = 'id-button-update-a-share-list'
id_dpl_share_code = 'id-dpl-share-code'
id_date_picker_range = 'id-date-picker-range'
id_button_query_trend = 'id-button-query-trend'
id_graph_hist_trend_graphic = 'id-graph-hist-trend-graphic'



# ts_token
ts_pro = ts.pro_api('XXXXXX') # 放入自己的tushare token

# default file to save stock codes list
A_STOCK_FILE = 'data/a_stock_list.csv'
# bootstrap file 放在assets 文件夹中


def update_a_share_list():
    # display a stock share list from csv
    a_share_list_df = pd.read_csv(A_STOCK_FILE, index_col='ts_code')
    share_dict = a_share_list_df.to_dict()['name']
    return share_dict_to_option(share_dict)


def get_A_stock_list():
    # fetch list from internet
    share_df = ts_pro.stock_basic(
        exchange='',
        list_status='L',
        fields='ts_code,name')
    share_df.set_index('ts_code', inplace=True)
    return share_df


def write_A_stock_list_to_csv(file):
    # sync to csv
    share_df = get_A_stock_list()
    share_df.to_csv(file)


def share_dict_to_option(share_dict):
    # convert name and code to options
    name_list = [str(key) + '-' + str(value)
                 for key, value in share_dict.items()]
    return list_to_option_list(name_list)


def split_share_to_code(share):
    # split options to get code
    code = share.split('-')[0]
    return code


def list_to_option_list(list):
    # quick create dropdown options from list
    return [{"label": i, "value": i} for i in list]


def get_trend_df(code, start_date, end_date):
    # get history tend by ts_code,start_date,end_datess
    df = ts_pro.daily(ts_code=code, start_date=start_date, end_date=end_date)
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df.set_index('trade_date', inplace=True)
    df.sort_index(inplace=True)  # sort result by date (datetime)
    return df


def plot_candlestick(df):
    # plot candlestick
    # customize the color to China Stock
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                increasing=dict(
                    line=dict(
                        color="#FF0000")),
                decreasing=dict(
                    line=dict(
                        color="#00FF00")),
            )])

    fig.update_layout(
        margin=dict(l=2, r=10, t=20, b=20),
        paper_bgcolor="LightSteelBlue",
    )  # change the layout of figure
    return fig


# create dash app
app = dash.Dash(__name__)

# layout design
header = html.H2("股票查询", className='text-center')
# update list button
update_a_share_button = dbc.Button(
    id=id_button_update_a_share_list,
    color='light',
    children='更新列表', outline=True)
# share select dropdown list
select_share = dcc.Dropdown(
    id=id_dpl_share_code,
    options=update_a_share_list()
)
# datetime picker
date_pick = dcc.DatePickerRange(
    id='id-date-picker-range',
    max_date_allowed=datetime.today(),
    start_date=datetime.today().date() - timedelta(days=60),
    end_date=datetime.today().date()
)
# query button
query_button = dbc.Button(
    "查询",
    color="warning",
    className="mr-1",
    id=id_button_query_trend,
    outline=False)

# make a better row/col layout
select_share_row = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [select_share],
                    className='col-5'),
                dbc.Col(
                    [update_a_share_button],
                    className='col-2')]),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    [date_pick],
                    className='col-5'),
                dbc.Col(
                    [query_button],
                    className='col-1')])])

# get default figure
default_fig = plot_candlestick(
    get_trend_df(
        '000001.SZ',
        start_date='20200101',
        end_date='20200707'))
# graphic div
graphic_div = dbc.Container(
    [dcc.Graph(figure=default_fig, id=id_graph_hist_trend_graphic)])

# fully layout
app.layout = html.Div([
    header,
    select_share_row,
    html.Br(),
    graphic_div,
])

# callback function to update plot


@app.callback(
    Output(id_graph_hist_trend_graphic, 'figure'),
    [Input(id_button_query_trend, 'n_clicks')],
    [State(id_dpl_share_code, 'value'),
     State(id_date_picker_range, 'start_date'),
     State(id_date_picker_range, 'end_date')]
)
def update_output_div(query, share, start, end):
    if query is not None:
        share_code = split_share_to_code(share)
        start_str = start.replace("-", "")
        end_str = end.replace("-", "")
        return plot_candlestick(
            get_trend_df(
                share_code,
                start_date=start_str,
                end_date=end_str))
    else:
        raise PreventUpdate

# callback to update the options


@app.callback(
    Output(id_dpl_share_code, 'options'),
    [Input(id_button_update_a_share_list, 'n_clicks')]
)
def update_output_div(update):
    if update is not None:
        write_A_stock_list_to_csv(A_STOCK_FILE)
        return update_a_share_list()
    else:
        raise PreventUpdate


if __name__ == '__main__':
    app.run_server(debug=True)
