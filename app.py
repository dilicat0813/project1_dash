#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Install modules Area

# !pip install dash
# !pip install jupyter-dash
# !pip install dash-bootstrap-components
# !pip install statsmodels
# !pip install catboost

## Import Modules Area
import pandas as pd
import dash
import plotly.express as px
from jupyter_dash import JupyterDash
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from catboost import CatBoostClassifier
import base64


# In[2]:


# DataFrame 정의 Area

# import year_earn, buyer form csv file
year_earn = pd.read_csv('data/year_earn.csv')
buyer = pd.read_csv('data/buyer.csv')

#스캐터 5개 그래프 
scatter5 = pd.read_csv('data/scatter5.csv')
scatter5_names = ['충전실온도', '실링온도', '쿠킹온도', '쿠킹스팀압력', '실링압력'] #dropdown 목록 지정을 위한 정의
# fig1 = px.scatter(scatter5, x = scatter5['생산일자'], y = scatter5['쿠킹스팀압력'], color = scatter5['오류발생여부'])
# fig2 = px.scatter(scatter5, x = scatter5['생산일자'], y = scatter5['충전실온도'], color = scatter5['오류발생여부'])
# fig3 = px.scatter(scatter5, x = scatter5['생산일자'], y = scatter5['실링온도'], color = scatter5['오류발생여부'])
# fig4 = px.scatter(scatter5, x = scatter5['생산일자'], y = scatter5['쿠킹온도'], color = scatter5['오류발생여부'])
# fig5 = px.scatter(scatter5, x = scatter5['생산일자'], y = scatter5['실링압력'], color = scatter5['오류발생여부'])

#파이그래프1(110)
pie1 = pd.read_csv('data/pie1.csv')
pie1 = pie1.drop('name_y', axis=1)
pie1.rename(columns={'name_x': 'name'}, inplace=True) #간단한 드롭
pie1= pie1.groupby('name')['Error_class'].count()
# fig_pie1 = px.pie(pie1, values='Error_class', names=pie1.index)

#파이그래프2(1074)
pie2 = pd.read_csv('data/errored_pie.csv')
pie2 = pie2.groupby('name')['Error_class'].count()
#fig_pie2 = px.pie(pie2, values='Error_class', names=pie2.index)

#연단위 오류 발생률
Error_per = pd.read_csv('data/Error_per.csv')
Error_per['per'] = Error_per['per'].round(2)
# fig = px.line(Error_per, x="year", y="per", text="per")
# fig.update_traces(textposition="bottom right")

#바이올린 차트_생산시간
violin_cooking = pd.read_csv('data/violin_cooking.csv')
#vio1 = px.violin(violin_cooking, box=True, y='생산시간')

#바이올린 차트_오류발생시간
violin_error = pd.read_csv('data/violin_error.csv')
#vio2 = px.violin(violin_error, box=True, y='오류조치시간_보정')

#HMR 시장 확대 
hmrexpand = pd.read_csv('data/hmrexpand.csv')

#간편식 HMR 구입 변화 예상
hmrchange = pd.read_csv('data/hmr_change_predict.csv')
fig3 = px.pie(hmrchange, values='values', names='names')
fig3.update_traces(textposition='inside', textinfo='percent+label')
fig3.update_layout(
    annotations=[dict(x=0.5, y=1.2, showarrow=False, text='향후 1년 HMR 구입 변화 예상', font=dict(size=20))],
    margin=dict(t=100)
)

#라인2개 그래프 & 상관관계 그래프 데이터 프레임
cooking_error_quarter = pd.read_csv('data/cooking_error_quarter.csv')

#상관관계
# fig = px.scatter(cooking_error_quarter, x=cooking_error_quarter["생산수"], y=cooking_error_quarter["오류발생여부"], trendline="ols")
# fig.update_layout(title_text="분기별 생산품목과 오류발생")

#라인2개 그래프
# fig = make_subplots(specs=[[{"secondary_y": True}]])
# fig.add_trace(
#     go.Scatter( x = cooking_error_quarter["quarter"], y = cooking_error_quarter["생산수"],
#     mode = 'lines', name = '분기별 간편식 생산품목 수'),
#     secondary_y=False,)
# fig.add_trace(
#     go.Scatter( x = cooking_error_quarter["quarter"], y = cooking_error_quarter["오류발생여부"],
#     mode = 'lines', name = '분기별 오류발생 현황'),
#     secondary_y=True,)
# # Add figure title
# fig.update_layout(
#     title_text="분기별 생산품목과 오류발생")
# fig.update_xaxes(title_text="분기")
# fig.update_yaxes(title_text="<b>생산품목</b>", secondary_y=False)
# fig.update_yaxes(title_text="<b>오류발생</b>", secondary_y=True)

#f1-score table
f1_score_df = pd.read_csv('data/f1_score.csv')

# In[3]:


#모델 평가 부분 불러오기
from cb_model import train_score, test_score, y_pred, report, importance, X, savemodel
savemodel

#CatBoost 학습 모델 load Area
model = CatBoostClassifier()
model.load_model('model/model.dump') 


# In[4]:

# In[5]:


# function/module/compoenets def area

def Header2(name, app):
    title = html.H2(name, style={"margin-top": 5})
    return dbc.Row([dbc.Col(title, md=9)])
#style={'text-align': 'center', 'font-size': '20px'}
def Header1(name, app):
    title = html.H1(name, style={"margin-top": 5})
    return dbc.Row([dbc.Col(title, md=9)])

# Card components

# Basement Cards
cards = [
    dbc.Card(
        [
            html.H2(f"+{5/45*100:.2f}%", className="card-title"),
            html.P("HMR 시장규모 증가(2021~2022)", className="card-text"),
        ],
        body=True,
        color="#334557",
        inverse=True,
    ),
    dbc.Card(
        [
            html.H2(f"-38%", className="card-title"),
            html.P("수주금액 변동추이(2018~2021)", className="card-text"),
        ],
        body=True,
        color="#588195",
        inverse=True,
    ),
]

# Analyze Cards
cards2 = [
    dbc.Card(
        [
            html.H2(f"상관관계↑", className="card-title"),
            html.P("즉석밥 생산수와 오류발생", className="card-text"),
        ],
        body=True,
        color="#334557",
        inverse=True,
    ),
    dbc.Card(
        [
            html.H2(f"5% 이하", className="card-title"),
            html.P("최근 오류발생률", className="card-text"),
        ],
        body=True,
        color="#588195",
        inverse=True,
    ),
]

cards4 = [
    dbc.Card(
        [
            html.H2(f"75분 / 44분 ", className="card-title"),
            html.P("평균 생산시간/오류발생시간", className="card-text"),
        ],
        body=True,
        color="#334557",
        inverse=True,
    ),
    dbc.Card(
        [
            html.H2(f"0:정상 / 1:공정전 / 2:공정중 / 3:기타", className="card-title"),
            html.P("오류메세지 분류", className="card-text"),
        ],
        body=True,
        color="#588195",
        inverse=True,
    ),
]

cards5 = [
    dbc.Card(
        [
            html.H2(f"충전실온도 / 실링온도 / 쿠킹온도 / 쿠킹스팀압력 / 실링압력", className="card-title"),
            html.P("오류발생 현황과 주 요인", className="card-text"),
        ],
        body=True,
        color="#334557",
        inverse=True,
    ),
]

# Predict Cards(추후 추가)
cards3 = [
    dbc.Card(
        [
            html.H2(f"{train_score:.4f} / {test_score:.4f}", className="card-title"),
            html.P("Train / Test Score", className="card-text"),
        ],
        body=True,
        color="#334557",
        inverse=True,
    ),
    dbc.Card(
        [
            html.H2(f"{f1_score_df.iloc[4,2]}", className="card-title"),
            html.P("Weighted Avg f1-score", className="card-text"),
        ],
        body=True,
        color="#588195",
        inverse=True,
    ),
    dbc.Card(
        [
            html.H2(f"CatBoost Classifier", className="card-title"),
            html.P("Machine Learning Model", className="card-text"),
        ],
        body=True,
        color="#EFF0F2",
        inverse=False,
    ),
]

# Graph components
# Basement_Graphs
graphs1 = [[dcc.Graph(id="update_graph"), #1행 1열
            dcc.RangeSlider(hmrexpand['year'].min(),
                            hmrexpand['year'].max(),
                            value=[hmrexpand['year'].min(), hmrexpand['year'].max()], #수정사항, 슬라이더 초기값(처음값과 끝값)
                            marks={str(year): str(year) for year in hmrexpand['year'].unique()}, #수정사항 RangeSlider, Slider 모두에 적용
                            id='slider1')]
            , [dcc.Graph(id="update_graph2", figure=fig3)]] #1행 2열

graphs2 = [[dcc.Graph(id='booking-with-graph'), #2행 1열
            dcc.RangeSlider(id='year-slider',
                            min=year_earn['year'].min(),
                            max=year_earn['year'].max(),
                            value=[year_earn['year'].min(), year_earn['year'].max()],
                            marks={str(year): str(year) for year in year_earn['year'].unique()})],
           [dcc.Graph(id='yearly-transaction-graph'), #2행 2열
            dcc.RangeSlider(id='year-slider2',
                            min= buyer['year'].min(),
                            max= buyer['year'].max(),
                            step=None,
                            value=[buyer['year'].min(), buyer['year'].max()],
                            marks={str(year): str(year) for year in buyer['year'].unique()})
            ]]
        
# Analyze Graph

#1행

graphs3 = [[dcc.Graph(id="lines"),
            dcc.Dropdown(id='lines-select',
                         options=[{'label':'분기별 생산품목과 오류발생', 'value':'fig_line1'},
                                  {'label':'생산수에 따른 오류발생 여부', 'value':'fig_line2'}],
                         value='fig_line1'
                         )],
            [dcc.Graph(id='Error_per'), 
            dcc.RangeSlider(id='year-per',
                            min=Error_per['year'].min(),
                            max=Error_per['year'].max(),
                            value=[Error_per['year'].min(), Error_per['year'].max()],
                            marks={str(year): str(year) for year in Error_per['year'].unique()})
            ]]  

#2행
graphs4 = [[dcc.Graph(id="vios2"),
            dcc.Dropdown(id='violin-select',
                         options = [{'label':'평균 생산시간', 'value':'평균 생산시간'},
                                    {'label':'평균 오류발생시간', 'value':'평균 오류발생시간'}],
                         value='평균 생산시간'
                         )],
            [dcc.Graph(id="pies"), 
            dcc.Dropdown(id='pie-select',
                         options=[{'label':'오류메세지 분류(총 110종)', 'value':'fig_pie1'},
                                  {'label':'전체오류(1074건) 재분류 후 분포', 'value':'fig_pie2'}],
                         value='fig_pie1'
                         )]
            ]

#3행
graphs5 = [[dcc.Graph(id="scatter-5"),
            dcc.Dropdown(id='yvar_name',
                     options=scatter5_names,
                     value=scatter5_names[0],
                     placeholder='Select X-axis column')]  
           ]


# Predict Graph
radio_items1 = dcc.RadioItems(
    id='barr',
    options=[
        {'label': 'Train/Test Score', 'value': 'bar_1'},
        {'label': 'Catboost Multiclass', 'value': 'bar_2'}
    ],
    value='bar_2'
)

graphs6 = [
    dcc.Graph(id="bars")
]

#hr_style def
hr_style = {
    'border': '3px solid black' , # 선 두께와 색상을 지정합니다.
}

# 메뉴바 정의
menu = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Current Situation", href="/basement")),
        dbc.NavItem(dbc.NavLink("Analysis", href="/analyze")),
        dbc.NavItem(dbc.NavLink("Modeling/Predict", href="/predict")),
    ],
    brand="불량예측 모델을 통한 수익성 개선, 알파코 AI엔지니어 부트캠프 5기, 5조",
    brand_href="/",
    sticky="top",
    color="black",
    dark=True,
)

#이미지 정의부

image_filename1 = 'data/conclusion.png'
encoded_image1 = base64.b64encode(open(image_filename1, 'rb').read())


# In[6]:


#페이지별 레이아웃 정의 Area

app = JupyterDash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

# 각 페이지별 레이아웃
basement_layout = dbc.Container(
    [
        html.Br(),
        Header1("Current Situation", app),
        html.Br(),
        dbc.Row([dbc.Col(card) for card in cards]),
        html.Hr(),
        dbc.Row([dbc.Col(graph) for graph in graphs1]),
        html.Br(),
        dbc.Row([dbc.Col(graph) for graph in graphs2]),
        html.Br(),
    ], 
    fluid=False,
)

analyze_layout = dbc.Container(
    [
        html.Br(),
        Header1("Analysis", app),
        html.Br(),
        dbc.Row([dbc.Col(card) for card in cards2]),
        html.Hr(),        
        dbc.Row([dbc.Col(graph) for graph in graphs3]), #그래프4 (파이)
        html.Br(),
        dbc.Row([dbc.Col(card) for card in cards4]),
        html.Hr(),
        dbc.Row([dbc.Col(graph) for graph in graphs4]), #그래프4 (파이)
        html.Br(),
        dbc.Row([dbc.Col(card) for card in cards5]),
        html.Hr(),
        dbc.Row([dbc.Col(graph) for graph in graphs5]), #그래프5 (스캐터)
        html.Br(),
    ], 
    fluid=False,
)

predict_layout = dbc.Container(
    [
        html.Br(),
        Header1("Modeling", app),
        html.Br(),
        dbc.Row([dbc.Col(card) for card in cards3]),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        radio_items1,
                        dcc.Graph(id="bars"),
                    ],
                    md=6, # 1행 2열에서 오른쪽 열
                    align='center', 
                ),
                dbc.Col(
                    [
                        html.Img(id='image-display', className='img-fluid', width=600, src='data:image/png;base64,{}'.format(encoded_image1.decode())),                                                
                    ], 
                    md=6, # 1행 2열에서 왼쪽 열
                    align='center', 
                ),
                
            ],
             justify='center' 
        ),
        html.Br(),
        html.Br(),
        html.Hr(), #hr_style 적용 예시 
        Header1("Predict", app),
        html.Br(), 
        html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label('충전실 온도'),
                                dcc.Slider(id='input1', min=68, max=75, value=71, step=0.1, marks={68: '68', 69: '69', 70: '70', 71: '71', 72: '72', 73: '73', 74: '74', 75: '75'})
                            ],
                            md=6,
                            style={'padding': '10px'}
                        ),
                        dbc.Col(
                            [
                                html.Label('실링 온도'),
                                dcc.Slider(id='input2', min=65, max=141, value=101, step=0.1, marks={65: '65', 75: '75', 85: '85', 95: '95', 105: '105', 115: '115', 125: '125', 135: '135', 141: '141'})
                            ],
                            md=6,
                            style={'padding': '10px'}
                        )
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label('쿠킹 온도'),
                                dcc.Slider(id='input3', min=100, max=177, value=137, step=0.1, marks={100: '100', 110: '110', 120: '120', 130: '130', 140: '140', 150: '150', 160: '160', 170: '170', 177: '177'})
                            ],
                            md=6,
                            style={'padding': '10px'}
                        ),
                        dbc.Col(
                            [
                                html.Label('쿠킹 스팀압력'),
                                dcc.Slider(id='input4', min=22, max=25, value=23.5, step=0.01, marks={22: '22', 22.5: '22.5', 23: '23', 23.5: '23.5', 24: '24', 24.5: '24.5', 25: '25'})
                            ],
                            md=6,
                            style={'padding': '10px'}
                        )
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label('실링 압력'),
                                dcc.Slider(id='input5', min=192, max=223, value=208, step=0.1, marks={192: '192', 195: '195', 200: '200', 205: '205', 210: '210', 215: '215', 220: '220', 223: '223'})
                            ],
                            md=6,
                            style={'padding': '10px'}
                        ),
                        dbc.Col(
                            [
                                html.Button(id='submit-button', n_clicks=0, children='결과확인', style={'text-align': 'center', 'font-size': '20px'}),                                
                            ],
                            md=6,
                            style={'padding': '20px', 'display': 'flex', 'justify-content': 'center'}
                        )
                    ]
                )
            ],            
        ),
        html.Br(),        
        html.Div(id='output', style={'text-align': 'center', 'font-size': '20px'}),
        html.Br()  
    ],
    fluid=False,
)


# In[7]:



# 전체 레이아웃 Area
app.layout = html.Div(
    [
        menu,
        dcc.Location(id="url"),
        html.Div(id="page-content"),
    ]
)


# In[8]:


#@Callback Area

# 각 페이지별 앱 콜백
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def render_page_content(pathname):
    if pathname == "/basement":
        return basement_layout
    elif pathname == "/analyze":
        return analyze_layout
    elif pathname == "/predict":
        return predict_layout
    else:
        return basement_layout
        


# In[9]:



#@Callback Area
#Basement Graph Callback list
    #그래프 1행 1열 콜백
@app.callback(
    Output(component_id='update_graph', component_property='figure'), 
    Input(component_id='slider1', component_property='value')
)
def update_graph(year):
    filtered_df = hmrexpand[(hmrexpand['year'] >= year[0]) & (hmrexpand['year'] <= year[1])] #수정사항, 슬라이더 값지정하는 임시 df생성
    fig1 = px.bar(filtered_df, x=filtered_df['year'], y=filtered_df['market'])
    fig1.update_layout(
    title={
        'text': "HMR Market Changes",
        'font': {'size': 20},
        'x': 0.5
    },
    xaxis_title="연도",
    yaxis_title="시장규모",
    font=dict(size=15)
)       
    return(fig1)


    #그래프 2행 1열 콜백
@app.callback(
    Output(component_id='booking-with-graph', component_property='figure'), 
    Input(component_id='year-slider', component_property='value')
)
def update_graph2(year):
    filtered_df2 = year_earn[(year_earn['year'] >= year[0]) & (year_earn['year'] <= year[1])]
    fig2 = px.bar(filtered_df2, x='year', y='수주금액')
    fig2.update_layout(
    title={
        'text': "연도별 수주금액추이",
        'font': {'size': 20},
        'x': 0.5
    },
    xaxis_title="연도",
    font=dict(size=15)
)
    return(fig2)           

    #그래프 2행 2열 콜백
@app.callback(
    Output(component_id='yearly-transaction-graph', component_property='figure'),
    Input(component_id='year-slider2', component_property='value')
)
def update_figure(selected_years):
    filtered_df3 = buyer[(buyer['year'] >= selected_years[0]) & (buyer['year'] <= selected_years[1])]
    fig4 = px.bar(filtered_df3, x=filtered_df3['year'], y=filtered_df3['거래처코드'])
    fig4.update_layout(
    title={
        'text': "연도별 거래건수 변동추이",
        'font': {'size': 20},
        'x': 0.5
    },
    xaxis_title="연도",
    yaxis_title="거래건수",
    font=dict(size=15)
)
    return fig4
     


# In[10]:


#Analyze Graph Callback list
    #그래프 1행 1열 콜백
@app.callback(
    Output(component_id='lines', component_property='figure'),
    Input(component_id='lines-select', component_property='value')
)
def update_graph1by1(selected_value):
    if selected_value == 'fig_line1':
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
        go.Scatter( x = cooking_error_quarter["quarter"], y = cooking_error_quarter["생산수"],
        mode = 'lines', name = '분기별 간편식 생산품목 수'),
        secondary_y=False,)
        fig.add_trace(
        go.Scatter( x = cooking_error_quarter["quarter"], y = cooking_error_quarter["오류발생여부"],
        mode = 'lines', name = '분기별 오류발생 현황'),
        secondary_y=True,)        
# # Add figure title
        fig.update_layout(title={'text': "분기별 생산품목과 오류발생", 'font': {'size': 20}, 'x': 0.5}, xaxis_title="분기", font=dict(size=15), legend=dict(x=0.17, y=1.15, font=dict(size=10), orientation="h"), xaxis_tickfont=dict(size=10))    
        fig.update_yaxes(title_text="생산품목", secondary_y=False, title_font=dict(size=20), title_standoff=10)
        fig.update_yaxes(title_text="오류발생", secondary_y=True, title_font=dict(size=20), title_standoff=10)
        return(fig)
    else:
        fig2 = px.scatter(cooking_error_quarter, x=cooking_error_quarter["생산수"], y=cooking_error_quarter["오류발생여부"], trendline="ols")
        fig2.update_layout(title={'text': "생산수에 따른 오류발생 여부", 'font': {'size': 20}, 'x': 0.5},
                           xaxis_title="생산품목 수",
                           yaxis_title="오류발생",
                           font=dict(size=15))
        return(fig2)    

    #그래프 1행 2열 콜백
@app.callback(
    Output(component_id='Error_per', component_property='figure'), 
    Input(component_id='year-per', component_property='value')
)
def update_graph1by2(year):
    filtered_year_per = Error_per[(Error_per['year'] >= year[0]) & (Error_per['year'] <= year[1])]
    fig_year_per = px.line(filtered_year_per, x='year', y='per', text='per')
    fig_year_per.update_traces(textposition="bottom right")
    fig_year_per.update_layout(
        title={'text': "연도별 오류발생률",
               'font': {'size': 20},
               'x': 0.5},
        xaxis_title="연도",
        yaxis_title="오류발생률",
        font=dict(size=15))
    return(fig_year_per)           

    #그래프 2행 1열 콜백
@app.callback(
    Output(component_id='vios2', component_property='figure'),
    Input(component_id='violin-select', component_property='value')
)
def update_graph2by1(selected_value):
    if selected_value == '평균 생산시간':
        vio1 = px.violin(violin_cooking, box=True, y='생산시간')
        vio1.update_layout(
        title={'text': "평균 생산시간",
               'font': {'size': 20},
               'x': 0.5},
        yaxis_title="생산시간",
        font=dict(size=15))
        return vio1
    else:
        vio2 = px.violin(violin_error, box=True, y='오류조치시간_보정')
        vio2.update_layout(
        title={'text': "오류조치 소요시간",
               'font': {'size': 20},
               'x': 0.5},
        yaxis_title="오류조치시간",
        font=dict(size=15))
        return vio2

    #그래프 2행 2열 콜백
@app.callback(
    Output(component_id='pies', component_property='figure'),
    Input(component_id='pie-select', component_property='value')
)
def update_graph2by2(selected_value):
    if selected_value == 'fig_pie1':
        fig_pie1 = px.pie(pie1, values='Error_class', names=pie1.index)
        fig_pie1.update_layout(title={'text': "오류메세지(110종) 분류",
                                      'font': {'size': 20},
                                      'x': 0.5},                                
                                font=dict(size=15))
        return fig_pie1
    else:
        fig_pie2 = px.pie(pie2, values='Error_class', names=pie2.index)
        fig_pie2.update_layout(title={'text': "전체오류(1074건) 재분류 후 분포",
                                'font': {'size': 20},
                                'x': 0.5},
                                font=dict(size=15))
        return fig_pie2
    
    #그래프 3행 1열 콜백
@app.callback(
    Output(component_id='scatter-5', component_property='figure'),
     Input(component_id='yvar_name', component_property='value')
)
def update_graph3by1(yvar):
    fig = px.scatter(scatter5, x=scatter5['생산일자'], y=yvar, color=scatter5['오류발생여부'])
    fig.update_layout(title={'text': "오류발생요인("+yvar+')에 따른 오류제품 분포',
                             'font': {'size': 20},
                             'x': 0.5}, font=dict(size=15))
    return(fig) 


# In[11]:



#Predict Graph Callback list
    #그래프 1행 1열 콜백      
@app.callback(
    Output(component_id="bars", component_property='figure'),
    Input(component_id='barr', component_property='value')
)
def update_graph(selected_value):
    if selected_value == 'bar_1':
       fig = go.Figure(data=[go.Bar(x=['Train', 'Test'], y=[train_score, test_score])])
       fig.update_layout(title='Train/Test Score', yaxis_title='Accuracy')
       return fig

    else:    
       fig2 = go.Figure([go.Bar(x=X.columns, y=importance)])
       fig2.update_layout(title='Catboost Multiclass', yaxis_title='온도와 압력')
       return fig2
    #그래프 1행 2열 콜백

    #그래프 2행 1열 콜백

    #그래프 2행 2열 콜백

    #그래프 3행 1열 콜백(예측 예제 콜백 전용 구간)
@app.callback(
    dash.dependencies.Output('output', 'children'),
    [dash.dependencies.Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('input1', 'value'),
     dash.dependencies.State('input2', 'value'),
     dash.dependencies.State('input3', 'value'),
     dash.dependencies.State('input4', 'value'),
     dash.dependencies.State('input5', 'value')])
def predict(n_clicks, input1, input2, input3, input4, input5):
    # 입력 데이터 정규화
    input_data = np.array([[input1, input2, input3, input4, input5]])
        
    # 모델 예측
    prediction = model.predict(input_data)
    
    # 예측 결과 반환    
    if prediction == 0:
        return html.Div([
            f'충전실 온도: {input1}, 실링 온도: {input2}, 쿠킹 온도: {input3}, 쿠킹 스팀압력: {input4}, 실링 압력: {input5}',
            html.Br(),
            f'예측 결과: 정상 생산'
        ], style={'font-size': '20px', 'text-align': 'center'})
    else:
        return html.Div([
            f'충전실 온도: {input1}, 실링 온도: {input2}, 쿠킹 온도: {input3}, 쿠킹 스팀압력: {input4}, 실링 압력: {input5}',
            html.Br(),
            f'예측 결과: 오류 발생 - 에러코드 {prediction}'
        ], style={'font-size': '20px', 'text-align': 'center'})


# In[12]:



# APP excute
if __name__ == '__main__':
    app.run_server(debug=False)
    

