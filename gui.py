import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output
import plotly.express as px
import processing, analysis, data_extraction


# import dash_bootstrap_components as dbc

def extract(name: str = 'r01'):
    data, _ = processing.open_record_abd(record_name=name, qrs=False)
    return data


def preprocess(data, bp_range: tuple = (1, 100), order: int = 1):
    filtered = processing.bandpass_filter(data, bp_range[1], bp_range[0], order=order)
    preprocessed = processing.bwr_signals(filtered)
    return preprocessed


def extract_fecg(data, mica: int = 0):
    """
    Fetal extraction algorithm
    :param data: data
    :param mica: choose mothers channel
    :param fica: choose fetal channel
    :return: fecg
    """
    # FastICA - TS
    ica1 = analysis.fast_ica(data, 4, processing.tanh)
    r_peaks = analysis.find_qrs(ica1[mica, :], peak_search='original')
    r_peaks = analysis.peak_enhance(ica1[mica, :], peaks=r_peaks, window=0.3)
    subtracted = analysis.ts_method(ica1, peaks=r_peaks, template_duration=0.3, fs=processing.FS, window=10)
    return ica1, subtracted


def rr_analysis(fetal_ecg, median_kernel: tuple = (6,), mode: str = 'bpm'):
    """
    Analysis fetal VHR

    :param mode: choose mode for "processing.calculate_rr
    :param median_kernel: choose kernel for analysis.median_filtration
    :param fetal_ecg: Fetal ECG
    :return: rr_intervals, sample frequency
    """
    peaks = analysis.find_qrs(fetal_ecg, processing.FS, peak_search='Original')
    enhanced_peaks = analysis.peak_enhance(fetal_ecg, peaks, window=0.08)
    rr_intervals, fs_rr = analysis.calculate_rr(enhanced_peaks, mode=mode, time=True)
    med_rr = analysis.median_filtration(rr_intervals, kernel=median_kernel)
    return med_rr, fs_rr


fs = processing.FS
data = extract()[1:, 0:300000]
data = preprocess(data)
size = data.shape
t = np.arange(0, (size[1] * 1 / fs), 1 / fs)
ica1, subtracted = extract_fecg(data)
ic1 = ica1
median, fs_rr = rr_analysis(subtracted[0, :], median_kernel=(6,), mode='bpm')
fs_r = fs_rr
media = median
tim = np.arange(0, (len(median) * 1 / fs_rr), 1 / fs_rr)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# external_stylesheets=[dbc.themes.COSMO]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.DataFrame(np.arange(1000).reshape(2, 500))

app.layout = html.Div([
    html.Div([
        html.Center([
            html.H6('Make several callbacks', style={'font-style': 'italic', 'font-weight': 'bold'})
        ]),
    ]),
    html.Div([
        html.Div([
            dcc.Graph(id='g1', config={'displayModeBar': False})
        ], className='five columns'),
        html.Div([
            dcc.Graph(id='g2', config={'displayModeBar': False})
        ], className='five columns'),
        html.Div([
            html.Button('restart', id='btn', n_clicks=0),
            html.H6('Mother\'s IC'),
            dcc.RadioItems(
                id='radio',
                options=[
                    {'label': 'mica 0', 'value': 0},
                    {'label': 'mica 1', 'value': 1},
                    {'label': 'mica 2', 'value': 2},
                    {'label': 'mica 3', 'value': 3}],
                value=0)],
            className='two columns')
    ], className="row", style={'height': '200'}),
    html.Div([
        html.Div([
            dcc.Graph(id='g3', config={'displayModeBar': False})
        ], className='five columns'),
        html.Div([
            dcc.Graph(id='g4', config={'displayModeBar': False})
        ], className='five columns'),
        html.Div([
            html.H6('Fetal IC'),
            dcc.RadioItems(
                id='radio2',
                options=[
                    {'label': 'fica 0', 'value': 0},
                    {'label': 'fica 1', 'value': 1},
                    {'label': 'fica 2', 'value': 2},
                    {'label': 'fica 3', 'value': 3}],
                value=0)],
            className='two columns')
    ], className="row", style={'height': '200'}),
    html.Div([
        dcc.Graph(id='median', config={"displayModeBar": False}),
        html.H4("Slider_name"),
        dcc.RangeSlider(id='slider',
                        min=0,
                        max=size[1] * 1 / fs,
                        step=1,
                        allowCross=False,
                        value=[0, size[1] * 1 / fs]
                        )

    ], style={'height': '40vh'})
], style={'height': '90vh', 'weight': '100vh'})

"""
Rebuild for several callbacks, this one is too laggy (with 300k) 

@app.callback(
    Output('g1', 'figure'),
    Output('g2', 'figure'),
    Output('g3', 'figure'),
    Output('g4', 'figure'),
    Output('median', 'figure'),
    Input('slider', 'value'),
    Input('btn', 'n_clicks'),
    Input('radio', 'value'),
    Input('radio2', 'value')
)
def update_graph(rang_values, btn, mica, fica):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    fig1 = px.line(x=t, y=ic1[0, :])
    fig2 = px.line(x=t, y=ic1[1, :])
    fig3 = px.line(x=t, y=ic1[2, :])
    fig4 = px.line(x=t, y=ic1[3, :])
    fig5 = px.line(x=tim, y=media)
    if 'btn' in changed_id:
        fig1 = px.line(x=t, y=data[0, :])
        fig2 = px.line(x=t, y=data[1, :])
        fig3 = px.line(x=t, y=data[2, :])
        fig4 = px.line(x=t, y=data[3, :])
        fig5 = px.line(x=t, y=data[1, :])
    elif 'radio' in changed_id:
        ica1, subtracted = extract_fecg(data, mica=mica)
        fig1 = px.line(x=t, y=ica1[0, :])
        fig2 = px.line(x=t, y=ica1[1, :])
        fig3 = px.line(x=t, y=ica1[2, :])
        fig4 = px.line(x=t, y=ica1[3, :])
        fecg = subtracted[fica, :]
        median, fs_rr = rr_analysis(fecg, median_kernel=(6,), mode='bpm')
        time = np.arange(0, (len(median) * 1 / fs_rr), 1 / fs_rr)
        fig5 = px.line(x=time, y=median)
    elif 'radio2' in changed_id:
        ica1, subtracted = extract_fecg(data, mica=mica)
        fecg = subtracted[fica, :]
        median, fs_rr = rr_analysis(fecg, median_kernel=(6,), mode='bpm')
        time = np.arange(0, (len(median) * 1 / fs_rr), 1 / fs_rr)
        fig5 = px.line(x=time, y=median)

    fig1.update_xaxes(range=[rang_values[0], rang_values[1]])
    fig2.update_xaxes(range=[rang_values[0], rang_values[1]])
    fig3.update_xaxes(range=[rang_values[0], rang_values[1]])
    fig4.update_xaxes(range=[rang_values[0], rang_values[1]])
    fig5.update_xaxes(range=[rang_values[0], rang_values[1]])

    fig1.update_layout(height=250, margin=dict(t=5, b=5, r=5, l=5))
    fig2.update_layout(height=250, margin=dict(t=5, b=5, r=5, l=5))
    fig3.update_layout(height=250, margin=dict(t=5, b=5, r=5, l=5))
    fig4.update_layout(height=250, margin=dict(t=5, b=5, r=5, l=5))
    fig5.update_layout(height=250, margin=dict(t=5, b=5, r=5, l=5))
    return fig1, fig2, fig3, fig4, fig5
"""

if __name__ == '__main__':
    app.run_server(debug=True)
