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
    return


def preprocess(data, bp_range, order):
    filtered = processing.bandpass_filter(data, bp_range[1], bp_range[0], order=order)
    preprocessed = processing.bwr_signals(filtered)
    return preprocessed


def extract_fecg(data, mica: int = 0, fica: int = 0):
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
    processing.bwr_signals(ica1)
    fecg = analysis.ts_method(ica1[fica, :], peaks=r_peaks, template_duration=0.6, fs=processing.FS, window=10)

    return fecg


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


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# external_stylesheets=[dbc.themes.COSMO]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.DataFrame(np.arange(1000).reshape(2, 500))

app.layout = html.Div([
    html.Div([
        html.Center([
            html.H6('Algorithm Debug GUI')
        ]),
    ]),
    html.Div([
        html.Div([
            dcc.Graph(id='g1', config={'displayModeBar': False})
        ], className='five columns'),
        html.Div([
            dcc.Graph(id='g2', config={'displayModeBar': False})
        ], className='five columns')
    ], className="row", style={'height': '200'}),
    html.Div([
        html.Div([
            dcc.Graph(id='g3', config={'displayModeBar': False})
        ], className='five columns'),
        html.Div([
            dcc.Graph(id='g4', config={'displayModeBar': False})
        ], className='five columns')
    ], className="row", style={'height': '200'}),
    html.Div([
        dcc.Graph(id='median', config={"displayModeBar": False}),
        html.H4("Slider_name"),
        dcc.RangeSlider(id='slider',
                        min=df[0].min(),
                        max=df[0].max(),
                        step=1,
                        allowCross=False,
                        value=[df[0].min(), df[0].max()],
                        )

    ], style={'height': '40vh'})
], style={'height': '90vh', 'weight': '100vh'})


@app.callback(
    Output('median', 'figure'),
    Output('g4', 'figure'),
    Output('g3', 'figure'),
    Output('g2', 'figure'),
    Output('g1', 'figure'),
    Input('slider', 'value'))
def slider_fig(rang_values):
    fig1 = px.scatter(df.loc[0, rang_values[0]:rang_values[1]], height=300)
    fig2 = px.scatter(df.loc[1, rang_values[0]:rang_values[1]], height=300)
    fig3 = px.scatter(df.loc[0, rang_values[0]:rang_values[1]], height=300)
    fig4 = px.scatter(df.loc[1, rang_values[0]:rang_values[1]], height=300)
    fig5 = px.scatter(df.loc[1, rang_values[0]:rang_values[1]], height=300)
    fig5.update_xaxes(rangeselector_visible=True)
    fig1.update_xaxes(rangeselector_visible=True, rangeslider_visible=True)
    return fig1, fig2, fig3, fig4, fig5


if __name__ == '__main__':
    app.run_server(debug=True)
