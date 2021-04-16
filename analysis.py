import numpy as np
from processing import FS
import plotly.graph_objects as go


def filter_templates(fs: int, qrs_template: int = 1):
    """Creates fetal QRS and PQRST templates from [22]
    qrs: (1) h' = -0.007^2 * t * e^(-t^2/0.007^2)
         (2) h = e^(-t^2/0.007^2)
         (3) h3 = −0.2 * e^(-(t + 0.007)^2 / 0.005^2)
                  +e^(-t^2 / 0.005^2)
                  -0.2 * e^(-(t - 0.007)^2 / 0.005^2)"""

    N = int(0.04 * fs + 1)
    qrs = np.zeros((N, 2))
    qrs[:, 1] = np.arange(-0.02, 0.02 + 1 / fs, 1 / fs)  # Time

    ### Templates
    if qrs_template == 1:
        qrs[:, 0] = -2 * np.exp(-np.power(qrs[:, 1], 2) / np.power(0.007, 2)) * qrs[:, 1] / (np.power(0.007, 2))
    elif qrs_template == 2:
        qrs[:, 0] = np.exp(-np.power(qrs[:, 1], 2) / np.power(0.007, 2))
    else:
        qrs[:, 0] = - 0.2 * np.exp(-np.power(qrs[:, 1] + 0.007, 2) / np.power(0.005, 2)) \
                    + np.exp(-np.power(qrs[:, 1], 2) / np.power(0.005, 2)) \
                    - 0.2 * np.exp(-np.power(qrs[:, 1] - 0.007, 2) / np.power(0.005, 2))
    ###

    qrs[:, 0] = (qrs[:, 0] - qrs[:, 0].mean()) / (qrs[:, 0].std()) # Normalization
    qrs[:, 1] = qrs[::-1, 1]  # Reversing => fir_coeffs(b)
    #########
    pqrst = None
    return qrs, pqrst


if __name__ == '__main__':
    templ, _ = filter_templates(FS, 3)

    # Plot template
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=templ[:, 1], y=templ[:, 0]))
    fig.show()
