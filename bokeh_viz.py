from bokeh.io import curdoc
from bokeh.layouts import column, row

from bokeh.models import ColumnDataSource, Select, DatetimeTickFormatter, Span
from bokeh.plotting import figure
from bokeh.embed import components

import pandas as pd

TOOLS = 'wheel_zoom,pan,reset'
CHART_DATA_DETAIL = {'HR': 'Heart Rate', 'SBP': 'Systolic Blood Pressure',\
    'DBP': 'Diastolic Blood Pressure', 'MBP': 'Mean Blood Pressure', 'RR': 'Respiratory Rate', \
    'Temp': 'Temperature', 'SpO2': 'Peripheral Oxygen Saturation', 'VTobs': 'Observed Tidal Volume', \
    'VTspot': 'Spontaneous Tidal Volume', 'PIP': 'Peak Inspiratory Pressure', \
    'PEEP': 'Positive end-expiratory pressure', 'GCStot': 'Glasgow Coma Scale', \
    'GCSeye': 'Glasgow Coma Scale - Eye opening', 'GCSverbal': 'Glasgow Coma Scale - Verbal Response', \
    'GCSmotor': 'Glasgow Coma Scale - Motor Response'}

def create_bokeh_viz(chart_data, sel_vs):
    # Extract selected vital sign
    source, unit = load_vital_sign(chart_data, sel_vs)
    plot = make_plot(source, sel_vs, unit)

    return components(plot)


def load_vital_sign(chart_data, vs):
    # Get data of selected vital sign, 'vs'.
    df_vs = chart_data[chart_data['vital_sign'] == vs]
    unit = df_vs[df_vs['vital_sign'] == vs]['unit'].iloc[0]

    df_vs = df_vs.set_index('charttime')
    df_vs.sort_index(inplace=True)
    df_vs = df_vs.groupby(['charttime']).mean()

    return ColumnDataSource(data=pd.DataFrame({'vital_sign': df_vs['valuenum']})), unit


def make_plot(source, vs, unit):
    plot = figure(x_axis_type='datetime', plot_width=800, tools=TOOLS)
    plot.toolbar.logo = None

    plot.background_fill_color = '#30404C'
    plot.border_fill_color = '#30404C'

    plot.title.text = CHART_DATA_DETAIL[vs]
    plot.title.text_font_size = '2em'
    plot.title.text_font_style = 'bold'
    plot.title.text_color = '#B7C5D3'

    plot.xaxis.axis_label = 'Date and Time'
    plot.xaxis.axis_label_text_font_size = '1.5em'
    plot.xaxis.axis_label_text_color = '#B7C5D3'
    plot.xaxis.major_label_text_font_size = '1em'
    plot.xaxis.major_label_text_color = '#B7C5D3'
    plot.xgrid.grid_line_alpha = 0.1
    plot.xgrid.grid_line_color = 'black'
    plot.xgrid.grid_line_dash = [6, 4]
    plot.xaxis.formatter = DatetimeTickFormatter(minutes=["%H:%M"],
                                                 hours=["%H:%M"],
                                                 days=["%Y.%m.%d"])

    plot.yaxis.axis_label = '{0:} ({1:})'.format(CHART_DATA_DETAIL[vs], unit)
    plot.yaxis.axis_label_text_font_size = '1.5em'
    plot.yaxis.axis_label_text_color = '#B7C5D3'
    plot.yaxis.major_label_text_font_size = '1em'
    plot.yaxis.major_label_text_color = '#B7C5D3'
    plot.ygrid.grid_line_alpha = 0.1
    plot.ygrid.grid_line_color = 'black'
    plot.ygrid.grid_line_dash = [6, 4]

    plot.circle('charttime', 'vital_sign', source=source, size=8, color='#68A1F0')
    plot.line('charttime', 'vital_sign', source=source, line_dash='dotted', color='#B7C5D3', line_alpha=0.8)

    vline = Span(location=source.data['charttime'][-1], dimension='height', \
        line_color='red', line_width=1.5, line_dash='dashed')
    plot.add_layout(vline)

    vline = Span(location=source.data['charttime'][0], dimension='height', \
        line_color='green', line_width=1.5, line_dash='dashed')
    plot.add_layout(vline)

    return plot
