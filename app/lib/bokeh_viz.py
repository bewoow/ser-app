from bokeh.io import curdoc
from bokeh.layouts import column, row

from bokeh.models import ColumnDataSource, Select, DatetimeTickFormatter, Span, BoxZoomTool, HoverTool
from bokeh.plotting import figure
from bokeh.embed import components

import pandas as pd
import numpy as np

TOOLS = 'box_zoom,wheel_zoom,pan,reset'
TOOLTIPS = [
    ("Chart time", "@charttime{%F %H:%M}"),
    ("Value", "@vital_sign{0.00}")]

CHART_DATA_DETAIL = {'HR': 'Heart Rate', 'SBP': 'Systolic Blood Pressure',\
    'DBP': 'Diastolic Blood Pressure', 'MBP': 'Mean Blood Pressure', 'RR': 'Respiratory Rate', \
    'Temp': 'Temperature', 'SpO2': 'Peripheral Oxygen Saturation', 'VTobs': 'Observed Tidal Volume', \
    'VTspot': 'Spontaneous Tidal Volume', 'PIP': 'Peak Inspiratory Pressure', \
    'PEEP': 'Positive end-expiratory pressure', 'GCStot': 'Glasgow Coma Scale', \
    'GCSeye': 'Glasgow Coma Scale - Eye opening', 'GCSverbal': 'Glasgow Coma Scale - Verbal Response', \
    'GCSmotor': 'Glasgow Coma Scale - Motor Response', 'PaCO2': 'Partial Pressure of Carbon Dioxide', \
    'PaO2': 'Partial Pressure of Oxygen', 'FiO2': 'Fraction of Inspired Oxygen', \
    'CVP': 'Central Venous Pressure', 'PAP': 'Mean Pulmonary Arterial Pressure', \
    'PAPSys': 'Systolic Pulmonary Arterial Pressure'}

BACKGROUND_COLOR = '#30404C'
MAIN_PLOT_COLOR = '#B7C5D3'

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
    plot = figure(x_axis_type='datetime', plot_width=800, tools=TOOLS, active_drag='box_zoom', active_scroll='wheel_zoom')
    plot.toolbar.logo = None
    plot.add_tools(HoverTool(tooltips=TOOLTIPS,
                            formatters={'charttime': 'datetime'}))

    plot.background_fill_color = BACKGROUND_COLOR
    plot.border_fill_color = BACKGROUND_COLOR

    plot.title.text = CHART_DATA_DETAIL[vs]
    plot.title.text_font_size = '2em'
    plot.title.text_font_style = 'bold'
    plot.title.text_color = MAIN_PLOT_COLOR

    plot.xaxis.axis_label = 'Chart time'
    plot.xaxis.axis_label_text_font_size = '1.5em'
    plot.xaxis.axis_label_text_color = MAIN_PLOT_COLOR
    plot.xaxis.axis_line_color = MAIN_PLOT_COLOR
    plot.xaxis.major_label_text_font_size = '1.2em'
    plot.xaxis.major_label_text_color = MAIN_PLOT_COLOR
    plot.xaxis.major_tick_line_color = MAIN_PLOT_COLOR
    plot.xgrid.grid_line_alpha = 0.1
    plot.xgrid.grid_line_color = MAIN_PLOT_COLOR
    plot.xgrid.grid_line_dash = 'dotted'
    plot.xaxis.formatter = DatetimeTickFormatter(minutes=["%H:%M"],
                                                 hours=["%H:%M"],
                                                 days=["%F"])
    plot.xaxis.major_label_orientation = np.pi / 4

    if unit:
        plot.yaxis.axis_label = '{0:} ({1:})'.format(vs, unit)
    else:
        plot.yaxis.axis_label = vs
    plot.yaxis.axis_label_text_font_size = '1.5em'
    plot.yaxis.axis_label_text_color = MAIN_PLOT_COLOR
    plot.yaxis.axis_line_color = MAIN_PLOT_COLOR
    plot.yaxis.major_label_text_font_size = '1.2em'
    plot.yaxis.major_label_text_color = MAIN_PLOT_COLOR
    plot.yaxis.major_tick_line_color = MAIN_PLOT_COLOR
    plot.yaxis.minor_tick_line_color = None
    plot.ygrid.grid_line_alpha = 0.1
    plot.ygrid.grid_line_color = MAIN_PLOT_COLOR
    plot.ygrid.grid_line_dash = 'dotted'

    # Set MV start and end lines
    vline = Span(location=source.data['charttime'][-1], dimension='height', \
        line_color='red', line_width=1.5, line_dash='dashed', line_alpha=0.7)
    plot.add_layout(vline)

    vline = Span(location=source.data['charttime'][0], dimension='height', \
        line_color='green', line_width=1.5, line_dash='dashed', line_alpha=0.7)
    plot.add_layout(vline)

    # Plot chart data
    plot.line('charttime', 'vital_sign', source=source, line_dash='dashdot',
        line_width=1.5, color='#B7C5D3', line_alpha=0.9)
    plot.circle('charttime', 'vital_sign', source=source, size=9, color='#68A1F0')

    # Add box zoom tool
    zoom_overlay = plot.select_one(BoxZoomTool).overlay
    zoom_overlay.line_color = MAIN_PLOT_COLOR
    zoom_overlay.line_width = 4
    zoom_overlay.line_dash = "solid"
    zoom_overlay.fill_color = None

    return plot
