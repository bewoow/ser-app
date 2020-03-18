from bokeh.io import curdoc
from bokeh.layouts import column, row

from bokeh.models import ColumnDataSource, Select, DatetimeTickFormatter, Span, BoxZoomTool, HoverTool
from bokeh.plotting import figure
from bokeh.embed import components

import pandas as pd
import numpy as np

TOOLS = 'pan,reset'
TOOLTIPS = [
    ("Time", "@time{%F %H:%M}"),
    ("SER", "@ser_risk{0.00}")]

BACKGROUND_COLOR = '#30404C'
MAIN_PLOT_COLOR = '#B7C5D3'


def create_bokeh_viz_ser_risk(df_ser_pred):
    # Extract selected vital sign
    source = ColumnDataSource(data=pd.DataFrame({'ser_risk': df_ser_pred['ser_risk']}))
    plot = make_plot_ser(source, 'Self-Extubation Risk')

    return components(plot)


def make_plot_ser(source, title):
    plot = figure(x_axis_type='datetime', plot_width=800, plot_height=400,
                  tools=TOOLS, active_drag='pan')
    plot.toolbar.logo = None
    plot.add_tools(HoverTool(tooltips=TOOLTIPS,
                             formatters={'time': 'datetime'}))

    plot.background_fill_color = BACKGROUND_COLOR
    plot.border_fill_color = BACKGROUND_COLOR

    plot.title.text = title
    plot.title.text_font_size = '2em'
    plot.title.text_font_style = 'bold'
    plot.title.text_color = MAIN_PLOT_COLOR

    plot.xaxis.axis_label = 'Time'
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

    plot.yaxis.axis_label = 'SER'
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

    # Set MV end lines
    vline = Span(location=source.data['time'][-1], dimension='height',
                 line_color='red', line_width=1.5, line_dash='dashed', line_alpha=0.7)
    plot.add_layout(vline)

    # Plot chart data
    plot.line('time', 'ser_risk', source=source,
              line_width=2, color='#68A1F0', line_alpha=1)

    return plot
