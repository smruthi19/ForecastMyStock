import matplotlib.pyplot as plt
import mpld3
from datetime import datetime


css = """
text.mpld3-text, div.mpld3-tooltip {
  font-family: Arial, Helvetica, sans-serif;
  font-weight: bold;
  color: black;
  opacity: 1.0;
  padding: 2px;
  border: 0px;
}
"""

fig, ax = plt.subplots()
dates = [datetime(2015, 9, 10), datetime(2015, 9, 11), datetime(2015, 9, 12), datetime(2015, 9, 13)]
values = [2, 4, 6, 8]

line = ax.plot(dates, values, linewidth=3, color=colorstr, marker="o", markerfacecolor="none", markeredgecolor="b")

labels = [(date.strftime("%Y-%m-%d"), " {0}".format(val)) for date, val in zip(dates, values)]
tooltip = mpld3.plugins.PointHTMLTooltip(line[0], labels,
                                   voffset=-20, hoffset=10, css=css)
mpld3.plugins.connect(fig, tooltip)

mpld3.save_html(fig, "./mpld3_htmltooltip.html")
