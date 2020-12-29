from bokeh.io import show ,output_file
from bokeh.plotting import figure

x=[1,2,3,4,5,6]
y=[2,4,6,8,10,12]
plot=figure(x_axis_label="x axis",y_axis_label="y axis")
plot.line(x,y,color="red",alpha=0.4)
plot.circle(x,y,color="green")
output_file("a.html")
show(plot);
