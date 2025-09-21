package net.tvburger.jdl.plots;

import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.XYSeries;
import org.knowm.xchart.style.lines.SeriesLines;
import org.knowm.xchart.style.markers.None;
import org.knowm.xchart.style.markers.SeriesMarkers;

public class Plot implements DataRenderer {

    private final XYChart chart;

    public Plot(String title) {
        this.chart = createChart(title);
    }

    public XYChart createChart(String title) {
        return new XYChartBuilder()
                .width(800).height(500)
                .title(title)
                .xAxisTitle("x")
                .yAxisTitle("y")
                .build();
    }

    public void setSeries(String name, float[] x, float[] y) {
        XYSeries series = chart.addSeries(name, x, y);
        series.setMarker(new None());
    }

    public void setPoints(String name, float[] x, float[] y) {
        XYSeries series = chart.addSeries(name, x, y);
        series.setMarker(SeriesMarkers.CIRCLE);
        series.setLineStyle(SeriesLines.NONE);
    }

    @Override
    public void display() {
        chart.getStyler().setXAxisDecimalPattern("0.##");
        new SwingWrapper<>(chart).displayChart();
    }
}
