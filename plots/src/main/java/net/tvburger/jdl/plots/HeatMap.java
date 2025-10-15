package net.tvburger.jdl.plots;

import org.knowm.xchart.HeatMapChart;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;

public class HeatMap {

    private final HeatMapChart chart;
    private SwingWrapper<XYChart> wrapper;

    public HeatMap(String title) {
        this.chart = null;//createChart(title);
    }

    private static XYChart createChart(String title) {
        return new XYChartBuilder()
                .width(600).height(300)
                .title(title)
                .xAxisTitle("x")
                .yAxisTitle("y")
                .build();
    }

    public void drawMap(double[][] map) {
    }
}
