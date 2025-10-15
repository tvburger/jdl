package net.tvburger.jdl.plots;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.utils.Pair;
import net.tvburger.jdl.common.utils.Threads;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.scalars.UnaryEstimationFunction;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.XYSeries;
import org.knowm.xchart.style.lines.SeriesLines;
import org.knowm.xchart.style.markers.None;
import org.knowm.xchart.style.markers.SeriesMarkers;

public class Plot implements DataRenderer {

    private final XYChart chart;
    private SwingWrapper<XYChart> wrapper;

    public Plot(String title) {
        this.chart = createChart(title);
    }

    private static XYChart createChart(String title) {
        return new XYChartBuilder()
                .width(600).height(300)
                .title(title)
                .xAxisTitle("x")
                .yAxisTitle("y")
                .build();
    }

    public XYChart getChart() {
        return chart;
    }

    public void setYRange(double min, double max) {
        chart.getStyler().setYAxisMin(min);
        chart.getStyler().setYAxisMax(max);
    }

    public void setSeries(String name, float[] x, float[] y) {
        double[] xd = new double[x.length];
        for (int i = 0; i < xd.length; i++) {
            xd[i] = x[i];
        }
        double[] yd = new double[y.length];
        for (int i = 0; i < yd.length; i++) {
            yd[i] = y[i];
        }
        setSeries(name, xd, yd);
    }

    public void setSeries(String name, double[] x, double[] y) {
        XYSeries xySeries = chart.getSeriesMap().get(name);
        if (xySeries != null) {
            chart.updateXYSeries(name, x, y, null);
        } else {
            XYSeries series = chart.addSeries(name, x, y);
            series.setMarker(new None());
        }
    }

    public void setPoints(String name, float[] x, float[] y) {
        XYSeries series = chart.addSeries(name, x, y);
        series.setMarker(SeriesMarkers.CIRCLE);
        series.setLineStyle(SeriesLines.NONE);
    }

    @Override
    public void display() {
        chart.getStyler().setXAxisDecimalPattern("0.##");
        wrapper = new SwingWrapper<>(chart);
        wrapper.setTitle(chart.getTitle());
        chart.setTitle("");
        wrapper.displayChart();
        Threads.sleepSilently(1_000);
    }

    public <N extends Number> void plotTargetOutput(UnaryEstimationFunction<N> function, String name) {
        plotTargetOutput(function, name, 100);
    }

    public <N extends Number> void plotTargetOutput(UnaryEstimationFunction<N> function, String name, int n) {
        Pair<float[], float[]> pair = computeXY(function, n);
        setSeries(name, pair.left(), pair.right());
    }

    public <N extends Number> Pair<float[], float[]> computeXY(UnaryEstimationFunction<N> function, int n) {
        float min = 0.0f;
        float max = 1.0f;
        JavaNumberTypeSupport<N> typeSupport = function.getCurrentNumberType();
        N range = typeSupport.subtract(typeSupport.valueOf(max), typeSupport.valueOf(min));
        N counter = typeSupport.zero();
        N n_min_1 = typeSupport.valueOf(n - 1);
        float[] x = new float[n];
        float[] y = new float[x.length];
        for (int i = 0; i < n; i++) {
            N input = typeSupport.add(typeSupport.multiply(typeSupport.divide(range, n_min_1), counter), typeSupport.valueOf(min));
            N output = function.estimateUnary(input);
            x[i] = input.floatValue();
            y[i] = output.floatValue();
            counter = typeSupport.add(counter, typeSupport.one());
        }
        return Pair.of(x, y);
    }

    public void plotDataSet(DataSet<?> dataSet, String name) {
        float[] x = new float[dataSet.size()];
        float[] y = new float[x.length];
        for (int i = 0; i < dataSet.size(); i++) {
            x[i] = dataSet.samples().get(i).features()[0].floatValue();
            y[i] = dataSet.samples().get(i).targetOutputs()[0].floatValue();
        }
        setPoints(name, x, y);
    }

    public void redraw() {
        if (wrapper != null) {
            wrapper.repaintChart();
        }
    }

    public void addToSeries(String name, float[] x, float[] y) {
        XYSeries xySeries = chart.getSeriesMap().get(name);
        double[] xData = xySeries == null ? new double[0] : xySeries.getXData();
        double[] yData = xySeries == null ? new double[0] : xySeries.getYData();
        double[] xDataNew = new double[xData.length + x.length];
        double[] yDataNew = new double[yData.length + y.length];
        System.arraycopy(xData, 0, xDataNew, 0, xData.length);
        System.arraycopy(yData, 0, yDataNew, 0, yData.length);
        for (int i = 0; i < x.length; i++) {
            xDataNew[i + xData.length] = x[i];
        }
        for (int i = 0; i < y.length; i++) {
            yDataNew[i + yData.length] = y[i];
        }
        setSeries(name, xDataNew, yDataNew);
    }
}
