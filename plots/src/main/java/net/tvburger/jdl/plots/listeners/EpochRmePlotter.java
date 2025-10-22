package net.tvburger.jdl.plots.listeners;

import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.training.Optimizer;
import net.tvburger.jdl.model.training.TrainableFunction;
import net.tvburger.jdl.model.training.regimes.EpochRegime;
import net.tvburger.jdl.plots.DataRenderer;
import net.tvburger.jdl.plots.Plot;

import java.util.concurrent.atomic.AtomicInteger;

public class EpochRmePlotter implements DataRenderer {

    private final AtomicInteger PLOT_COUNTER = new AtomicInteger(0);

    private final Plot rmePlot = new Plot("RME");

    public static EpochRegime.EpochCompletionListener createListener() {
        return createListener(null);
    }

    private String createUniqueModelName() {
        return "Model-" + PLOT_COUNTER.incrementAndGet();
    }

    public static EpochRegime.EpochCompletionListener createListener(String name) {
        EpochRmePlotter plot = new EpochRmePlotter();
        plot.display();
        return plot.attach(name == null ? plot.createUniqueModelName() : name);
    }

    public EpochRegime.EpochCompletionListener attach() {
        return attach(createUniqueModelName(), null);
    }

    public EpochRegime.EpochCompletionListener attach(String name) {
        return attach(name, null);
    }

    public EpochRegime.EpochCompletionListener attach(String name, DataSet<?> validationSet) {
        return new EpochRegime.EpochCompletionListener() {
            @SuppressWarnings("unchecked")
            @Override
            public <N extends Number> void epochCompleted(EpochRegime epochRegime, int currentEpoch, TrainableFunction<N> model, DataSet<N> trainingSet, Optimizer<? extends TrainableFunction<N>, N> optimizer) {
                synchronized (EpochRmePlotter.this) {
                    rmePlot.addToSeries(name, new float[]{currentEpoch}, new float[]{calculateRme(model, validationSet == null ? trainingSet : (DataSet<N>) validationSet)});
                    rmePlot.redraw();
                }
            }
        };
    }

    private <N extends Number> float calculateRme(TrainableFunction<N> model, DataSet<N> samples) {
        float mse = 0.0f;
        for (DataSet.Sample<N> sample : samples) {
            Array<N> estimate = model.estimate(sample.features());
            float sumSquaredErrors = 0.0f;
            for (int i = 0; i < estimate.length(); i++) {
                float error = estimate.get(i).floatValue() - sample.targetOutputs().get(i).floatValue();
                sumSquaredErrors += error * error;
            }
            float meanSquaredError = sumSquaredErrors / estimate.length();
            mse += (float) Math.sqrt(meanSquaredError) / samples.size();
        }
        return mse;
    }

    @Override
    public void display() {
        rmePlot.display();
    }
}
