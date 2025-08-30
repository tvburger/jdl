package net.tvburger.jdl.model.training.regimes;

import net.tvburger.jdl.common.patterns.StaticFactory;
import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.common.utils.Floats;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;
import net.tvburger.jdl.model.training.Regime;

/**
 * A {@link Regime} decorator that implements <em>early stopping</em>
 * based on lack of improvement in the objective (loss).
 *
 * <p>This regime monitors the relative improvement in loss between epochs
 * using an {@link ObjectiveReportingRegime}. If the improvement falls below
 * a configured threshold for more than {@link #getMaxStalledEpochs()} epochs
 * in a row, training is terminated early. This prevents wasting time on
 * epochs that do not lead to meaningful progress.</p>
 *
 * <h3>Behavior</h3>
 * <ul>
 *   <li>Wraps the delegate regime with an {@link ObjectiveReportingRegime}
 *   if it is not already one.</li>
 *   <li>Each call to {@link #train} checks the last recorded relative
 *   improvement.</li>
 *   <li>If the improvement is greater than the configured
 *   {@link #getMinImprovement()}, the stall counter is reset to zero.</li>
 *   <li>If the improvement is less than or equal to the threshold,
 *   the stall counter is incremented.</li>
 *   <li>If the stall counter exceeds {@link #getMaxStalledEpochs()}, the
 *   {@link #train} call returns immediately and {@link #terminated} is set
 *   to {@code true}.</li>
 * </ul>
 *
 * @see ObjectiveReportingRegime
 * @see DelegatedRegime
 */
@Strategy(Strategy.Role.CONCRETE)
public class StopIfNoImprovementRegime extends DelegatedRegime {

    private int maxStalledEpochs;
    private float minImprovement;
    private volatile int currentStalled;
    private volatile int lastEpoch;
    private volatile boolean terminated;

    /**
     * Static factory for creating a {@code StopIfNoImprovementRegime}.
     * <p>If the given regime is not already an {@link ObjectiveReportingRegime},
     * it will be wrapped in one to measure loss improvements. Optionally,
     * loss values can be dumped to the console.</p>
     *
     * @param regime           the underlying regime to delegate to
     * @param maxStalledEpochs maximum consecutive epochs with insufficient improvement before stopping
     * @param minImprovement   minimum relative improvement threshold (percentage) to be considered progress
     * @param dumpLosses       whether to print loss values during training
     * @return a configured {@code StopIfNoImprovementRegime}
     */
    @StaticFactory
    public static StopIfNoImprovementRegime create(Regime regime, int maxStalledEpochs, float minImprovement, boolean dumpLosses) {
        ObjectiveReportingRegime objectiveReportingRegime = regime instanceof ObjectiveReportingRegime
                ? (ObjectiveReportingRegime) regime
                : new ObjectiveReportingRegime(regime, false);
        objectiveReportingRegime.setDumpingLossValues(dumpLosses);
        return new StopIfNoImprovementRegime(objectiveReportingRegime, maxStalledEpochs, minImprovement);
    }

    /**
     * Creates a new stop-if-no-improvement regime.
     *
     * @param regime           the reporting regime to delegate to
     * @param maxStalledEpochs maximum consecutive stalled epochs before stopping
     * @param minImprovement   minimum relative improvement to avoid being counted as stalled
     */
    private StopIfNoImprovementRegime(ObjectiveReportingRegime regime, int maxStalledEpochs, float minImprovement) {
        super(regime);
        this.maxStalledEpochs = maxStalledEpochs;
        this.minImprovement = minImprovement;
    }

    /**
     * @return the maximum number of consecutive stalled epochs tolerated
     */
    public int getMaxStalledEpochs() {
        return maxStalledEpochs;
    }

    /**
     * Sets the maximum number of consecutive stalled epochs tolerated.
     */
    public void setMaxStalledEpochs(int maxStalledEpochs) {
        this.maxStalledEpochs = maxStalledEpochs;
    }

    /**
     * @return the minimum improvement threshold used to detect stalling
     */
    public float getMinImprovement() {
        return minImprovement;
    }

    /**
     * Sets the minimum improvement threshold used to detect stalling.
     */
    public void setMinImprovement(float minImprovement) {
        this.minImprovement = minImprovement;
    }

    /**
     * @return the current count of consecutive stalled epochs
     */
    public int getCurrentStalled() {
        return currentStalled;
    }

    /**
     * Sets the current count of consecutive stalled epochs.
     */
    public void setCurrentStalled(int currentStalled) {
        this.currentStalled = currentStalled;
    }

    /**
     * Trains the model, but stops early if there has been insufficient
     * improvement in the last {@link #getMaxStalledEpochs()} epochs.
     *
     * @param estimationFunction the model to train
     * @param trainingSet        the dataset to train on
     * @param objective          the objective (loss) function
     * @param optimizer          the optimizer to apply updates
     * @param <E>                the type of estimation function
     */
    @Override
    public synchronized <E extends EstimationFunction> void train(E estimationFunction, DataSet trainingSet, ObjectiveFunction objective, Optimizer<? super E> optimizer) {
        Float improvement = ((ObjectiveReportingRegime) regime).getRelativeImprovement();
        if (improvement != null && Floats.greaterThan(improvement, minImprovement)) {
            currentStalled++;
        } else {
            currentStalled = 0;
        }
        if (currentStalled > maxStalledEpochs) {
            terminated = true;
            return;
        }
        lastEpoch++;
        regime.train(estimationFunction, trainingSet, objective, optimizer);
    }

    /**
     * Resets the internal counters and terminated flag so this regime
     * can be reused for a new training run.
     */
    public void reset() {
        terminated = false;
        lastEpoch = 0;
        currentStalled = 0;
    }
}
