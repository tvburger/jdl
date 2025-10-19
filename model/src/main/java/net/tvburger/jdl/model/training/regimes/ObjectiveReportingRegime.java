package net.tvburger.jdl.model.training.regimes;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.patterns.StaticFactory;
import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.common.utils.Pair;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;
import net.tvburger.jdl.model.training.Regime;
import net.tvburger.jdl.model.training.TrainableFunction;

import java.util.List;

/**
 * A {@link Regime} decorator that wraps another training regime and reports the
 * objective (loss) value at each epoch.
 * <p>
 * This class is typically used for debugging or monitoring: it computes the
 * aggregated loss before the first epoch (baseline) and after each epoch, then
 * prints the results if {@code dump} is enabled. It also tracks the relative
 * improvement between epochs.
 * </p>
 * <h3>Improvement Sign Convention</h3>
 * The reported relative improvement is expressed as a percentage change in
 * loss compared to the previous epoch:
 * <ul>
 *   <li>A negative value means the loss decreased (improved).</li>
 *   <li>A positive value means the loss increased (worsened).</li>
 * </ul>
 * <h3>Thread Safety</h3>
 * This class is not thread-safe. Create a new instance per training run or call
 * {@link #reset()} if you want to reuse the same instance across runs.
 *
 * @see DelegatedRegime
 */
@Strategy(Strategy.Role.CONCRETE)
public class ObjectiveReportingRegime extends DelegatedRegime {

    /**
     * The hyperparameter name to dump the objective
     */
    public static final String HP_OBJECTIVE_DUMP = "objectiveDump";

    private Number improvement;
    private Number currentLoss;

    /**
     * Creates a new reporting regime that wraps the given regime.
     *
     * @param regime the underlying training regime to delegate to
     */
    @StaticFactory
    public static <N extends Number> ObjectiveReportingRegime create(Regime regime) {
        return new ObjectiveReportingRegime(regime, false);
    }

    /**
     * Creates a new reporting regime that wraps the given regime.
     *
     * @param regime the underlying training regime to delegate to
     * @param dump   whether to print loss values and improvement at each epoch
     */
    public ObjectiveReportingRegime(Regime regime, boolean dump) {
        super(regime);
        setHyperparameter(HP_OBJECTIVE_DUMP, dump);
    }

    /**
     * Returns whether this regime is currently printing loss values.
     *
     * @return {@code true} if reporting is enabled, {@code false} otherwise
     */
    public boolean isDumpingLossValues() {
        return getHyperparameter(HP_OBJECTIVE_DUMP, Boolean.class);
    }

    /**
     * Enables or disables reporting of loss values.
     *
     * @param dump {@code true} to enable reporting, {@code false} to disable
     */
    public void setDumpingLossValues(boolean dump) {
        setHyperparameter(HP_OBJECTIVE_DUMP, dump);
    }

    /**
     * Trains the given estimation function using the wrapped regime and reports
     * the aggregated loss before and after training.
     * <p>
     * On the first epoch, the baseline aggregated loss is computed before any
     * updates. After each epoch, the new aggregated loss is computed, the
     * relative improvement is calculated, and results are printed if reporting
     * is enabled.
     * </p>
     *
     * @param estimationFunction the model to train
     * @param trainingSet        the dataset to train on
     * @param objective          the loss function to evaluate
     * @param optimizer          the optimizer to update parameters
     * @param <E>                the type of estimation function
     */
    public <E extends TrainableFunction<N>, N extends Number> void train(E estimationFunction, DataSet<N> trainingSet, ObjectiveFunction<N> objective, Optimizer<? super E, N> optimizer) {
        train(estimationFunction, trainingSet, objective, optimizer, (Integer) null);
    }

    public <E extends TrainableFunction<N>, N extends Number> void train(E estimationFunction, DataSet<N> trainingSet, ObjectiveFunction<N> objective, Optimizer<? super E, N> optimizer, int step) {
        train(estimationFunction, trainingSet, objective, optimizer, (Integer) step);
    }

    @SuppressWarnings("unchecked")
    private <E extends TrainableFunction<N>, N extends Number> void train(E estimationFunction, DataSet<N> trainingSet, ObjectiveFunction<N> objective, Optimizer<? super E, N> optimizer, Integer step) {
        JavaNumberTypeSupport<N> typeSupport = estimationFunction.getCurrentNumberType();
        N previousLoss;
        if (objective != null && step == 1) {
            List<Pair<N[], N[]>> batch = trainingSet.samples().stream().map(s -> Pair.of(estimationFunction.estimate(s.features()), s.targetOutputs())).toList();
            previousLoss = objective.calculateLoss(batch, estimationFunction.getParameters());
            if (isDumpingLossValues()) {
                System.out.printf("[Measurement %4d] Aggregated loss = %.4s (baseline)%n", 0, previousLoss);
            }
        } else {
            previousLoss = (N) currentLoss;
        }
        if (step == null) {
            regime.train(estimationFunction, trainingSet, objective, optimizer);
        } else {
            regime.train(estimationFunction, trainingSet, objective, optimizer, step);
        }
        if (objective != null) {
            List<Pair<N[], N[]>> batch = trainingSet.samples().stream().map(s -> Pair.of(estimationFunction.estimate(s.features()), s.targetOutputs())).toList();
            currentLoss = objective.calculateLoss(batch, estimationFunction.getParameters());
            improvement = typeSupport.multiply(typeSupport.divide(typeSupport.subtract(previousLoss, (N) currentLoss), previousLoss), -100);
            if (isDumpingLossValues()) {
                System.out.printf("[Measurement %4d] Aggregated loss = %.4s (%.2s%%)%n", step, currentLoss, improvement);
            }
        }
    }

    /**
     * Returns the relative improvement in aggregated loss compared to the
     * previous epoch.
     * <p>
     * For example, if epoch N yields a loss of 10 and epoch N+1 yields a loss
     * of 5, this method returns -50.0f. If the loss increases, the returned
     * value will be positive.
     * </p>
     *
     * @return the relative improvement percentage, or {@code null} if no
     * training has happened yet
     */
    public Number getRelativeImprovement() {
        return improvement;
    }

    /**
     * Returns the aggregated loss from the most recent epoch.
     *
     * @return the last aggregated loss value, or {@code null} if no training
     * has happened yet
     */
    public Number getCurrentLoss() {
        return currentLoss;
    }

    /**
     * Resets this reporting regime so it can be reused for a new training runs.
     * This clears the iteration count, current loss, and improvement history.
     */
    public void reset() {
        this.currentLoss = null;
        this.improvement = null;
    }
}
