package net.tvburger.jdl.model.training.regimes;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;
import net.tvburger.jdl.model.nn.NeuralNetwork;
import net.tvburger.jdl.model.nn.NeuralNetworks;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;
import net.tvburger.jdl.model.training.Regime;

/**
 * A {@link Regime} decorator that dumps the internal node activations of a
 * {@link NeuralNetwork} before and/or after each training epoch.
 *
 * <p>This regime is typically used for debugging and inspection. It wraps
 * another regime, delegates the actual training to it, and uses
 * {@link NeuralNetworks#dump(NeuralNetwork, boolean)} to output the state of
 * the network. The dump can optionally include input nodes.</p>
 *
 * <h3>Behavior</h3>
 * <ul>
 *   <li>If {@code firstTime} is true, the network is dumped once before the
 *   first training step.</li>
 *   <li>The network is always dumped after each delegated training epoch.</li>
 *   <li>If {@code includeInputs} is true, input node values are included in the
 *   dump; otherwise only hidden and output nodes are dumped.</li>
 * </ul>
 *
 * @see NeuralNetworks#dump(NeuralNetwork, boolean)
 * @see DelegatedRegime
 */
@Strategy(Strategy.Role.CONCRETE)
public class DumpNodesRegime extends DelegatedRegime {

    private boolean firstTime;
    private boolean includeInputs;

    /**
     * Creates a new regime that dumps node values of the given network during training.
     *
     * @param regime        the underlying regime to delegate training to
     * @param firstTime     if true, the network is dumped once before training begins
     * @param includeInputs if true, input nodes are included in the dump
     */
    public DumpNodesRegime(Regime regime, boolean firstTime, boolean includeInputs) {
        super(regime);
        this.firstTime = firstTime;
        this.includeInputs = includeInputs;
    }

    /**
     * Returns whether the network will be dumped once before training starts.
     *
     * @return {@code true} if the network is dumped before the first epoch, otherwise {@code false}
     */
    public boolean isFirstTime() {
        return firstTime;
    }

    /**
     * Sets whether the network should be dumped before training begins.
     *
     * @param firstTime {@code true} to dump before the first epoch, {@code false} otherwise
     */
    public void setFirstTime(boolean firstTime) {
        this.firstTime = firstTime;
    }

    /**
     * Returns whether input nodes are included in the dump.
     *
     * @return {@code true} if input nodes are included, {@code false} otherwise
     */
    public boolean isIncludeInputs() {
        return includeInputs;
    }

    /**
     * Sets whether to include input nodes in the dump.
     *
     * @param includeInputs {@code true} to include input nodes, {@code false} otherwise
     */
    public void setIncludeInputs(boolean includeInputs) {
        this.includeInputs = includeInputs;
    }

    /**
     * Trains the given estimation function with the wrapped regime and dumps the
     * network nodes before and/or after training.
     * <p>
     * If the estimation function is a {@link NeuralNetwork}, its nodes are
     * printed using {@link NeuralNetworks#dump(NeuralNetwork, boolean)}.
     * </p>
     *
     * @param estimationFunction the model to train (only {@link NeuralNetwork} is dumped)
     * @param trainingSet        the dataset to train on
     * @param objective          the objective (loss) function
     * @param optimizer          the optimizer to apply updates
     * @param <E>                the type of estimation function
     */
    @Override
    public <E extends EstimationFunction> void train(E estimationFunction, DataSet trainingSet, ObjectiveFunction objective, Optimizer<? super E> optimizer) {
        if (firstTime) {
            dumpNodes(estimationFunction);
        }
        regime.train(estimationFunction, trainingSet, objective, optimizer);
        dumpNodes(estimationFunction);
    }

    private void dumpNodes(EstimationFunction estimationFunction) {
        if (estimationFunction instanceof NeuralNetwork neuralNetwork) {
            NeuralNetworks.dump(neuralNetwork, includeInputs);
        }
    }
}
