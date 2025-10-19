package net.tvburger.jdl.model.training.regimes;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;
import net.tvburger.jdl.model.training.Regime;
import net.tvburger.jdl.model.training.TrainableFunction;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * A {@link Regime} decorator that repeats training for a fixed number of epochs.
 *
 * <p>This regime wraps another regime and delegates training to it multiple times.
 * Each call to {@link #train} will invoke the delegate regime exactly
 * {@link #getEpochs()} times over the same dataset. This is useful for
 * composing training behaviors where you want to control how many epochs of
 * training are performed independently of other concerns (batching, reporting,
 * early stopping, etc.).</p>
 *
 * <h3>Notes</h3>
 * <ul>
 *   <li>The number of epochs must be at least 1.</li>
 *   <li>This regime is typically used in a {@link net.tvburger.jdl.model.training.regimes.ChainedRegime}
 *   together with other decorators.</li>
 * </ul>
 *
 * @see DelegatedRegime
 */
@Strategy(Strategy.Role.CONCRETE)
public final class EpochRegime extends DelegatedRegime implements EpochConfigurable {

    @FunctionalInterface
    public interface EpochCompletionListener {

        <N extends Number> void epochCompleted(EpochRegime epochRegime, int currentEpoch, TrainableFunction<N> model, DataSet<N> trainingSet, Optimizer<? extends TrainableFunction<N>, N> optimizer);

    }

    public static EpochCompletionListener sample(int times, EpochCompletionListener listener) {
        return new EpochCompletionListener() {
            @Override
            public <N extends Number> void epochCompleted(EpochRegime epochRegime, int currentEpoch, TrainableFunction<N> model, DataSet<N> trainingSet, Optimizer<? extends TrainableFunction<N>, N> optimizer) {
                boolean call;
                int totalEpochs = epochRegime.getEpochs();
                if (totalEpochs < times || totalEpochs == currentEpoch || currentEpoch == 1) {
                    call = true;
                } else {
                    call = (currentEpoch % (totalEpochs / times)) == 0;
                }
                if (call) {
                    listener.epochCompleted(epochRegime, currentEpoch, model, trainingSet, optimizer);
                }
            }
        };
    }

    private final List<EpochCompletionListener> listeners = new CopyOnWriteArrayList<>();

    /**
     * Registers an epoch completion listener.
     *
     * @param listener the listener to add
     */
    public void addEpochCompletionListener(EpochCompletionListener listener) {
        listeners.add(listener);
    }

    /**
     * Removes an epoch completion listener.
     *
     * @param listener the listener to remove
     */
    public void removeEpochCompletionListener(EpochCompletionListener listener) {
        listeners.remove(listener);
    }

    /**
     * Creates a new {@code EpochRegime} that delegates to the given regime
     * and repeats training for the specified number of epochs.
     *
     * @param regime the underlying regime to delegate training to
     * @param epochs the number of epochs to run (must be {@code >= 1})
     */
    public EpochRegime(Regime regime, int epochs) {
        super(regime);
        setHyperparameter(HP_EPOCHS, epochs);
    }

    /**
     * Trains the given estimation function by repeatedly delegating to the
     * wrapped regime for the configured number of epochs.
     *
     * @param estimationFunction the model to train
     * @param trainingSet        the dataset to train on
     * @param objective          the loss function to evaluate
     * @param optimizer          the optimizer to apply parameter updates
     * @param <E>                the type of estimation function
     */
    @Override
    public <E extends TrainableFunction<N>, N extends Number> void train(E estimationFunction, DataSet<N> trainingSet, ObjectiveFunction<N> objective, Optimizer<? super E, N> optimizer) {
        for (int i = 1; i <= getEpochs(); i++) {
            regime.train(estimationFunction, trainingSet, objective, optimizer, i);
            for (EpochCompletionListener listener : listeners) {
                listener.epochCompleted(this, i, estimationFunction, trainingSet, optimizer);
            }
        }
    }

    @Override
    public <E extends TrainableFunction<N>, N extends Number> void train(E estimationFunction, DataSet<N> trainingSet, ObjectiveFunction<N> objective, Optimizer<? super E, N> optimizer, int step) {
        throw new UnsupportedOperationException();
    }

}