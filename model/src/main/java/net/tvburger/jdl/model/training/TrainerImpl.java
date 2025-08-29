package net.tvburger.jdl.model.training;

import net.tvburger.jdl.common.patterns.Mediator;
import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;

/**
 * Default implementation of the {@link Trainer} interface.
 *
 * <p>
 * {@code TrainerImpl} wires together an {@link Initializer},
 * {@link ObjectiveFunction}, {@link Optimizer}, and {@link Regime}
 * to execute a full training process for an {@link EstimationFunction}.
 * </p>
 *
 * <p>
 * This class is typically not instantiated directly; instead, clients
 * are encouraged to use the factory method
 * {@link Trainer#of(Initializer, ObjectiveFunction, Optimizer, Regime)},
 * which constructs and configures an instance of {@code TrainerImpl}.
 * </p>
 *
 * <p>
 * The {@link #train(EstimationFunction, DataSet)} method follows the
 * canonical training sequence:
 * </p>
 * <ol>
 *   <li>Initialize the estimation function using the configured
 *       {@link Initializer}.</li>
 *   <li>Apply the configured {@link Regime}, which in turn uses the
 *       {@link ObjectiveFunction} and {@link Optimizer} to guide and
 *       update the estimation function during training.</li>
 * </ol>
 *
 * @param <E> the type of {@link EstimationFunction} being trained
 * @see Trainer
 * @see Initializer
 * @see ObjectiveFunction
 * @see Optimizer
 * @see Regime
 */
@Mediator
@Strategy(role = Strategy.Role.CONCRETE)
public class TrainerImpl<E extends EstimationFunction> implements Trainer<E> {

    private Initializer<? super E> initializer;
    private ObjectiveFunction objective;
    private Optimizer<? super E> optimizer;
    private Regime regime;

    /**
     * {@inheritDoc}
     */
    @Override
    public Initializer<? super E> getInitializer() {
        return initializer;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setInitializer(Initializer<? super E> initializer) {
        this.initializer = initializer;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ObjectiveFunction getObjective() {
        return objective;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setObjective(ObjectiveFunction objective) {
        this.objective = objective;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Optimizer<? super E> getOptimizer() {
        return optimizer;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setOptimizer(Optimizer<? super E> optimizer) {
        this.optimizer = optimizer;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Regime getRegime() {
        return regime;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setRegime(Regime regime) {
        this.regime = regime;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void train(E estimationFunction, DataSet trainingSet) {
        if (optimizer == null) {
            throw new IllegalStateException("No optimizer defined!");
        }
        if (regime == null) {
            throw new IllegalStateException("No regime defined!");
        }
        if (initializer != null) {
            initializer.initialize(estimationFunction);
        }
        regime.train(estimationFunction, trainingSet, objective, optimizer);
    }
}
