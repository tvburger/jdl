package net.tvburger.jdl.model.training;

import net.tvburger.jdl.common.patterns.DomainObject;
import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;

/**
 * Defines a configurable trainer for {@link EstimationFunction}s.
 *
 * <p>
 * A {@code Trainer} encapsulates the core components of a training
 * pipeline: an {@link Initializer}, an {@link ObjectiveFunction},
 * an {@link Optimizer}, and a {@link Regime}. By composing these
 * elements, the trainer is able to initialize a model, compute loss
 * and parameterGradients, apply optimization steps, and control the overall
 * training process.
 * </p>
 *
 * <p>
 * This interface exposes getters and setters for each component,
 * allowing dynamic reconfiguration of the training strategy. It also
 * defines a {@link #train(TrainableFunction, DataSet)} method as the
 * entry point for executing the training process.
 * </p>
 *
 * @param <E> the type of {@link TrainableFunction} being trained
 * @see Initializer
 * @see ObjectiveFunction
 * @see Optimizer
 * @see Regime
 */
@DomainObject
@Strategy(Strategy.Role.INTERFACE)
public interface Trainer<E extends TrainableFunction<N>, N extends Number> {

    /**
     * Returns the initializer used to prepare the estimation function
     * prior to training.
     *
     * @return the initializer
     */
    Initializer<? super E, N> getInitializer();

    /**
     * Sets the initializer used to prepare the estimation function
     * prior to training.
     *
     * @param initializer the initializer to set
     */
    void setInitializer(Initializer<? super E, N> initializer);

    /**
     * Returns the objective function used to evaluate and guide training.
     *
     * @return the objective function
     */
    ObjectiveFunction<N> getObjective();

    /**
     * Sets the objective function used to evaluate and guide training.
     *
     * @param objective the objective function to set
     */
    void setObjective(ObjectiveFunction<N> objective);

    /**
     * Returns the optimizer used to update the parameters of the
     * estimation function.
     *
     * @return the optimizer
     */
    Optimizer<? super E, N> getOptimizer();

    /**
     * Sets the optimizer used to update the parameters of the
     * estimation function.
     *
     * @param optimizer the optimizer to set
     */
    void setOptimizer(Optimizer<? super E, N> optimizer);

    /**
     * Returns the regime that controls the overall training flow,
     * such as batching and number of epochs.
     *
     * @return the training regime
     */
    Regime getRegime();

    /**
     * Sets the regime that controls the overall training flow,
     * such as batching and number of epochs.
     *
     * @param regime the training regime to set
     */
    void setRegime(Regime regime);

    /**
     * Trains the given estimation function on the provided training set
     * using the configured initializer, objective function, optimizer,
     * and regime.
     *
     * @param estimationFunction the model or estimation function to train
     * @param trainingSet        the dataset used for training
     */
    void train(E estimationFunction, DataSet<N> trainingSet);

    /**
     * Factory method for creating a new {@code Trainer} with the given
     * components.
     *
     * <p>
     * This method constructs a {@link TrainerImpl}, configures it with
     * the provided initializer, objective, optimizer, and regime, and
     * returns it as a {@link Trainer} instance.
     * </p>
     *
     * @param <E>         the type of estimation function to train
     * @param initializer the initializer to use
     * @param objective   the objective function to use
     * @param optimizer   the optimizer to use
     * @param regime      the training regime to use
     * @return a configured trainer instance
     */
    static <E extends TrainableFunction<N>, N extends Number> Trainer<E, N> of(Initializer<? super E, N> initializer, ObjectiveFunction<N> objective, Optimizer<? super E, N> optimizer, Regime regime) {
        TrainerImpl<E, N> trainer = new TrainerImpl<>();
        trainer.setInitializer(initializer);
        trainer.setObjective(objective);
        trainer.setOptimizer(optimizer == null ? Optimizer.nullOptimizer() : optimizer);
        trainer.setRegime(regime);
        return trainer;
    }
}
