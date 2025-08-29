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
 * and gradients, apply optimization steps, and control the overall
 * training process.
 * </p>
 *
 * <p>
 * This interface exposes getters and setters for each component,
 * allowing dynamic reconfiguration of the training strategy. It also
 * defines a {@link #train(EstimationFunction, DataSet)} method as the
 * entry point for executing the training process.
 * </p>
 *
 * @param <E> the type of {@link EstimationFunction} being trained
 * @see Initializer
 * @see ObjectiveFunction
 * @see Optimizer
 * @see Regime
 */
@DomainObject
@Strategy(role = Strategy.Role.INTERFACE)
public interface Trainer<E extends EstimationFunction> {

    /**
     * Returns the initializer used to prepare the estimation function
     * prior to training.
     *
     * @return the initializer
     */
    Initializer<? super E> getInitializer();

    /**
     * Sets the initializer used to prepare the estimation function
     * prior to training.
     *
     * @param initializer the initializer to set
     */
    void setInitializer(Initializer<? super E> initializer);

    /**
     * Returns the objective function used to evaluate and guide training.
     *
     * @return the objective function
     */
    ObjectiveFunction getObjective();

    /**
     * Sets the objective function used to evaluate and guide training.
     *
     * @param objective the objective function to set
     */
    void setObjective(ObjectiveFunction objective);

    /**
     * Returns the optimizer used to update the parameters of the
     * estimation function.
     *
     * @return the optimizer
     */
    Optimizer<? super E> getOptimizer();

    /**
     * Sets the optimizer used to update the parameters of the
     * estimation function.
     *
     * @param optimizer the optimizer to set
     */
    void setOptimizer(Optimizer<? super E> optimizer);

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
    void train(E estimationFunction, DataSet trainingSet);

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
    static <E extends EstimationFunction> Trainer<E> of(Initializer<? super E> initializer, ObjectiveFunction objective, Optimizer<? super E> optimizer, Regime regime) {
        TrainerImpl<E> trainer = new TrainerImpl<>();
        trainer.setInitializer(initializer);
        trainer.setObjective(objective);
        trainer.setOptimizer(optimizer == null ? (e, s, o) -> {} : optimizer);
        trainer.setRegime(regime);
        return trainer;
    }
}
