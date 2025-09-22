package net.tvburger.jdl.model.training;

import net.tvburger.jdl.common.patterns.DomainObject;
import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.EstimationFunction;

/**
 * Defines an initialization strategy for an {@link EstimationFunction}.
 *
 * <p>
 * An {@code Initializer} is responsible for preparing an estimation
 * function (e.g., a machine learning model) before training begins.
 * This may involve setting parameter values, allocating resources,
 * or performing other setup tasks required to ensure that the model
 * starts from a valid and meaningful state.
 * </p>
 *
 * <p>
 * Different implementations can provide different initialization
 * strategies, such as random initialization, Xavier/Glorot initialization,
 * or restoring from a saved checkpoint.
 * </p>
 *
 * @param <E> the type of estimation function being initialized
 * @see EstimationFunction
 */
@DomainObject
@Strategy(Strategy.Role.INTERFACE)
public interface Initializer<E extends EstimationFunction<N>, N extends Number> {

    /**
     * Initializes the given estimation function according to the
     * initialization strategy defined by the implementation.
     *
     * @param estimationFunction the estimation function (model) to initialize
     */
    void initialize(E estimationFunction);

}
