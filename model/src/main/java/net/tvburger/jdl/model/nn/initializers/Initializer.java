package net.tvburger.jdl.model.nn.initializers;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.nn.Neuron;

/**
 * The initializer is used to set the initial parameter value of a neuron
 */
@Strategy(role = Strategy.Role.INTERFACE)
public interface Initializer {

    /**
     * Initializes the parameters of the given neuron
     *
     * @param neuron the neuron for which the parameters should be initialized
     */
    void initialize(Neuron neuron);

}
