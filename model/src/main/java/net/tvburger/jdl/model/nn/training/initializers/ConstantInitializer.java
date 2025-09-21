package net.tvburger.jdl.model.nn.training.initializers;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.nn.Neuron;

/**
 * Implements an initializer that sets all parameters (weights and bias) to a given constant.
 */
@Strategy(Strategy.Role.CONCRETE)
public class ConstantInitializer implements NeuralNetworkInitializer {

    private final float constant;

    /**
     * Constructs a new initializer that sets all parameters to the given constant
     *
     * @param constant the value to initialize the parameters with
     */
    public ConstantInitializer(float constant) {
        this.constant = constant;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void initialize(Neuron neuron) {
        for (int p = 0; p < neuron.getParameterCount(); p++) {
            neuron.setParameter(p, constant);
        }
    }
}
