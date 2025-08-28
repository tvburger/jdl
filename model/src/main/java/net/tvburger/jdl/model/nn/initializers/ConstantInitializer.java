package net.tvburger.jdl.model.nn.initializers;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.model.nn.Neuron;

import java.util.Arrays;

/**
 * Implements an initializer that sets all parameters (weights and bias) to a given constant.
 */
@Strategy(role = Strategy.Role.CONCRETE)
public class ConstantInitializer implements Initializer {

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
        float[] weights = neuron.getWeights();
        Arrays.fill(weights, constant);
        neuron.setBias(constant);
    }
}
