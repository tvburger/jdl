package net.tvburger.jdl.perceptron;

import net.tvburger.jdl.nn.Neuron;
import net.tvburger.jdl.nn.activations.Activations;

import java.util.List;

public class AssociationUnit extends Neuron {

    public AssociationUnit(String name, List<Neuron> inputs) {
        super(name, inputs, Activations.linear());
    }

}
