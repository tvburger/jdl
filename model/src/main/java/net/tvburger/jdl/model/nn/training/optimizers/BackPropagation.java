package net.tvburger.jdl.model.nn.training.optimizers;

import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.linalg.TypedVector;
import net.tvburger.jdl.linalg.Vector;
import net.tvburger.jdl.model.nn.ActivationsCachedNeuron;
import net.tvburger.jdl.model.nn.NeuralNetwork;
import net.tvburger.jdl.model.nn.Neuron;
import net.tvburger.jdl.model.scalars.LinearCombination;
import net.tvburger.jdl.model.scalars.activations.ActivationFunction;
import net.tvburger.jdl.model.training.optimizer.GradientDescentModelDecomposer;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

@Strategy(Strategy.Role.CONCRETE)
public class BackPropagation implements GradientDescentModelDecomposer<NeuralNetwork, Float> {

    @Override
    public Stream<GradientDecomposition<Float>> calculateDecompositionGradients(NeuralNetwork neuralNetwork, Vector<Float> objectiveGradients, Array<Float> inputs) {
        List<GradientDecomposition<Float>> decompositions = new ArrayList<>();
        Map<Neuron, Float> errorSignals = new IdentityHashMap<>();
        for (int k = 0; k < neuralNetwork.coArity(); k++) {
            decomposeAndSetErrorSignalForOutputNode(decompositions, neuralNetwork, objectiveGradients, errorSignals, k);
        }
        for (int l = neuralNetwork.getDepth() - 1; l >= 1; l--) {
            int width = neuralNetwork.getWidth(l);
            for (int j = 0; j < width; j++) {
                decomposeAndSetErrorSignalForHiddenNode(decompositions, neuralNetwork, errorSignals, l, j);
            }
        }
        return decompositions.stream();
    }

    private static void decomposeAndSetErrorSignalForOutputNode(List<GradientDecomposition<Float>> decompositions, NeuralNetwork neuralNetwork, Vector<Float> objectiveGradients, Map<Neuron, Float> errorSignals, int j) {
        ActivationsCachedNeuron outputNode = neuralNetwork.getNeuron(neuralNetwork.getDepth(), j, ActivationsCachedNeuron.class);
        ActivationsCachedNeuron.Activation activation = outputNode.getCache().removeLast();

        // determine error signal for output node
        ActivationFunction activationFunction = outputNode.getNeuronFunction().getActivationFunction();
        float errorSignal = objectiveGradients.get(j + 1) * activationFunction.determineGradientForOutput(activation.output());
        decompositions.add(decompose(outputNode.getNeuronFunction().getLinearCombination(), activation.inputs(), errorSignal));
        errorSignals.put(outputNode, errorSignal);
    }

    private static void decomposeAndSetErrorSignalForHiddenNode(List<GradientDecomposition<Float>> decompositions, NeuralNetwork neuralNetwork, Map<Neuron, Float> errorSignals, int l, int j) {
        ActivationsCachedNeuron hiddenNode = neuralNetwork.getNeuron(l, j, ActivationsCachedNeuron.class);
        ActivationsCachedNeuron.Activation activation = hiddenNode.getCache().removeLast();

        // determine error signal for hidden node using back propagation
        float backPropagation = calculateBackPropagation(neuralNetwork, errorSignals, l, j);
        ActivationFunction activationFunction = hiddenNode.getNeuronFunction().getActivationFunction();
        float errorSignal = backPropagation * activationFunction.determineGradientForOutput(activation.output());
        decompositions.add(decompose(hiddenNode.getNeuronFunction().getLinearCombination(), activation.inputs(), errorSignal));
        errorSignals.put(hiddenNode, errorSignal);
    }

    private static <N extends NeuralNetwork> float calculateBackPropagation(N neuralNetwork, Map<Neuron, Float> errorSignals, int l, int j) {
        float backPropagation = 0.0f;
        Map<Neuron, Float> downstreamNodes = neuralNetwork.getOutputConnections(l, j);
        for (Map.Entry<Neuron, Float> entry : downstreamNodes.entrySet()) {
            float downstreamWeight = entry.getValue();
            float downstreamErrorSignal = errorSignals.getOrDefault(entry.getKey(), 0.0f);
            backPropagation += downstreamErrorSignal * downstreamWeight;
        }
        return backPropagation;
    }

    private static GradientDecomposition<Float> decompose(LinearCombination<Float> linearCombination, Array<Float> inputs, float errorSignal) {
        Array<Float> parameterGradients = JavaNumberTypeSupport.FLOAT.createArray(inputs.length() + 1);
        parameterGradients.set(0, errorSignal); // bias term
        for (int d = 1; d < parameterGradients.length(); d++) {
            parameterGradients.set(d, errorSignal * inputs.get(d - 1));
        }
        return new GradientDecomposition<>(linearCombination, new TypedVector<>(parameterGradients, true, linearCombination.getNumberTypeSupport()));
    }
}
