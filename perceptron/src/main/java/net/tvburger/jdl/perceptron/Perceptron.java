package net.tvburger.jdl.perceptron;

import net.tvburger.jdl.nn.ActivationFunction;
import net.tvburger.jdl.nn.DefaultNeuralNetwork;
import net.tvburger.jdl.nn.InputNeuron;
import net.tvburger.jdl.nn.Neuron;
import net.tvburger.jdl.nn.activations.Activations;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Perceptron extends DefaultNeuralNetwork {

    private static final Random random = new Random();

    public static Perceptron create() {
        return create(400, 512, 8);
    }

    public static Perceptron create(int sensors, int responses) {
        return create(sensors, 0, responses);
    }

    public static Perceptron create(int sensors, int associations, int responses) {
        return create(sensors, associations, responses, Activations.step());
    }

    public static Perceptron create(int sensors, int associations, int responses, ActivationFunction activationFunction) {
        List<List<Neuron>> layers = new ArrayList<>();
        List<Neuron> sensorNodes = new ArrayList<>();
        for (int i = 0; i < sensors; i++) {
            sensorNodes.add(new InputNeuron("Sensor(" + i + ")"));
        }
        layers.add(sensorNodes);

        List<Neuron> associationNodes = new ArrayList<>();
        for (int i = 0; i < associations; i++) {
            associationNodes.add(new Neuron("Association(" + i + ")", randomSelect(sensorNodes), activationFunction));
        }
        layers.add(associationNodes);

        List<Neuron> responseNodes = new ArrayList<>();
        for (int i = 0; i < responses; i++) {
            responseNodes.add(new Neuron("Response(" + i + ")", associationNodes.isEmpty() ? sensorNodes : associationNodes, activationFunction));
        }
        layers.add(responseNodes);
        return new Perceptron(layers);
    }

    private static List<Neuron> randomSelect(List<Neuron> sensorNodes) {
        List<Neuron> selected = new ArrayList<>();
        for (Neuron sensorNode : sensorNodes) {
            if (random.nextBoolean()) {
                selected.add(sensorNode);
            }
        }
        return selected;
    }

    private Perceptron(List<List<Neuron>> layers) {
        super(layers);
    }

}
