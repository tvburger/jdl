package net.tvburger.jdl.model.training.regimes;

import net.tvburger.jdl.model.DataSet;
import net.tvburger.jdl.model.EstimationFunction;
import net.tvburger.jdl.model.nn.NeuralNetwork;
import net.tvburger.jdl.model.nn.NeuralNetworks;
import net.tvburger.jdl.model.training.ObjectiveFunction;
import net.tvburger.jdl.model.training.Optimizer;
import net.tvburger.jdl.model.training.Regime;

public class DumpNodesRegime extends DelegatedRegime {

    public DumpNodesRegime(Regime regime) {
        super(regime);
    }

    @Override
    public <E extends EstimationFunction> void train(E estimationFunction, DataSet trainingSet, ObjectiveFunction objective, Optimizer<? super E> optimizer) {
        if (estimationFunction instanceof NeuralNetwork neuralNetwork) {
            NeuralNetworks.dump(neuralNetwork);
        }
        regime.train(estimationFunction, trainingSet, objective, optimizer);
    }
}
