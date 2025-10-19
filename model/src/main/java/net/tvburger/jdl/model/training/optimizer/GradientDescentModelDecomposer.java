package net.tvburger.jdl.model.training.optimizer;

import net.tvburger.jdl.common.patterns.Strategy;
import net.tvburger.jdl.linalg.Vector;
import net.tvburger.jdl.model.scalars.LinearCombination;
import net.tvburger.jdl.model.training.TrainableFunction;

import java.util.stream.Stream;

@Strategy(Strategy.Role.INTERFACE)
public interface GradientDescentModelDecomposer<E extends TrainableFunction<N>, N extends Number> {

    record GradientDecomposition<N extends Number>(LinearCombination<N> linearCombination, Vector<N> parameterGradients) {
    }

    Stream<GradientDecomposition<N>> calculateDecompositionGradients(E model, Vector<N> objectiveGradients, N[] inputs);

}
