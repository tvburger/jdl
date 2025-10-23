package net.tvburger.jdl.model.training.optimizer;

import net.tvburger.jdl.common.numbers.NumberTypeAgnostic;
import net.tvburger.jdl.linalg.Vector;
import net.tvburger.jdl.model.training.TrainableFunction;
import net.tvburger.jdl.model.training.regularization.ExplicitRegularization;

import java.util.Set;

public interface UpdateStep<E extends TrainableFunction<N>, N extends Number> extends NumberTypeAgnostic<N> {

    Vector<N> calculateUpdate(Vector<N> gradients, E model, int step, Set<ExplicitRegularization<N>> regularizations);

}
