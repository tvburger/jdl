package net.tvburger.jdl.linear.optimizer;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.linalg.TypedVector;
import net.tvburger.jdl.linalg.Vector;
import net.tvburger.jdl.linear.LinearBasisFunctionModel;
import net.tvburger.jdl.model.training.optimizer.GradientDescentModelDecomposer;

import java.util.stream.Stream;

public class LinearBasisFunctionModelDecomposer<N extends Number> implements GradientDescentModelDecomposer<LinearBasisFunctionModel<N>, N> {

    @Override
    public Stream<GradientDecomposition<N>> calculateDecompositionGradients(LinearBasisFunctionModel<N> model, Vector<N> objectiveGradients, N[] inputs) {
        if (objectiveGradients.getDimensions() != 1) {
            throw new IllegalArgumentException("Objective gradients must have exactly one dimension");
        }
        JavaNumberTypeSupport<N> typeSupport = model.getCurrentNumberType();
        N[] features = model.getFeatureExtractor().extractFeatures(inputs[0]);
        N[] inputGradients = typeSupport.createArray(features.length + 1);
        inputGradients[0] = objectiveGradients.get(1);
        for (int i = 0; i < features.length; i++) {
            inputGradients[i + 1] = typeSupport.multiply(objectiveGradients.get(1), features[i]);
        }
        Vector<N> inputGradientsVector = new TypedVector<>(inputGradients, true, typeSupport);
        return Stream.of(new GradientDecomposition<>(model, inputGradientsVector));
    }

}
