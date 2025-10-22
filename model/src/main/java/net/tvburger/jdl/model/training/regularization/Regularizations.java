package net.tvburger.jdl.model.training.regularization;

import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.patterns.StaticUtility;
import net.tvburger.jdl.linalg.TypedVector;
import net.tvburger.jdl.linalg.Vector;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

@StaticUtility
public final class Regularizations {

    private Regularizations() {
    }

    public static <N extends Number> Vector<N> applyExplicitRegularization(Set<ExplicitRegularization<N>> regularizations, Vector<N> thetas, Vector<N> gradients) {
        if (regularizations.isEmpty()) {
            return gradients;
        }
        Array<N> values = gradients.asArray().clone();
        for (ExplicitRegularization<N> regularization : regularizations) {
            for (int i = 0; i < thetas.getDimensions(); i++) {
                N adjustment = regularization.gradientAdjustment(thetas.get(i + 1));
                values.set(i, regularization.getNumberTypeSupport().add(values.get(i), adjustment));
            }
        }
        return new TypedVector<>(values, gradients.isColumnVector(), gradients.getNumberTypeSupport());
    }

    private static final Map<JavaNumberTypeSupport<?>, RegularizationFactory<?>> factories = new HashMap<>();

    @SuppressWarnings("unchecked")
    public static <N extends Number> RegularizationFactory<N> getFactory(JavaNumberTypeSupport<N> typeSupport) {
        if (!factories.containsKey(typeSupport)) {
            synchronized (factories) {
                if (!factories.containsKey(typeSupport)) {
                    factories.put(typeSupport, new RegularizationFactory<>(typeSupport));
                }
            }
        }
        return (RegularizationFactory<N>) factories.get(typeSupport);
    }
}
