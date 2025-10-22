package net.tvburger.jdl.model.training.regularization;

import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.model.HyperparameterConfigurable;

import java.util.Map;

/* Least Absolute Shrinkage and Selection Operator */
@Regularization.Target(Regularization.Target.Type.WEIGHTS)
@Regularization.Effect(Regularization.Effect.Type.SPARSITY)
public class LASSO<N extends Number> implements ExplicitRegularization<N>, HyperparameterConfigurable {

    public static String HP_LAMBDA = "lambda";

    private final JavaNumberTypeSupport<N> typeSupport;
    private N lambda;

    public LASSO(JavaNumberTypeSupport<N> typeSupport) {
        this.typeSupport = typeSupport;
    }

    @Override
    public JavaNumberTypeSupport<N> getNumberTypeSupport() {
        return typeSupport;
    }

    public N getLambda() {
        return lambda;
    }

    public void setLambda(N lambda) {
        this.lambda = lambda;
    }

    @Override
    public N lossPenalty(Array<N> parameters) {
        N sum = getNumberTypeSupport().zero();
        for (N param : parameters) {
            sum = getNumberTypeSupport().add(sum, getNumberTypeSupport().absolute(param));
        }
        return getNumberTypeSupport().multiply(sum, lambda);
    }

    @Override
    public N gradientAdjustment(N parameter) {
        if (getNumberTypeSupport().isZero(parameter)) {
            return getNumberTypeSupport().zero();
        } else if (getNumberTypeSupport().isPositive(parameter)) {
            return lambda;
        } else {
            return getNumberTypeSupport().negate(lambda);
        }
    }

    @Override
    public Map<String, Object> getHyperparameters() {
        return Map.of(HP_LAMBDA, getLambda());
    }

    @Override
    public void setHyperparameter(String name, Object value) {
        if (name.equals(HP_LAMBDA)) {
            setLambda(getNumberTypeSupport().cast(value));
        }
    }
}
