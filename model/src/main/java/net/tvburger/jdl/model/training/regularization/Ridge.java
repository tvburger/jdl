package net.tvburger.jdl.model.training.regularization;

import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.model.HyperparameterConfigurable;

import java.util.Map;

@Regularization.Target(Regularization.Target.Type.WEIGHTS)
@Regularization.Effect(Regularization.Effect.Type.SHRINKAGE)
public class Ridge<N extends Number> implements ExplicitRegularization<N>, HyperparameterConfigurable {

    public static String HP_LAMBDA = "lambda";

    private final JavaNumberTypeSupport<N> typeSupport;
    private N lambda;
    private N lambdaTwice;

    public Ridge(JavaNumberTypeSupport<N> typeSupport) {
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
        this.lambdaTwice = getNumberTypeSupport().add(lambda, lambda);
    }

    @Override
    public N lossPenalty(Array<N> parameters) {
        N sum = getNumberTypeSupport().zero();
        for (N param : parameters) {
            sum = getNumberTypeSupport().add(sum, getNumberTypeSupport().multiply(param, param));
        }
        return getNumberTypeSupport().multiply(sum, lambda);
    }

    @Override
    public N gradientAdjustment(N parameter) {
        return getNumberTypeSupport().multiply(lambdaTwice, parameter);
    }

    @Override
    public Map<String, Object> getHyperparameters() {
        return Map.of(HP_LAMBDA, lambda);
    }

    @Override
    public void setHyperparameter(String name, Object value) {
        if (name.equals(HP_LAMBDA)) {
            this.lambda = getNumberTypeSupport().cast(value);
        }
    }
}
