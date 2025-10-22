package net.tvburger.jdl.model.training.regularization;

import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.model.HyperparameterConfigurable;

import java.util.Map;

@Regularization.Target(Regularization.Target.Type.WEIGHTS)
@Regularization.Effect({Regularization.Effect.Type.SPARSITY, Regularization.Effect.Type.SHRINKAGE})
public class ElasticNet<N extends Number> implements ExplicitRegularization<N>, HyperparameterConfigurable {

    public static String HP_LAMBDA_1 = "lambda1";
    public static String HP_LAMBDA_2 = "lambda2";

    private final LASSO<N> lasso;
    private final Ridge<N> ridge;

    public ElasticNet(JavaNumberTypeSupport<N> typeSupport) {
        this(new LASSO<>(typeSupport), new Ridge<>(typeSupport));
    }

    private ElasticNet(LASSO<N> lasso, Ridge<N> ridge) {
        this.lasso = lasso;
        this.ridge = ridge;
    }

    @Override
    public JavaNumberTypeSupport<N> getNumberTypeSupport() {
        return lasso.getNumberTypeSupport();
    }

    public N getLambda1() {
        return lasso.getLambda();
    }

    public void setLambda1(N lambda1) {
        lasso.setLambda(lambda1);
    }

    public N getLambda2() {
        return ridge.getLambda();
    }

    public void setLambda2(N lambda2) {
        ridge.setLambda(lambda2);
    }

    public N lossPenalty(Array<N> parameters) {
        N loss1 = lasso.lossPenalty(parameters);
        N loss2 = ridge.lossPenalty(parameters);
        return getNumberTypeSupport().add(loss1, loss2);
    }

    public N gradientAdjustment(N parameter) {
        N grad1 = lasso.gradientAdjustment(parameter);
        N grad2 = ridge.gradientAdjustment(parameter);
        return getNumberTypeSupport().add(grad1, grad2);
    }

    @Override
    public Map<String, Object> getHyperparameters() {
        return Map.of(HP_LAMBDA_1, getLambda1(), HP_LAMBDA_2, getLambda2());
    }

    @Override
    public void setHyperparameter(String name, Object value) {
        if (HP_LAMBDA_1.equals(name)) {
            setLambda1(getNumberTypeSupport().cast(value));
        } else if (HP_LAMBDA_2.equals(name)) {
            setLambda2(getNumberTypeSupport().cast(value));
        }
    }
}
