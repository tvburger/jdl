package net.tvburger.jdl.model.training.regularization;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.numbers.NumberTypeAgnostic;

public class RegularizationFactory<N extends Number> implements NumberTypeAgnostic<N> {

    private final JavaNumberTypeSupport<N> typeSupport;

    public RegularizationFactory(JavaNumberTypeSupport<N> typeSupport) {
        this.typeSupport = typeSupport;
    }

    public LASSO<N> createLASSO(double lambda) {
        LASSO<N> lasso = new LASSO<>(typeSupport);
        lasso.setLambda(typeSupport.valueOf(lambda));
        return lasso;
    }

    public Ridge<N> createRidge(double lambda) {
        Ridge<N> ridge = new Ridge<>(typeSupport);
        ridge.setLambda(typeSupport.valueOf(lambda));
        return ridge;
    }

    public ElasticNet<N> createElasticNet(double lambda1, double lambda2) {
        ElasticNet<N> elasticNet = new ElasticNet<>(typeSupport);
        elasticNet.setLambda1(typeSupport.valueOf(lambda1));
        elasticNet.setLambda2(typeSupport.valueOf(lambda2));
        return elasticNet;
    }

    @Override
    public JavaNumberTypeSupport<N> getNumberTypeSupport() {
        return typeSupport;
    }
}
