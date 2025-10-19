package net.tvburger.jdl.model.training.regularization;

import net.tvburger.jdl.common.numbers.NumberTypeAgnostic;

import java.lang.annotation.*;

public interface Regularization<N extends Number> extends NumberTypeAgnostic<N> {

    @Inherited
    @Documented
    @Retention(RetentionPolicy.SOURCE)
    @java.lang.annotation.Target(ElementType.TYPE)
    @interface Mechanism {
        enum Type {
            PENALTY_BASED,
            CONSTRAINT_BASED,
            STOCHASTIC_BASED,
            EARLY_STOPPING
        }

        Type[] value();
    }

    @Inherited
    @Documented
    @Retention(RetentionPolicy.SOURCE)
    @java.lang.annotation.Target(ElementType.TYPE)
    @interface Target {
        enum Type {
            WEIGHTS,
            ACTIVATIONS,
            OUTPUTS,
            DATA
        }

        Type[] value();
    }

    @Inherited
    @Documented
    @Retention(RetentionPolicy.SOURCE)
    @java.lang.annotation.Target(ElementType.TYPE)
    @interface Effect {
        enum Type {
            SHRINKAGE,
            SPARSITY,
            SMOOTHING,
            ROBUSTNESS,
            STABILITY
        }

        Type[] value();
    }

}
