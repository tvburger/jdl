package net.tvburger.jdl.linear;

import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.linear.basis.PolynomialFunction;

public class PolynomialRegression<N extends Number> extends LinearBasisFunctionModel<N> {

    public PolynomialRegression(int m, JavaNumberTypeSupport<N> typeSupport) {
        super(m, new PolynomialFunction.Generator<>(typeSupport));
    }

}
