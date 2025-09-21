package net.tvburger.jdl.linear;

import net.tvburger.jdl.linear.basis.PolynomialFunction;

public class PolynomialRegression extends LinearBasisFunctionModel {

    public PolynomialRegression(int m) {
        super(m, new PolynomialFunction.Generator());
    }

}
