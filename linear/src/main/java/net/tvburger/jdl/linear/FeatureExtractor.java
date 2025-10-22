package net.tvburger.jdl.linear;

import net.tvburger.jdl.common.numbers.Array;
import net.tvburger.jdl.common.numbers.JavaNumberTypeSupport;
import net.tvburger.jdl.common.patterns.Strategy;

@Strategy(Strategy.Role.INTERFACE)
public interface FeatureExtractor<N extends Number> {

    JavaNumberTypeSupport<N> getTypeSupport();

    Array<N> extractFeatures(N input);

    int featureCount();

}
