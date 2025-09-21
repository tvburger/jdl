package net.tvburger.jdl.linear;

import net.tvburger.jdl.common.patterns.Strategy;

@Strategy(Strategy.Role.INTERFACE)
public interface FeatureExtractor {

    float[] extractFeatures(float input);

    int featureCount();

}
