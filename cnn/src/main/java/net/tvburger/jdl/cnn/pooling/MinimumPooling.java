package net.tvburger.jdl.cnn.pooling;

import net.tvburger.jdl.cnn.PoolingFunction;
import net.tvburger.jdl.common.numbers.Array;

public class MinimumPooling implements PoolingFunction {

    @Override
    public Float pool(Array<Float> elements) {
        Float min = null;
        for (int i = 1; i < elements.length(); i++) {
            if (min == null || min > elements.get(i)) {
                min = elements.get(i);
            }
        }
        return min;
    }
}
