package net.tvburger.jdl.cnn.pooling;

import net.tvburger.jdl.cnn.PoolingFunction;
import net.tvburger.jdl.common.numbers.Array;

public class MaximumPooling implements PoolingFunction {

    @Override
    public Float pool(Array<Float> elements) {
        Float max = null;
        for (int i = 1; i < elements.length(); i++) {
            if (max == null || max < elements.get(i)) {
                max = elements.get(i);
            }
        }
        return max;
    }
}
