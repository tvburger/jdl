package net.tvburger.jdl.cnn.pooling;

import net.tvburger.jdl.cnn.PoolingFunction;
import net.tvburger.jdl.common.numbers.Array;

public class AveragePooling implements PoolingFunction {

    @Override
    public Float pool(Array<Float> elements) {
        Float avg = 0.0f;
        for (int i = 1; i < elements.length(); i++) {
            avg += elements.get(i);
        }
        return avg / elements.length();
    }
}
