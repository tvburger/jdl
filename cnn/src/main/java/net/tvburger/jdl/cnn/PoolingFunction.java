package net.tvburger.jdl.cnn;

import net.tvburger.jdl.common.numbers.Array;

public interface PoolingFunction {

    Float pool(Array<Float> elements);

}
