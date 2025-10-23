package net.tvburger.jdl.common.numbers;

public interface Tensor<N> {

    N get(int i);

    N get(int... i);

    default int span() {
        return dimensions().length;
    }

    int[] dimensions();

}
