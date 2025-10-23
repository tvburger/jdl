package net.tvburger.jdl.common.numbers;

public interface Scalar<N> extends Tensor<N> {

    int[] DIMENSIONS = new int[0];

    N get();

    static <N> Scalar<N> of(N value) {
        return new Scalar<>() {

            @Override
            public N get(int i) {
                return value;
            }

            @Override
            public N get(int... i) {
                return value;
            }

            @Override
            public int[] dimensions() {
                return DIMENSIONS;
            }

            @Override
            public N get() {
                return value;
            }
        };
    }
}
