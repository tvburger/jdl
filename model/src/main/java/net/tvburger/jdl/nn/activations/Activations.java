package net.tvburger.jdl.nn.activations;

public final class Activations {

    private static final ReLU relu = new ReLU();
    private static final Linear linear = new Linear();
    private static final Sigmoid sigmoid = new Sigmoid();
    private static final Step step = new Step();

    private Activations() {
    }

    public static ReLU reLU() {
        return relu;
    }

    public static Linear linear() {
        return linear;
    }

    public static Sigmoid sigmoid() {
        return sigmoid;
    }

    public static Step step() {
        return step;
    }
}
