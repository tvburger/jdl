package net.tvburger.dlp.nn.activations;

public final class Activations {

    private static final ReLU relu = new ReLU();
    private static final Linear linear = new Linear();
    private static final Sigmoid sigmoid = new Sigmoid();

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
}
