package net.tvburger.dlp.activations;

public final class Activations {

    private static final ReLU relu = new ReLU();

    private Activations() {
    }

    public static ReLU reLU() {
        return relu;
    }

}
