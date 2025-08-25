package net.tvburger.dlp.nn.initializers;

public final class Initializers {

    private static final RandomInitializer random = new RandomInitializer();

    private Initializers() {
    }

    public static RandomInitializer random() {
        return random;
    }

}
