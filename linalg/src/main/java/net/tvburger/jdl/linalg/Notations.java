package net.tvburger.jdl.linalg;

public final class Notations {

    private Notations() {
    }

    public static final String INVERSE = "⁻¹";
    public static final String TRANSPOSED = "ᵀ";
    public static final String PHI = "Φ";
    public static final String PSEUDO_INVERSE = "†";

    public static final String LAMBDA = "λ";

    public static String group(String notation) {
        return "(" + notation + ")";
    }

}
