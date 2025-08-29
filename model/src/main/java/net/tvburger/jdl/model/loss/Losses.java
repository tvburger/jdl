package net.tvburger.jdl.model.loss;

import net.tvburger.jdl.common.patterns.StaticUtility;

@StaticUtility
public final class Losses {

    private static final LossFunction mse = new ScaledError(0.5f, new MeanError(new SquaredError()));
    private static final LossFunction bce = new BinaryCrossEntropy();

    private Losses() {
    }

    public static LossFunction mSE() {
        return mse;
    }

    public static LossFunction bCE() {
        return bce;
    }
}
