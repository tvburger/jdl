package net.tvburger.jdl.model.training.regimes;

import net.tvburger.jdl.common.patterns.StaticUtility;

@StaticUtility
public final class Regimes {

    private Regimes() {
    }

    public static BatchRegime batch() {
        return new BatchRegime();
    }

    public static MiniBatchRegime miniBatch(int batchSize) {
        return new MiniBatchRegime(batchSize);
    }

    public static OnlineRegime online() {
        return new OnlineRegime();
    }

    public static ChainedRegime.BottomUpChainer chainBottom() {
        return new ChainedRegime.BottomUpChainer();
    }

    public static ChainedRegime.TopDownChainer chainTop() {
        return new ChainedRegime.TopDownChainer();
    }
}
