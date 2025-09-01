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

    public static ChainedRegime.Builder epochs(int epochs) {
        return new ChainedRegime.Builder().epochs(epochs);
    }

    public static ChainedRegime.Builder reportObjective() {
        return new ChainedRegime.Builder().reportObjective();
    }

    public static ChainedRegime.Builder dumpNodes() {
        return new ChainedRegime.Builder().dumpNodes();
    }

    public static ChainedRegime.Builder dumpNodes(boolean firstTime) {
        return new ChainedRegime.Builder().dumpNodes(firstTime);
    }

    public final ChainedRegime.Builder dumpNodes(boolean firstTime, boolean includeInputs) {
        return new ChainedRegime.Builder().dumpNodes(firstTime, includeInputs);
    }

}