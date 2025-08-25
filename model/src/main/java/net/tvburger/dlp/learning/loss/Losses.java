package net.tvburger.dlp.learning.loss;

public final class Losses {

    private static final MSE mse0_5 = new MSE(0.5f);
    private static final MSE mse1 = new MSE(1.0f);


    private Losses() {
    }

    public static MSE halfMSE() {
        return mse0_5;
    }

    public static MSE mSE() {
        return mse1;
    }
}
