package net.tvburger.jdl.common.utils;

public class Threads {

    private Threads() {
    }

    public static void runAsynchronously(Runnable runnable) {
        new Thread(runnable).start();
    }

    public static void sleepSilently(long millis) {
        if (millis <= 0) {
            return;
        }
        try {
            Thread.sleep(millis);
        } catch (InterruptedException cause) {
            Thread.currentThread().interrupt();
        }
    }

}
