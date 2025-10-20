package net.tvburger.jdl.cnn;

import net.tvburger.jdl.common.shapes.Shape4D;

import java.util.Arrays;

public class ConvolutionalShape implements Shape4D, Cloneable {

    public static final int DEFAULT_CHANNELS = 1;
    public static final int DEFAULT_CHANNEL = 1;
    public static final int DEFAULT_COUNT = 1;
    public static final int DEFAULT_INDEX = 0;

    private final int width;
    private final int height;
    private final int channels;
    private final int count;
    private final Float[] pixels;

    public ConvolutionalShape(int width, int height) {
        this(width, height, DEFAULT_CHANNELS);
    }

    public ConvolutionalShape(int width, int height, int channels) {
        this(width, height, channels, DEFAULT_COUNT);
    }

    public ConvolutionalShape(int width, int height, int channels, int count) {
        this(width, height, channels, count, new Float[width * height * channels * count]);
    }

    public ConvolutionalShape(int width, int height, int channels, int count, Float[] pixels) {
        this.width = width;
        this.height = height;
        this.channels = channels;
        this.count = count;
        this.pixels = pixels;
    }

    public Float[] getPixels() {
        return pixels;
    }

    @Override
    public int getWidth() {
        return width;
    }

    @Override
    public int getHeight() {
        return height;
    }

    @Override
    public int getDepth() {
        return channels;
    }

    @Override
    public int getCount() {
        return count;
    }

    public void setPixel(float value, int x, int y) {
        setPixel(value, x, y, DEFAULT_CHANNEL);
    }

    public void setPixel(float value, int x, int y, int channel) {
        setPixel(value, x, y, channel, DEFAULT_INDEX);
    }

    public void setPixel(float value, int x, int y, int channel, int index) {
        pixels[index * (width * height * channels) + (channel - 1) * (width * height) + y * width + x] = value;
    }

    public float getPixel(int x, int y) {
        return getPixel(x, y, DEFAULT_CHANNEL);
    }

    public float getPixel(int x, int y, int channel) {
        return getPixel(x, y, channel, DEFAULT_INDEX);
    }

    public float getPixel(int x, int y, int channel, int index) {
        return pixels[index * (width * height * channels) + (channel - 1) * (width * height) + y * width + x];
    }

    public int getChannels() {
        return channels;
    }

    @Override
    public ConvolutionalShape clone() {
        return new ConvolutionalShape(width, height, channels, count, pixels == null ? new Float[width * height * channels * count] : Arrays.copyOf(pixels, pixels.length));
    }

    public ConvolutionalShape withPixels(Float[] pixels) {
        return new ConvolutionalShape(width, height, channels, count, pixels);
    }
}
