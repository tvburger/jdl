package net.tvburger.jdl.cnn;

public interface ConvolutionalNetworkLayer {

    ConvolutionalShape getInputShape();

    ConvolutionalShape getOutputShape();

    ConvolutionalShape transform(ConvolutionalShape input);

}
