package network;

import data.Image;
import layer.Layer;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import static data.MatrixUtility.add;
import static data.MatrixUtility.multiply;

public class NeuralNetwork implements Serializable {
    private static final long serialVersionUID = 1L;

    private List<Layer> layers;
    private double scaleFactor;

    public NeuralNetwork(List<Layer> layers, double scaleFactor) {
        this.layers = layers;
        this.scaleFactor = scaleFactor;
        linkLayers();
    }

    public double[] getProbabilities(Image image) {
        // Scale the input image data for the forward pass
        List<double[][]> inputData = new ArrayList<>();
        inputData.add(multiply(image.getData(), 1.0 / scaleFactor));

        // Perform the forward pass and return the output
        return layers.get(0).getOutput(inputData);
    }

    private void linkLayers() {
        for (int i = 0; i < layers.size(); i++) {
            Layer currentLayer = layers.get(i);
            if (i > 0) {
                currentLayer.setPreviousLayer(layers.get(i - 1));
            }
            if (i < layers.size() - 1) {
                currentLayer.setNextLayer(layers.get(i + 1));
            }
        }
    }

    public double[] getErrors(double[] networkOutput, int correctAnswer) {
        int numClasses = networkOutput.length;

        double[] expectedOutput = new double[numClasses];
        expectedOutput[correctAnswer] = 1;

        return add(networkOutput, multiply(expectedOutput, -1));
    }

    private int getMaxIndex(double[] output) {
        int maxIndex = 0;
        double maxValue = output[0];

        for (int i = 1; i < output.length; i++) {
            if (output[i] > maxValue) {
                maxValue = output[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    public int guess(Image image) {
        // Scale the image data for the forward pass
        List<double[][]> inputData = new ArrayList<>();
        inputData.add(multiply(image.getData(), 1.0 / scaleFactor));

        // Perform the forward pass and return the predicted class
        double[] output = layers.get(0).getOutput(inputData);
        return getMaxIndex(output);
    }

    public float test(List<Image> images) {
        int correctCount = 0;

        for (Image img : images) {
            if (guess(img) == img.getLabel()) {
                correctCount++;
            }
        }

        return (float) correctCount / images.size();
    }

    public void train(List<Image> images) {
        for (Image img : images) {
            List<double[][]> inputData = new ArrayList<>();
            inputData.add(multiply(img.getData(), 1.0 / scaleFactor));

            // Forward pass to get the network output
            double[] output = layers.get(0).getOutput(inputData);

            // Compute the error and perform backpropagation
            double[] error = getErrors(output, img.getLabel());
            layers.get(layers.size() - 1).backPropagation(error);
        }
    }
}
