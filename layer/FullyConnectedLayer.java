package layer;

import java.util.List;
import java.util.Random;

public class FullyConnectedLayer extends Layer {

    private static final double LEAK = 0.01;

    private double[][] weights;
    private int inLength;
    private int outLength;
    private double learningRate;

    private double[] lastZ;
    private double[] lastX;

    public FullyConnectedLayer(int inLength, int outLength, long seed, double learningRate) {
        this.inLength = inLength;
        this.outLength = outLength;
        this.learningRate = learningRate;

        weights = new double[inLength][outLength];
        setRandomWeights(seed);
    }

    public double[] fullyConnectedForwardPass(double[] input) {
        lastX = input;

        double[] z = new double[outLength];
        double[] output = new double[outLength];

        // Matrix multiplication: z = X * W
        for (int i = 0; i < inLength; i++) {
            for (int j = 0; j < outLength; j++) {
                z[j] += input[i] * weights[i][j];
            }
        }

        lastZ = z;

        // Apply ReLU activation: out = ReLU(z)
        for (int j = 0; j < outLength; j++) {
            output[j] = reLu(z[j]);
        }

        return output;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        double[] vector = matrixToVector(input);
        return getOutput(vector);
    }

    @Override
    public double[] getOutput(double[] input) {
        double[] forwardPass = fullyConnectedForwardPass(input);
        return (nextLayer != null) ? nextLayer.getOutput(forwardPass) : forwardPass;
    }

    @Override
    public void backPropagation(double[] dLdO) {
        double[] dLdX = new double[inLength];

        for (int k = 0; k < inLength; k++) {
            double dLdX_sum = 0;

            for (int j = 0; j < outLength; j++) {
                // Compute gradient for weights
                double dOdz = derivativeReLu(lastZ[j]);
                double dzdw = lastX[k];
                double dLdw = dLdO[j] * dOdz * dzdw;
                weights[k][j] -= dLdw * learningRate;

                // Compute gradient for input
                double dzdx = weights[k][j];
                dLdX_sum += dLdO[j] * dOdz * dzdx;
            }

            dLdX[k] = dLdX_sum;
        }

        if (previousLayer != null) {
            previousLayer.backPropagation(dLdX);
        }
    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {
        double[] vector = matrixToVector(dLdO);
        backPropagation(vector);
    }

    @Override
    public int getOutputLength() {
        return outLength;
    }

    @Override
    public int getOutputRows() {
        return 1; // Fully connected layers have 1 row in the output
    }

    @Override
    public int getOutputCols() {
        return outLength;
    }

    @Override
    public int getOutputElements() {
        return outLength;
    }

    private void setRandomWeights(long seed) {
        Random random = new Random(seed);

        for (int i = 0; i < inLength; i++) {
            for (int j = 0; j < outLength; j++) {
                weights[i][j] = random.nextGaussian(); // Initialize weights using a Gaussian distribution
            }
        }
    }

    private double reLu(double input) {
        return Math.max(0, input); // ReLU activation: max(0, input)
    }

    private double derivativeReLu(double input) {
        return (input <= 0) ? LEAK : 1; // Leaky ReLU derivative
    }
}
