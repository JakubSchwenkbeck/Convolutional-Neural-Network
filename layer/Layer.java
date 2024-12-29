package layer;

import java.util.ArrayList;
import java.util.List;

// Abstract base class for neural network layers. Handles core layer connections and utility functions.
public abstract class Layer {

    // Reference to the next layer in the network.
    protected Layer nextLayer;

    // Reference to the previous layer in the network.
    protected Layer previousLayer;

    // Gets the next layer.
    public Layer getNextLayer() {
        return nextLayer;
    }

    // Sets the next layer.
    public void setNextLayer(Layer nextLayer) {
        this.nextLayer = nextLayer;
    }

    // Gets the previous layer.
    public Layer getPreviousLayer() {
        return previousLayer;
    }

    // Sets the previous layer.
    public void setPreviousLayer(Layer previousLayer) {
        this.previousLayer = previousLayer;
    }

    // Computes the output of the layer from a list of 3D input data.
    public abstract double[] getOutput(List<double[][]> input);

    // Computes the output of the layer from a 1D input array.
    public abstract double[] getOutput(double[] input);

    // Handles backpropagation with gradients given as a 1D array.
    public abstract void backPropagation(double[] gradients);

    // Handles backpropagation with gradients given as a list of 3D arrays.
    public abstract void backPropagation(List<double[][]> gradients);

    // Returns the total length of the layer's output as a flat vector.
    public abstract int getOutputLength();

    // Returns the number of rows in the layer's output matrix.
    public abstract int getOutputRows();

    // Returns the number of columns in the layer's output matrix.
    public abstract int getOutputCols();

    // Returns the total number of elements in the layer's output matrix.
    public abstract int getOutputElements();

    // Flattens a list of 3D matrices into a single 1D array.
    public double[] matrixToVector(List<double[][]> input) {
        int length = input.size(); // Number of matrices.
        int rows = input.get(0).length; // Rows in each matrix.
        int cols = input.get(0)[0].length; // Columns in each matrix.

        double[] vector = new double[length * rows * cols]; // Result array.
        int index = 0; // Keeps track of position in the result array.

        // Iterate through all matrices and flatten.
        for (int l = 0; l < length; l++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    vector[index] = input.get(l)[r][c];
                    index++;
                }
            }
        }

        return vector;
    }

    // Converts a flat 1D array into a list of 3D matrices with the specified dimensions.
    public List<double[][]> vectorToMatrix(double[] input, int length, int rows, int cols) {
        List<double[][]> output = new ArrayList<>(); // Holds the reconstructed matrices.
        int index = 0; // Keeps track of position in the input array.

        // Reconstruct matrices one by one.
        for (int l = 0; l < length; l++) {
            double[][] matrix = new double[rows][cols]; // Create a new matrix.

            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    matrix[r][c] = input[index]; // Fill matrix from input array.
                    index++;
                }
            }

            output.add(matrix); // Add the matrix to the result list.
        }

        return output;
    }
}
