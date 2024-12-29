package layer;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static data.MatrixUtility.add;
import static data.MatrixUtility.multiply;

// Convolutional layer for neural networks. This layer performs convolutions on input data using multiple filters.
public class ConvolutionLayer extends Layer {

    private long randomSeed; // Random seed for filter initialization.

    private List<double[][]> filters; // List of filters used for convolution.
    private int filterSize; // Size of each filter (assumed to be square).
    private int stepSize; // Step size for sliding the filter across the input.

    private int inputDepth; // Number of input feature maps.
    private int inputRows; // Number of rows in each input feature map.
    private int inputCols; // Number of columns in each input feature map.
    private double learningRate; // Learning rate for filter updates.

    private List<double[][]> lastInput; // Stores the last input passed to the layer for backpropagation.

    // Constructor for initializing the ConvolutionLayer.
    public ConvolutionLayer(int filterSize, int stepSize, int inputDepth, int inputRows, int inputCols, long randomSeed, int numFilters, double learningRate) {
        this.filterSize = filterSize;
        this.stepSize = stepSize;
        this.inputDepth = inputDepth;
        this.inputRows = inputRows;
        this.inputCols = inputCols;
        this.randomSeed = randomSeed;
        this.learningRate = learningRate;

        initializeFilters(numFilters);
    }

    // Initializes the filters with random values using a Gaussian distribution.
    private void initializeFilters(int numFilters) {
        List<double[][]> filterList = new ArrayList<>();
        Random random = new Random(randomSeed);

        for (int i = 0; i < numFilters; i++) {
            double[][] newFilter = new double[filterSize][filterSize];

            for (int row = 0; row < filterSize; row++) {
                for (int col = 0; col < filterSize; col++) {
                    newFilter[row][col] = random.nextGaussian(); // Assign a random value to the filter.
                }
            }

            filterList.add(newFilter);
        }

        this.filters = filterList;
    }

    // Performs a forward pass through the convolution layer.
    public List<double[][]> forwardPass(List<double[][]> input) {
        lastInput = input; // Store the input for backpropagation.

        List<double[][]> output = new ArrayList<>();

        // Convolve each input feature map with every filter.
        for (double[][] featureMap : input) {
            for (double[][] filter : filters) {
                output.add(convolve(featureMap, filter, stepSize));
            }
        }

        return output;
    }

    // Applies a single filter to the input using the specified step size.
    private double[][] convolve(double[][] input, double[][] filter, int stepSize) {
        int outputRows = (input.length - filter.length) / stepSize + 1;
        int outputCols = (input[0].length - filter[0].length) / stepSize + 1;

        double[][] output = new double[outputRows][outputCols];

        int outputRow = 0;

        // Slide the filter over the input matrix.
        for (int row = 0; row <= input.length - filter.length; row += stepSize) {
            int outputCol = 0;
            for (int col = 0; col <= input[0].length - filter[0].length; col += stepSize) {
                double sum = 0.0;

                // Perform element-wise multiplication between the filter and input.
                for (int filterRow = 0; filterRow < filter.length; filterRow++) {
                    for (int filterCol = 0; filterCol < filter[0].length; filterCol++) {
                        sum += filter[filterRow][filterCol] * input[row + filterRow][col + filterCol];
                    }
                }

                output[outputRow][outputCol] = sum; // Store the result in the output matrix.
                outputCol++;
            }
            outputRow++;
        }

        return output;
    }

    // Expands the input array by inserting zeros between the elements, based on the step size.
    public double[][] addPadding(double[][] input) {
        if (stepSize == 1) {
            return input; // No padding needed for step size 1.
        }

        int outputRows = (input.length - 1) * stepSize + 1;
        int outputCols = (input[0].length - 1) * stepSize + 1;

        double[][] output = new double[outputRows][outputCols];

        // Insert the input values at the appropriate spaced positions.
        for (int row = 0; row < input.length; row++) {
            for (int col = 0; col < input[0].length; col++) {
                output[row * stepSize][col * stepSize] = input[row][col];
            }
        }

        return output;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        List<double[][]> output = forwardPass(input);
        return nextLayer.getOutput(output); // Pass the result to the next layer.
    }

    @Override
    public double[] getOutput(double[] input) {
        List<double[][]> matrixInput = vectorToMatrix(input, inputDepth, inputRows, inputCols);
        return getOutput(matrixInput);
    }

    @Override
    public void backPropagation(double[] gradients) {
        List<double[][]> matrixInput = vectorToMatrix(gradients, inputDepth, inputRows, inputCols);
        backPropagation(matrixInput);
    }

    @Override
    public void backPropagation(List<double[][]> gradients) {

        List<double[][]> filterGradients = new ArrayList<>();
        List<double[][]> inputGradients = new ArrayList<>();

        // Initialize filter gradients.
        for (int i = 0; i < filters.size(); i++) {
            filterGradients.add(new double[filterSize][filterSize]);
        }

        // Compute the gradients for each filter and the input.
        for (int i = 0; i < lastInput.size(); i++) {

            double[][] inputError = new double[inputRows][inputCols];

            for (int filterIndex = 0; filterIndex < filters.size(); filterIndex++) {

                double[][] currentFilter = filters.get(filterIndex);
                double[][] error = gradients.get(i * filters.size() + filterIndex);

                double[][] paddedError = addPadding(error);
                double[][] filterGradient = convolve(lastInput.get(i), paddedError, 1);

                double[][] delta = multiply(filterGradient, learningRate * -1);
                double[][] updatedFilterGradient = add(filterGradients.get(filterIndex), delta);
                filterGradients.set(filterIndex, updatedFilterGradient);

                double[][] flippedError = flipHorizontally(flipVertically(paddedError));
                inputError = add(inputError, fullConvolve(currentFilter, flippedError));
            }

            inputGradients.add(inputError);
        }

        // Update filters using the computed gradients.
        for (int i = 0; i < filters.size(); i++) {
            double[][] updatedFilter = add(filterGradients.get(i), filters.get(i));
            filters.set(i, updatedFilter);
        }

        if (previousLayer != null) {
            previousLayer.backPropagation(inputGradients);
        }
    }

    // Flips the input array horizontally.
    public double[][] flipHorizontally(double[][] array) {
        int rows = array.length;
        int cols = array[0].length;

        double[][] output = new double[rows][cols];

        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                output[rows - row - 1][col] = array[row][col];
            }
        }
        return output;
    }

    // Flips the input array vertically.
    public double[][] flipVertically(double[][] array) {
        int rows = array.length;
        int cols = array[0].length;

        double[][] output = new double[rows][cols];

        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                output[row][cols - col - 1] = array[row][col];
            }
        }
        return output;
    }

    // Performs full convolution with the input and filter, including boundary conditions.
    private double[][] fullConvolve(double[][] input, double[][] filter) {
        int outputRows = (input.length + filter.length) + 1;
        int outputCols = (input[0].length + filter[0].length) + 1;

        double[][] output = new double[outputRows][outputCols];

        int outputRow = 0;

        // Apply the filter across the input with boundary handling.
        for (int row = -filter.length + 1; row < input.length; row++) {
            int outputCol = 0;
            for (int col = -filter[0].length + 1; col < input[0].length; col++) {

                double sum = 0.0;

                // Apply the filter around the current position in the input.
                for (int filterRow = 0; filterRow < filter.length; filterRow++) {
                    for (int filterCol = 0; filterCol < filter[0].length; filterCol++) {
                        int inputRowIndex = row + filterRow;
                        int inputColIndex = col + filterCol;

                        if (inputRowIndex >= 0 && inputColIndex >= 0 && inputRowIndex < input.length && inputColIndex < input[0].length) {
                            double value = filter[filterRow][filterCol] * input[inputRowIndex][inputColIndex];
                            sum += value;
                        }
                    }
                }

                output[outputRow][outputCol] = sum;
                outputCol++;
            }

            outputRow++;
        }

        return output;
    }

    @Override
    public int getOutputLength() {
        return filters.size() * inputDepth;
    }

    @Override
    public int getOutputRows() {
        return (inputRows - filterSize) / stepSize + 1;
    }

    @Override
    public int getOutputCols() {
        return (inputCols - filterSize) / stepSize + 1;
    }

    @Override
    public int getOutputElements() {
        return getOutputCols() * getOutputRows() * getOutputLength();
    }
}
