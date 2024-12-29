package layer;

import java.util.ArrayList;
import java.util.List;

public class MaxPoolLayer extends Layer {

    private int stepSize;
    private int windowSize;

    private int inLength;
    private int inRows;
    private int inCols;

    private List<int[][]> lastMaxRow;
    private List<int[][]> lastMaxCol;

    public MaxPoolLayer(int stepSize, int windowSize, int inLength, int inRows, int inCols) {
        this.stepSize = stepSize;
        this.windowSize = windowSize;
        this.inLength = inLength;
        this.inRows = inRows;
        this.inCols = inCols;
    }

    public List<double[][]> maxPoolForwardPass(List<double[][]> input) {
        List<double[][]> output = new ArrayList<>();
        lastMaxRow = new ArrayList<>();
        lastMaxCol = new ArrayList<>();

        for (double[][] inputMatrix : input) {
            output.add(pool(inputMatrix));
        }

        return output;
    }

    public double[][] pool(double[][] input) {
        int outputRows = getOutputRows();
        int outputCols = getOutputCols();
        double[][] output = new double[outputRows][outputCols];

        int[][] maxRows = new int[outputRows][outputCols];
        int[][] maxCols = new int[outputRows][outputCols];

        for (int r = 0; r < outputRows; r++) {
            for (int c = 0; c < outputCols; c++) {
                double max = Double.NEGATIVE_INFINITY;
                maxRows[r][c] = -1;
                maxCols[r][c] = -1;

                for (int x = 0; x < windowSize; x++) {
                    for (int y = 0; y < windowSize; y++) {
                        int rowIdx = r * stepSize + x;
                        int colIdx = c * stepSize + y;
                        if (rowIdx < inRows && colIdx < inCols && max < input[rowIdx][colIdx]) {
                            max = input[rowIdx][colIdx];
                            maxRows[r][c] = rowIdx;
                            maxCols[r][c] = colIdx;
                        }
                    }
                }

                output[r][c] = max;
            }
        }

        lastMaxRow.add(maxRows);
        lastMaxCol.add(maxCols);

        return output;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        List<double[][]> pooledOutput = maxPoolForwardPass(input);
        return nextLayer != null ? nextLayer.getOutput(pooledOutput) : new double[0];
    }

    @Override
    public double[] getOutput(double[] input) {
        List<double[][]> matrixList = vectorToMatrix(input, inLength, inRows, inCols);
        return getOutput(matrixList);
    }

    @Override
    public void backPropagation(double[] dLdO) {
        List<double[][]> matrixList = vectorToMatrix(dLdO, getOutputLength(), getOutputRows(), getOutputCols());
        backPropagation(matrixList);
    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {
        List<double[][]> dXdL = new ArrayList<>();

        for (int l = 0; l < dLdO.size(); l++) {
            double[][] error = new double[inRows][inCols];

            for (int r = 0; r < getOutputRows(); r++) {
                for (int c = 0; c < getOutputCols(); c++) {
                    int max_i = lastMaxRow.get(l)[r][c];
                    int max_j = lastMaxCol.get(l)[r][c];

                    if (max_i != -1 && max_j != -1) {
                        error[max_i][max_j] += dLdO.get(l)[r][c];
                    }
                }
            }

            dXdL.add(error);
        }

        if (previousLayer != null) {
            previousLayer.backPropagation(dXdL);
        }
    }

    @Override
    public int getOutputLength() {
        return inLength;
    }

    @Override
    public int getOutputRows() {
        return (inRows - windowSize) / stepSize + 1;
    }

    @Override
    public int getOutputCols() {
        return (inCols - windowSize) / stepSize + 1;
    }

    @Override
    public int getOutputElements() {
        return inLength * getOutputCols() * getOutputRows();
    }
}
