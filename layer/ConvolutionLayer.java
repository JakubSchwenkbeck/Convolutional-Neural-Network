package layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static data.MatrixUtility.add;
import static data.MatrixUtility.multiply;

public class ConvolutionLayer extends Layer{


    //seed for random filters
    private long SEED;

    // list of generated filters
    private List<double[][]> _filters;

    //size of filters(square)
    private int _filterSize;

    //size of steps the filter takes
    private int _stepsize;

    //values of input
    private int _inLength;
    private int _inRows;
    private int _inCols;

    //decimal between 0 and 1
    private double _learningRate;

    private List<double[][]> _lastInput;

    //constructor
    public ConvolutionLayer(int _filterSize, int _stepsize, int _inLength, int _inRows, int _inCols, long SEED, int numFilters, double learningRate) {
        this._filterSize = _filterSize;
        this._stepsize = _stepsize;
        this._inLength = _inLength;
        this._inRows = _inRows;
        this._inCols = _inCols;
        this.SEED = SEED;
        _learningRate = learningRate;

        generateRandomFilters(numFilters);

    }

    //generates random filters according to seed ( gaussian random)
    private void generateRandomFilters(int numFilters){
        List<double[][]> filters = new ArrayList<>();
        Random random = new Random(SEED);

        for(int n = 0; n < numFilters; n++) {
            double[][] newFilter = new double[_filterSize][_filterSize];

            for(int i = 0; i < _filterSize; i++){
                for(int j = 0; j < _filterSize; j++){

                    double value = random.nextGaussian();
                    newFilter[i][j] = value;
                }
            }

            filters.add(newFilter);

        }

        _filters = filters;

    }

    // Comv FWP convolutes each filter with each input matrix
    public List<double[][]> convolutionForwardPass(List<double[][]> list){
        _lastInput = list;

        List<double[][]> output = new ArrayList<>();

        for (int m = 0; m < list.size(); m++){
            for(double[][] filter : _filters){
                output.add(convolve(list.get(m), filter, _stepsize));
            }

        }

        return output;

    }

    // convolution takes a matrix and a matrixshaped filter and convolves the matrix over the filter
    private double[][] convolve(double[][] input, double[][] filter, int stepSize) {

        int outRows = (input.length - filter.length)/stepSize + 1;
        int outCols = (input[0].length - filter[0].length)/stepSize + 1;

        int inRows = input.length;
        int inCols = input[0].length;

        int fRows = filter.length;
        int fCols = filter[0].length;

        double[][] output = new double[outRows][outCols];

        int outRow = 0;
        int outCol;

        for(int i = 0; i <= inRows - fRows; i += stepSize){

            outCol = 0;

            for(int j = 0; j <= inCols - fCols; j+= stepSize){

                double sum = 0.0;


                for(int x = 0; x < fRows; x++){
                    for(int y = 0; y < fCols; y++){
                        int inputRowIndex = i+x;
                        int inpurColIndex = j+y;

                        double value = filter[x][y] * input[inputRowIndex][inpurColIndex];
                        sum+= value;
                    }
                }

                output[outRow][outCol] = sum;
                outCol++;
            }

            outRow++;

        }

        return output;

    }

    //for stepsizes != 1
    public double[][] spaceArray(double[][] input){

        if(_stepsize == 1){
            return input;
        }

        int outRows = (input.length - 1)*_stepsize + 1;
        int outCols = (input[0].length -1)*_stepsize+1;

        double[][] output = new double[outRows][outCols];

        for(int i = 0; i < input.length; i++){
            for(int j = 0; j < input[0].length; j++){
                output[i*_stepsize][j*_stepsize] = input[i][j];
            }
        }

        return output;
    }


    @Override
    public double[] getOutput(List<double[][]> input) {

        List<double[][]> output = convolutionForwardPass(input);

        return _nextLayer.getOutput(output);

    }

    @Override
    public double[] getOutput(double[] input) {

        List<double[][]> matrixInput = vectorToMatrix(input, _inLength, _inRows, _inCols);

        return getOutput(matrixInput);
    }

    @Override
    public void backPropagation(double[] dLdO) {
        List<double[][]> matrixInput = vectorToMatrix(dLdO, _inLength, _inRows, _inCols);
        backPropagation(matrixInput);
    }


    // Many many math pages and videos:
    @Override
    public void backPropagation(List<double[][]> dLdO) {

        List<double[][]> filtersDelta = new ArrayList<>();
        List<double[][]> dLdOPreviousLayer= new ArrayList<>();

        for(int f = 0; f < _filters.size(); f++){
            filtersDelta.add(new double[_filterSize][_filterSize]);
        }

        for(int i = 0; i < _lastInput.size(); i++){

            double[][] errorForInput = new double[_inRows][_inCols];

            for(int f = 0; f < _filters.size(); f++){

                double[][] currFilter = _filters.get(f);
                double[][] error = dLdO.get(i*_filters.size() + f );

                double[][] spacedError = spaceArray(error);
                double[][] dLdF = convolve(_lastInput.get(i), spacedError, 1);

                double[][] delta = multiply(dLdF, _learningRate*-1);
                double[][] newTotalDelta = add(filtersDelta.get(f), delta);
                filtersDelta.set(f, newTotalDelta);

                double[][] flippedError = flipArrayHorizontal(flipArrayVertical(spacedError));
                errorForInput = add(errorForInput, fullConvolve(currFilter, flippedError));

            }

            dLdOPreviousLayer.add(errorForInput);

        }

        for(int f =0; f < _filters.size(); f++){
            double[][] modified = add(filtersDelta.get(f), _filters.get(f));
            _filters.set(f,modified);
        }

        if(_prevLayer!= null){
            _prevLayer.backPropagation(dLdOPreviousLayer);
        }
    }

    public double[][] flipArrayHorizontal(double[][] array){
        int rows = array.length;
        int cols = array[0].length;

        double[][] output = new double[rows][cols];

        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                output[rows-i-1][j] = array[i][j];
            }
        }
        return output;
    }

    public double[][] flipArrayVertical(double[][] array){
        int rows = array.length;
        int cols = array[0].length;

        double[][] output = new double[rows][cols];

        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                output[i][cols-j-1] = array[i][j];
            }
        }
        return output;
    }

    private double[][] fullConvolve(double[][] input, double[][] filter) {

        int outRows = (input.length + filter.length) + 1;
        int outCols = (input[0].length + filter[0].length) + 1;

        int inRows = input.length;
        int inCols = input[0].length;

        int fRows = filter.length;
        int fCols = filter[0].length;

        double[][] output = new double[outRows][outCols];

        int outRow = 0;
        int outCol;

        for(int i = -fRows + 1; i < inRows; i ++){

            outCol = 0;

            for(int j = -fCols + 1; j < inCols; j++){

                double sum = 0.0;

                //Apply Filter around this position
                for(int x = 0; x < fRows; x++){
                    for(int y = 0; y < fCols; y++){
                        int inputRowIndex = i+x;
                        int inputColIndex = j+y;

                        if(inputRowIndex >= 0 && inputColIndex >= 0 && inputRowIndex < inRows && inputColIndex < inCols){
                            double value = filter[x][y] * input[inputRowIndex][inputColIndex];
                            sum+= value;
                        }
                    }
                }

                output[outRow][outCol] = sum;
                outCol++;
            }

            outRow++;

        }

        return output;

    }

    @Override
    public int getOutputLength() {
        return _filters.size()*_inLength;
    }

    @Override
    public int getOutputRow() {
        return (_inRows-_filterSize)/_stepsize + 1;
    }

    @Override
    public int getOutputCol() {
        return (_inCols-_filterSize)/_stepsize + 1;
    }

    @Override
    public int getOutputElements() {
        return getOutputCol()*getOutputRow()*getOutputLength();
    }
}