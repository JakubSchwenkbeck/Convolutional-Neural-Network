package layer;

import layer.Layer;

import java.util.ArrayList;
import java.util.List;

public class MaxPoolLayer extends Layer {

    private int _stepSize;
    private int _windowSize;

    private int _inLength;
    private int _inRows;
    private int _inCols;

    List<int[][]> _lastMaxRow;
    List<int[][]> _lastMaxCol;


    public MaxPoolLayer(int _stepSize, int _windowSize, int _inLength, int _inRows, int _inCols) {
        this._stepSize = _stepSize;
        this._windowSize = _windowSize;
        this._inLength = _inLength;
        this._inRows = _inRows;
        this._inCols = _inCols;
    }

    public List<double[][]> maxPoolForwardPass(List<double[][]> input) {

        List<double[][]> output = new ArrayList<>();
        _lastMaxRow = new ArrayList<>();
        _lastMaxCol = new ArrayList<>();

        for(int l =0; l < input.size(); l++){
            output.add(pool(input.get(l)));
        }

        return output;

    }

    public double[][] pool(double[][] input){

        double[][] output = new double[getOutputRow()][getOutputCol()];

        int[][] maxRows = new int[getOutputRow()][getOutputCol()];
        int[][] maxCols = new int[getOutputRow()][getOutputCol()];

        for(int r = 0; r < getOutputRow(); r+= _stepSize){
            for(int c = 0; c < getOutputCol(); c+= _stepSize){

                double max = 0.0;
                maxRows[r][c] = -1;
                maxCols[r][c] = -1;

                for(int x = 0; x < _windowSize; x++){
                    for(int y = 0; y < _windowSize; y++) {
                        if(max < input[r+x][c+y]){
                            max = input[r+x][c+y];
                            maxRows[r][c] = r+x;
                            maxCols[r][c] = c+y;
                        }
                    }
                }

                output[r][c] = max;

            }
        }

        _lastMaxRow.add(maxRows);
        _lastMaxCol.add(maxCols);

        return output;

    }


    @Override
    public double[] getOutput(List<double[][]> input) {
        List<double[][]> outputPool = maxPoolForwardPass(input);
        return _nextLayer.getOutput(outputPool);
    }

    @Override
    public double[] getOutput(double[] input) {
        List<double[][]> matrixList = vectorToMatrix(input, _inLength, _inRows, _inCols);
        return getOutput(matrixList);
    }

    @Override
    public void backPropagation(double[] dLdO) {
        List<double[][]> matrixList = vectorToMatrix(dLdO, getOutputLength(), getOutputRow(), getOutputCol());
        backPropagation(matrixList);
    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {

        List<double[][]> dXdL = new ArrayList<>();

        int l = 0;
        for(double[][] array: dLdO){
            double[][] error = new double[_inRows][_inCols];
            //System.out.println("Error lenth " + error.length);
         //   System.out.println("Array length : "+ array.length +" " + array[0].length);
            for(int r = 0; r < array.length-1; r++){
                for(int c = 0; c < array[0].length-1; c++){
                    if(l<8){
                    int max_i = _lastMaxRow.get(l)[r][c];
                    int max_j = _lastMaxCol.get(l)[r][c];

                    if((max_i != -1) && (r < array.length-1)){
                //        System.out.println("R :" + r + " and C: "+ c);
                        // System.out.println("Max i : " + max_i + "Max j : "+ max_j);
                        error[max_i][max_j] += array[r][c];
                    }}
                }
            }

            dXdL.add(error);
            l++;
        }

        if(_prevLayer!= null){
            _prevLayer.backPropagation(dXdL);
        }

    }

    @Override
    public int getOutputLength() {
        return _inLength;
    }

    @Override
    public int getOutputRow() {
       // System.out.println("outrows : " + (_inRows-_windowSize)/_stepSize + 1);
        return (_inRows-_windowSize)/_stepSize + 1;
    }

    @Override
    public int getOutputCol() {
       // System.out.println("outcols : "+(_inCols-_windowSize)/_stepSize + 1);


        return (_inCols-_windowSize)/_stepSize + 1;
    }

    @Override
    public int getOutputElements() {
        return _inLength*getOutputCol()*getOutputRow();
    }
}