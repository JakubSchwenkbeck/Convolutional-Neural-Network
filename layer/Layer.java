package layer;
import java.util.ArrayList;
import java.util.List;
public abstract class Layer {
//abstract class for all layer types to inherit from



    protected Layer _nextLayer;
    protected Layer _prevLayer;
    //getter and setter :
    public Layer get_nextLayer() {
        return _nextLayer;
    }

    public void set_nextLayer(Layer _nextLayer) {
        this._nextLayer = _nextLayer;
    }

    public Layer get_prevLayer() {
        return _prevLayer;
    }

    public void set_prevLayer(Layer _prevLayer) {
        this._prevLayer = _prevLayer;
    }


    //get output attributes
    public abstract int getOutputLength();
    public abstract int getOutputRow();
    public abstract int getOutputCol();
    public abstract int getOutputElements();


    // get output and backPropagation(like stack call) for both 1 and 2 dim arrays
    public abstract double[] getOutput(List<double[][]> input);
    public abstract double[] getOutput(double[] input);

    public abstract void backPropagation(double[] dLdO);
    public abstract void backPropagation(List<double[][]> dLdO);


    //conversion from one output type to the other
    public double[] matrixToVector(List<double[][]> input){
        int length = input.size();
        int rows = input.get(0).length;
        int cols = input.get(0)[0].length;

        int index = 0;
        double[] vec = new double[length*rows*cols];
        for (int i = 0; i < length; i++) {
            for (int j = 0; j < rows; j++) {
                for (int k = 0; k <cols ; k++) {
                    vec[index] = input.get(i)[j][k];
                    index++;
                }
            }
        }


        return vec;
    }


    public List<double[][]> vectorToMatrix(double[] input, int rows, int cols,int length) {
        List<double[][]> output = new ArrayList<>();
        int index = 0;

        for (int i = 0; i < length; i++) {


            double[][] mat = new double[rows][cols];


            for (int j = 0; j < rows; j++) {
                for (int k = 0; k < cols; k++) {
                    mat[j][k] = input[index];
                    index++;
                }
            }

            output.add(mat);

        }
        return output;
    }



}