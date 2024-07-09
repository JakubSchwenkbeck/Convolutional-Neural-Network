package layers;

import java.util.List;
import java.util.Random;

public class FCLayer extends Layer{
    //FullyConnectedLayer inherits from abstract Layer class

    private double[][] _weights; //set of weights in matrix

    private long seed; //seed, used to random generate weights in process


    private int _inputLength;
    private int _outputLength;

    private double learningrate;

    // keeping track of last outputs and inputs
    private double[] OutputArray ;
    private double[] LastInput ;
    //Constructor
    public FCLayer(int inputLength, int outputLength, long seed, double learningrate) {
        this._inputLength = inputLength;
        this._outputLength = outputLength;
        this.learningrate = learningrate;
        this.seed = seed;
        _weights = new double[_inputLength][_outputLength];
        randomSeed(); //fills the seed with random gaussian
    }


    //FC Forward Pass calculates and then passes outputs :
    public double[] FullyConnectedLayerForwardPass (double[] input){
        LastInput = input;
        double[] mid = new  double[_outputLength];
 //mid is z
        for (int i = 0; i < _inputLength; i++) {
            for (int j = 0; j < _outputLength; j++) {
                mid[j] = input[i]*_weights[i][j]; // fill array with weighted values
                OutputArray = mid;
                mid[j] = ReluAct(mid[j]); //smooth via ReluActivation function

            }
        }


        //might be able to do it in the first loop?
        /*
        double[] out = new double[_outputLength];
        for (int i = 0; i < _inputLength; i++) {
            for (int j = 0; j < _outputLength; j++) {
                out[j] = ReluAct(mid[j]);
            }

        } */

        return mid;
    }


    //overriding abstract functions
    @Override
    public int getOutputLength() {
        return 0;
    }

    @Override
    public int getOutputRow() {
        return 0;
    }

    @Override
    public int getOutputCol() {
        return 0;
    }

    @Override
    public int getOutputElements() {
        return _outputLength;
    }


    // convert via outputs and ovverride with FCForwardPass
    @Override
    public double[] getOutput(List<double[][]> input) {
        double[] vec = matrixToVector(input);
        return getOutput(vec);


    }

    @Override
    public double[] getOutput(double[] input) {
        double[] fpRes = FullyConnectedLayerForwardPass(input);

        if(_nextLayer != null){

            return _nextLayer.getOutput(fpRes);
        }else {return fpRes;}

    }

    @Override
    public void backPropagation(double[] dLdO) {
        double[] dldX = new  double[_inputLength];

        double dOdZ ; // differentiated Relu = 0 if z<0, 1 if z >= 0
        double dZdw ; //differntiated x*w = x
        double dLdw; // loss by weight  to update new weights
        double dZdX; // Z by input

        for (int i = 0; i < _inputLength; i++) {
            double dLdXsummation = 0; // loss by input
            for (int j = 0; j < _outputLength; j++) {

                dOdZ = derivativeReluAct(OutputArray[j]);
                dZdw = LastInput[i];
                dZdX = _weights[i][j];

                dLdw = dLdO[j] * dZdw *dOdZ; //via chainrule
                    //dLdw is the loss for each weight
                //substract dLdw for learning process ( scalar learningrate)
                _weights[i][j] -= dLdw*learningrate;

                dLdXsummation += dLdO[j]*dOdZ*dZdX  ; //Loss by mid output * output by midoutput * (dZdX = old weights)
            }
            dldX[i] = dLdXsummation;
        }
        if(_prevLayer != null){
        _prevLayer.backPropagation(dldX);}

    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {
        double[] vec = matrixToVector(dLdO);
         backPropagation(vec);
    }


    // through random.nextGaussian we get random values nicely distributed around seed for the weights
    public void randomSeed (){

        Random random = new Random(seed);

        for (int i = 0; i < _inputLength; i++) {
            for (int j = 0; j < _outputLength ; j++) {
                _weights[i][j] = random.nextGaussian();
            }

        }


    }

    // ReluActivation
    // z <= 0  -> 0
    // z >  0  -> z
    //helps with not punishing negative interpretations
    public double ReluAct (double input){

        if(input<= 0) {return 0;}else{return input;}

    }

    //calc the derivative of the relu func
    // z < 0 -> 0
    // z >=0 -> 1
    public double derivativeReluAct (double input){
        // Added a Leak of 0.001 to prevent deadspaces!
        if(input< 0) {return 0.001;}else{return 1;}

    }


}
