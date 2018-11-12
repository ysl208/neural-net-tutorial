// neural-net-tutorial.cpp

#include <vector>
#include <iostream>
#include <cassert>
#include <cstdlib>
#include <cmath>

#include <fstream>
#include <stdlib.h>
#include <time.h>

using namespace std;

struct Connection
{
    double weight;
    double deltaWeight;
};

class Neuron;
typedef vector<Neuron> Layer;

// *** class Neuron ***
class Neuron
{
  public:
    Neuron(unsigned numOutputs, unsigned myIndex, double eta, double alpha);
    void setOutputVal(double val)
    {
        m_outputVal = val;
    }
    double getOutputVal(void) const { return m_outputVal; }
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);
    void setLearningRate(const double lr);
    void setMomentum(const double momentum);

  private:
    static double eta; // [0.0,....1.0] overall net training rate
    static double alpha; // [0.0,..n] multiplier of last weight change (momentum)
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    static double sigmoid(double x);
    static double sigmoidPrime(double x);
    static double randomWeight(void) { return rand() / double(RAND_MAX); };
    double sumDOW(const Layer &nextLayer) const;
    double m_outputVal;
    vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    double m_gradient;
};

double Neuron::eta = 0.15; //overall net learning rate
double Neuron::alpha = 0.5; // momentum, multiplier of last deltaWeight [0.0,...n]

void Neuron::setLearningRate(const double lr)
{
    eta = lr;
}

void Neuron::setMomentum(const double momentum)
{
    alpha = momentum;
}
void Neuron::updateInputWeights( Layer &prevLayer)
{
// the weights to be updated are in the Connection container
    // in the neurons in the preceding layer
    for (unsigned n = 0; n < prevLayer.size(); ++n){
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
        double newDeltaWeight = 
        // individual input, magnified by the gradient and train rate:
            eta 
            * neuron.getOutputVal()
            * m_gradient
            // also add momentum = a fraction of the previous delta weight
            + alpha
            * oldDeltaWeight;
        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }

} 
double Neuron::sumDOW(const Layer &nextLayer) const{
    double sum = 0.0;
    // sum our contributions of the errors at the nodes we feed
    for (unsigned n = 0; n < nextLayer.size() -1; ++n){
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }
    return sum;
}
void Neuron::calcOutputGradients(double targetVal){
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::sigmoidPrime(m_outputVal);
}
void Neuron::calcHiddenGradients(const Layer &nextLayer){
    // no target value to compare it with
    // so look at sum of derivatives of weights of the next layer
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::sigmoid(m_outputVal);
}
double Neuron::transferFunction(double x)
{
    // tanh - output range [-1.0...1.0]
    // to scale the data
    return tanh(x);
}
double Neuron::transferFunctionDerivative(double x)
{
    // tanh derivative
    return 1 - x * x;
}

// the sigmoid function
double Neuron::sigmoid(double x){
    return 1/(1+exp(-x));
}
// the derivative of the sigmoid function
double Neuron::sigmoidPrime(double x){
    return exp(-x)/(pow(1+exp(-x), 2));
}

void Neuron::feedForward(const Layer &prevLayer)
{
    double sum = 0.0;
    // Sum the previous layer's outputs (which are our inputs)
    // Include the bias node from the previous layer

    for (unsigned n = 0; n < prevLayer.size(); ++n)
    {
        sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;

    }
    m_outputVal = Neuron::sigmoid(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex, double eta, double alpha)
{
    for (unsigned c = 0; c < numOutputs; ++c)
    {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
    m_myIndex = myIndex;
    setLearningRate(eta);
    setMomentum(alpha);
};

// *** class Net ***

class Net
{
  public:
    Net(const vector<unsigned> &topology, const double eta, const double alpha);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals);

  private:
    vector<Layer> m_layers; // m_layer[layerNum][neuronNum]
    double m_error;
    double m_recentAverageError;
    double m_recentAverageSmoothingFactor;
};

Net::Net(const vector<unsigned> &topology, const double eta, const double alpha)
{
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum)
    {
        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
        // We have made a new Layer, now fill it with neurons, and
        // add a bias neuron to the layer:
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
        {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum, eta, alpha));
            //cout << "Made a Neuron!" << endl;
        }
        // Force the bias node's output value to 1.0. It's the last neuron created above
        m_layers.back().back().setOutputVal(1.0);

    }
}

void Net::getResults(vector<double> &resultVals) {
    // gets the output from last layer
    resultVals.clear();

    for (unsigned n = 0; n < m_layers.back().size() - 1; ++n)
    {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}

void Net::backProp(const vector<double> &targetVals)
{
    // Calculate overall net error (Root mean square error (RMS) of output neuron errors)
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
    {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size() - 1; // get average error squared
    m_error = sqrt(m_error);           // RMS

    // Implement a recent average measurement:
    m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1.0);

    // Calculate output layer gradients
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
    {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    // Calculate gradients on hidden layers
    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum)
    {   
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum+1];
        for (unsigned n = 0; n< hiddenLayer.size(); ++n){
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    // For all layers from outputs to first hidden layer
    // update connection weights
    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum)
    {   
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum-1];
        for (unsigned n = 0; n<layer.size(); ++n){
            layer[n].updateInputWeights(prevLayer);
        }
        
    }
};

void Net::feedForward(const vector<double> &inputVals)
{
    assert(inputVals.size() == m_layers[0].size() - 1);
    // Assign (latch) the input values into the input neurons
    for (unsigned i = 0; i < inputVals.size(); ++i)
    {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }
    for (unsigned layerNum = 1; layerNum <= m_layers.size()-1; ++layerNum)
    {
        Layer &prevLayer = m_layers[layerNum - 1];

        for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n)
        {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
            
    }

            
}


double stepFunction(double x){
    if(x>0.9){
        return 1.0;
    }
    if(x<0.1){
        return 0.0;
    }
    return x;
}
void loadTraining(const char *filename, vector<vector<double> > &input, vector<vector<double> > &output){
    int trainingSize = 946;
    input.resize(trainingSize);
    output.resize(trainingSize);

    ifstream file(filename);
    if(file){
        string line;
        int n;

        for (int i=0 ; i<trainingSize ; ++i){ // load 946 examples
            for (int h=0 ; h<32 ; ++h){ // 'images' are 32*32 pixels
                getline(file, line);
                for (int w=0 ; w<32 ; ++w){
                    input[i].push_back(atoi(line.substr(w,1).c_str())); // atoi = converts string to integer
                }
            }
            output[i].resize(10); // output is a vector of size 10 for digist 0-9
            getline(file, line); // every 33rd line gives the result of the digit (see training file)
            n = atoi(line.substr(0,1).c_str()); // get the number that is represented by the array
            output[i][n] = 1; // set index that represent the number to 1, other are automatically 0 because of the resize()
        }
    }
    file.close();
}

int main()
{

    // get training data
    // learning digit recognition (0,1,2,3,4,5,6,7,8,9)
    std::vector<std::vector<double> > inputVector, outputVector;
    loadTraining("data/handwritten-numbers", inputVector, outputVector); // load data from file called "training"

    // 32*32=1024 input neurons (images are 32*32 pixels)
    // 15 hidden neurons (experimental)
    // 10 output neurons (for every input, the output is a vector of size 10, full of zeros and a 1 at the index of the number represented)
    // 0.7 learning rate (experimental)
    vector<unsigned> topology;
    topology.push_back(1024);
    topology.push_back(15);
    topology.push_back(10);
    Net myNet(topology, 0.7, 0.5);

    int length = inputVector.size()-10; // skip the last 10 examples to test the program at the end
    // train 30 iterations
    for (unsigned i=0 ; i<30 ; ++i){
        cout << "Epoch " << i+1 << "/30" << endl;
        for (unsigned j=0 ; j<length ; ++j){ 

            myNet.feedForward(inputVector[j]); //feed in j-th example
            myNet.backProp(outputVector[j]); //back prop with output
            //cout << "\t error = " << error;
        }
        //cout << endl;
    }
    // test network
    cout << endl << "expected output : actual output" << endl;

    vector<double> resultVals;
    int correct = 0;
    for (unsigned i=inputVector.size()-10 ; i<inputVector.size() ; ++i){ // testing on last 10 examples

        myNet.feedForward(inputVector[i]); // feed input through network
        myNet.getResults(resultVals);
        // since the sigmoid function never reaches 0.0 nor 1.0
        // it can be a good idea to consider values greater than 0.9 as 1.0 and values smaller than 0.1 as 0.0
        // hence the step function.
        for (unsigned j=0 ; j<10 ; ++j){ 
            cout << outputVector[i][j] << " ";
        }
        for (unsigned j=0 ; j<10 ; ++j){ 
            cout << " " << (stepFunction(resultVals[j]));
            if(stepFunction(resultVals[j]) == outputVector[i][j] && outputVector[i][j] == 1) {
                cout << "*";
                correct++;
            }
        }
        cout  << endl;
    }
    cout << "Total of " << correct << "/10 correct";


}