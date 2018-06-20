
import numpy as np

from util.loss_functions import CrossEntropyError, BinaryCrossEntropyError, MeanSquaredError
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier

from sklearn.metrics import accuracy_score

import sys

class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, 
                netStruct = [128, 10],
                actFunc = ['sigmoid', 'softmax'],
                inputWeights=None,
                outputTask='classification',
                dropout = False, dropoutRateHidden = 0.5, dropoutRateInput = 0.2,
                loss='bce', learningRate=0.01, epochs=50):

        """
        A MNIST recognizer based on multi-layer perceptron algorithm

        Parameters
        ----------
        train : list
        valid : list
        test : list
        learningRate : float
        epochs : positive int

        Attributes
        ----------
        trainingSet : list
        validationSet : list
        testSet : list
        learningRate : float
        epochs : positive int
        performances: array of floats
        """

        self.learningRate = learningRate
        self.epochs = epochs
        self.outputTask = outputTask  # Either classification or regression

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        ## Dropout constants
        self.dropout = dropout
        self.dropoutRateHidden = dropoutRateHidden
        self.dropoutRateInput = dropoutRateInput
        
        if loss == 'bce':
            self.loss = BinaryCrossEntropyError()
        elif loss == 'crossentropy':
            self.loss = CrossEntropyError()
        elif loss == 'sse':
            self.loss = SumSquaredError()
        elif loss == 'mse':
            self.loss = MeanSquaredError()
        elif loss == 'different':
            self.loss = DifferentError()
        elif loss == 'absolute':
            self.loss = AbsoluteError()
        else:
            raise ValueError('There is no predefined loss function ' +
                             'named ')

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        self.layers = []
        self._costructNetwork(netStruct, actFunc)

        self.nIn = netStruct[0]
        self.nOut = netStruct[-1]


    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self._get_layer(0)

    def _get_output_layer(self):
        return self._get_layer(-1)

    def _feed_forward(self, inp, dropout = False):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer

        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer

        ## LogistLayer class already saves the last activations, 
        ## so I wont save it here to minimize memory usage
        """

        if dropout:
            output = inp * (np.random.random(inp.shape) > self.dropoutRateInput) 
        else:
            output = inp


        for layer in self.layers:

            layer.forward(output)
            
            if not layer.isClassifierLayer:

                if dropout:
                    layer.outp = layer.outp * (np.random.random(layer.outp.shape) > self.dropoutRateHidden)

                output = np.insert(layer.outp, 0, 1)
            else:
                output = layer.outp

        return output
        
    def _compute_error(self, target):
        """
        Compute the total error of the network (error terms from the output layer)

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        return self.loss.calculateError(target, self.layers[-1].outp)

    def _update_weights(self, learningRate):
        """
        Update the weights of the layers by propagating back the error
        """
        for layer in self.layers:
            layer.updateWeights(learningRate)

        
    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        for epoch in range(self.epochs):
            totalError = 0

            for (x, label) in zip(self.trainingSet.input, self.trainingSet.label):

                output = self._feed_forward(x, self.dropout)

                y = np.zeros(self.nOut)
                y[label] = 1

                totalError += self._compute_error(y)


                next_derivatives = self.loss.calculateDerivative(y, output)
                next_weights = np.ones(next_derivatives.shape)

                for layer in reversed(self.layers):
                    next_derivatives = layer.computeDerivative(next_derivatives, next_weights)
                    next_weights = layer.weights[0:-1].T

                self._update_weights(self.learningRate)


            if verbose:
                validationAcc = accuracy_score(self.validationSet.label, self.evaluate(self.validationSet.input))
                print('Epoch {}: TotalError = {}, Validation Accuracy = {}'.format(epoch, totalError, validationAcc))
                self.performances.append(validationAcc)
     

    def classify(self, test_instance):
        return np.argmax(self._feed_forward(test_instance))
        

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def __del__(self):
        # Remove the bias from input data
        self.trainingSet.input = np.delete(self.trainingSet.input, 0, axis=1)
        self.validationSet.input = np.delete(self.validationSet.input, 0,
                                              axis=1)
        self.testSet.input = np.delete(self.testSet.input, 0, axis=1)

    
    def _costructNetwork(self, netStruct, activationFunctions):
        
        prevSize = self.trainingSet.input.shape[1] - 1
        for (size, func) in zip(netStruct, activationFunctions):
            self.layers.append(LogisticLayer(prevSize, size, None, func, False))
            prevSize = size

        self.layers[-1].isClassifierLayer = True
