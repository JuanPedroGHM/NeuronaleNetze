#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from model.mlp import MultilayerPerceptron

from report.evaluator import Evaluator
from report.performance_plot import PerformancePlot

import numpy as np
import matplotlib.pyplot as plt


def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
                                                    oneHot=False)

    data.trainingSet.input = np.insert(data.trainingSet.input, 0, 1,
                                        axis=1)
    data.validationSet.input = np.insert(data.validationSet.input, 0, 1,
                                          axis=1)
    data.testSet.input = np.insert(data.testSet.input, 0, 1, axis=1)

    myStupidClassifier = StupidRecognizer(data.trainingSet,
                                          data.validationSet,
                                          data.testSet)
    
    # myPerceptronClassifier = Perceptron(data.trainingSet,
    #                                     data.validationSet,
    #                                     data.testSet,
    #                                     learningRate=0.005,
    #                                     epochs=30)
                                        
    # myLRClassifier = LogisticRegression(data.trainingSet,
    #                                     data.validationSet,
    #                                     data.testSet,
    #                                     learningRate=0.005,
    #                                     epochs=30)

    MLPClassifier = MultilayerPerceptron(data.trainingSet, 
                        data.validationSet, 
                        data.testSet,
                        netStruct = [800, 100, 10], 
                        actFunc = ['relu', 'relu', 'softmax'], 
                        dropout = True,
                        loss = 'crossentropy',
                        learningRate = 0.001,
                        epochs = 300)

    
    # Report the result #
    print("=========================")
    evaluator = Evaluator()                                        

    # Train the classifiers
    print("=========================")
    print("Training..")

    # print("\nStupid Classifier has been training..")
    # myStupidClassifier.train()
    # print("Done..")
    #
    # print("\nPerceptron has been training..")
    # myPerceptronClassifier.train()
    # print("Done..")
    
    # print("\nLogistic Regression has been training..")
    # myLRClassifier.train()
    # print("Done..")

    print("\nMLP has been training..")
    MLPClassifier.train()
    print("Done..")

    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    # stupidPred = myStupidClassifier.evaluate()
    # perceptronPred = myPerceptronClassifier.evaluate()
    # lrPred = myLRClassifier.evaluate()
    mlpPred = MLPClassifier.evaluate()
    
    # Report the result
    print("=========================")
    evaluator = Evaluator()

    # print("Result of the stupid recognizer:")
    # #evaluator.printComparison(data.testSet, stupidPred)
    # evaluator.printAccuracy(data.testSet, stupidPred)
    #
    # print("\nResult of the Perceptron recognizer:")
    # #evaluator.printComparison(data.testSet, perceptronPred)
    # evaluator.printAccuracy(data.testSet, perceptronPred)
    
    # print("\nResult of the Logistic Regression recognizer:")
    # #evaluator.printComparison(data.testSet, lrPred)    
    # evaluator.printAccuracy(data.testSet, lrPred)

    print("\nResult of the MLP recognizer:")
    # evaluator.printComparison(data.testSet, lrPred)    
    evaluator.printAccuracy(data.testSet, mlpPred)

    # Draw
    # plot = PerformancePlot("MLP validation")
    # plot.draw_performance_epoch(MLPClassifier.performances,
    #                             MLPClassifier.epochs)

    plt.plot(range(MLPClassifier.epochs), MLPClassifier.performances, 'r--')
    plt.show()
    
    
if __name__ == '__main__':
    main()
