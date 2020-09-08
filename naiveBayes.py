import util
import classificationMethod
import math

one = 1
zero = 0

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
    """
    See the project description for the specifications of the Naive Bayes classifier.
    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """

    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 1  # this is the smoothing parameter, ** use it in your train method **
        self.automaticTuning = False  # Look at this flag to decide whether to choose k automatically ** use this in your train method **

    def setSmoothing(self, k):
        """
        This is used by the main method to change the smoothing parameter before training.
        Do not modify this method.
        """
        self.k = k

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Outside shell to call your method. Do not modify this method.
        """

        # might be useful in your code later...
        # this is a list of all features in the training set.
        self.features = list(set([f for datum in trainingData for f in datum.keys()]));

        if (self.automaticTuning):
            kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
        else:
            kgrid = [self.k]

        self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in kgrid to choose the smoothing parameter
        that gives the best accuracy on the held-out validationData.
        trainingData and validationData are lists of feature Counters.  The corresponding
        label lists contain the correct label for each datum.
        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """

        # counter is a dict where values don't have to be initialized

        # prior probability = how frequently does each label occur in the training data
        prior = util.Counter()
        # probability that a given feature has value one
        conditional_probability = util.Counter()
        # count the frequency of a feature for each label
        frequency = util.Counter()

        # calculate prior probability over training data
        for label in trainingLabels:
            prior[label] += one
        prior.normalize()

        for i in range(len(trainingData)):
            image = trainingData[i]
            label = trainingLabels[i]

            for location, value in image.items():
                frequency[(location, label)] += one
                if value == one:
                    # We can count either part of this probability and get the other by subtracting from 1
                    conditional_probability[location, label] += one

        highest_accuracy = zero

        for k in kgrid:
            k_prior = util.Counter()
            k_conditional = util.Counter()
            k_frequency = util.Counter()

            # fill in ^^^ with previously calculated values
            for key, value in prior.items():
                k_prior[key] = value
            for key, value in conditional_probability.items():
                k_conditional[key] = value
            for key, value in frequency.items():
                k_frequency[key] = value

            for label in self.legalLabels:
                for location in self.features:
                    k_frequency[(location, label)] += k
                    k_conditional[(location, label)] += k

            for x, count in k_conditional.items():
                # print "x: ", x, " count: ", count
                k_conditional[x] = float(count) / k_frequency[x]
                # print "new: ", k_conditional[x]

            self.prior = k_prior
            self.conditionalProb = k_conditional

            # calculate results
            predictions = self.classify(validationData)
            number_correct = zero

            for index in range(len(validationLabels)):
                if predictions[index] == validationLabels[index]:
                    number_correct += one

            percent_correct = (float(number_correct) / len(validationLabels)) * 100
            print "Accuracy for k: ", k, " = ", percent_correct

            if percent_correct > highest_accuracy:
                highest_accuracy = percent_correct
                self.k = (k_prior, k_conditional, k)

    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.
        You shouldn't modify this method.
        """
        guesses = []
        self.posteriors = []  # Log posteriors are stored for later data analysis (autograder).
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, image):
        """
        Returns the log-joint distribution over legal labels and the datum.
        """

        logJoint = util.Counter()

        # find the prediction with the highest percentage

        for label in self.legalLabels:
            logJoint[label] = math.log(self.prior[label])
            for location, value in image.items():
                # print self.conditionalProb[(location, value)]
                if value == zero:
                    x = one - self.conditionalProb[(location, label)]
                    logJoint[label] += math.log(x if x > zero else one)
                    # logJoint[label] += math.log(1 - self.conditionalProb[(location, label)])
                else:
                    logJoint[label] += math.log(self.conditionalProb[(location, label)])

        return logJoint

    def findHighOddsFeatures(self, label1, label2):
        """
        Returns the 100 best features for the odds ratio:
                P(feature=1 | label1)/P(feature=1 | label2)
        """
        featuresOdds = []
        util.raiseNotDefined()
        return featuresOdds