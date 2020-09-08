import util

PRINT = True
one = 1
zero = 0

class PerceptronClassifier:
    """
    Perceptron class definition
    """

    # Initialize constructor
    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {}
        for label in legalLabels:
            self.weights[label] = util.Counter()  # this is the data-structure you should use

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels);
        self.weights = weights

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        The training loop for the perceptron passes through the training data several
        times and updates the weight vector for each label based on classification errors.
        """
        self.features = trainingData[0].keys()
        number_of_errors = zero

        for iteration in range(self.max_iterations):
            if iteration > zero:
                print "Number of Errors: ", number_of_errors
                # print the number of errors after each iteration
                number_of_errors = zero # reset for next iteration
            print "Starting iteration ", iteration, "..."

            for i in range(len(trainingData)):
                # find the label that best represents each image

                top_score = -1*one
                best_label = None
                current_datum = trainingData[i]  # current image

                for label in self.legalLabels:
                    # Calculate the prediction on the current image for every labels weights (0-9)
                    result = current_datum * self.weights[label]
                    if result > top_score:
                        # save the result with the largest value - i.e most likely to be correct
                        top_score = result
                        best_label = label

                actual_label = trainingLabels[i]
                if best_label != actual_label:  # prediction is incorrect
                    number_of_errors += 1
                    # update weights
                    self.weights[actual_label] = self.weights[actual_label] + current_datum  # under predicted
                    self.weights[best_label] = self.weights[best_label] - current_datum  # over predicted

    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.
        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses

    def findHighWeightFeatures(self, label):
        """
        Returns a list of the 100 features with the greatest weight for some label
        """

        return self.weights[label].sortedKeys()[:100]
