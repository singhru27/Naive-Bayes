import numpy as np

class NaiveBayes(object):
    """ Bernoulli Naive Bayes model
    @attrs:
        n_classes: the number of classes
    """

    def __init__(self, n_classes):
        """ Initializes a NaiveBayes model with n_classes. """
        self.n_classes = n_classes
        self.priors = None
        self.attribute_distributions = None

    def train(self, X_train, y_train):
        """ Trains the model, using maximum likelihood estimation.
        @params:
            X_train: a n_examples x n_attributes numpy array
            y_train: a n_examples numpy array
        @return:
            a tuple consisting of:
                1) a 2D numpy array of the attribute distributions
                2) a 1D numpy array of the priors distribution
        """

        # Creating the priors array
        self.priors = np.zeros(self.n_classes)
        num_examples = np.size(y_train)

        ## Creating a count for each class
        class_counts = np.zeros(self.n_classes)
        for i in y_train:
            class_counts[i] = class_counts[i] + 1

        # Applying Laplace smoothing to the priors
        for i in range (0, self.n_classes):
            class_counts[i] = class_counts[i] + 1
            num_examples = num_examples + 1

        # Creating the priors array
        for i in range (0, self.n_classes):
            self.priors[i] = class_counts[i]/num_examples

        ## Creating the attribute distributions array
        num_attributes = X_train.shape[1]
        self.attribute_distributions = np.zeros((num_attributes, self.n_classes))

        ## Counting the number of elements with each respective attributes
        for i in range (0, X_train.shape[0]):
            for j in range (0, num_attributes):

                # j is the attribute number, y_train[i] is the classification of the current
                # example. X_train[i][j] = 1 if that current attribute is set to 1 for ths particular example
                self.attribute_distributions[j][y_train[i]] = self.attribute_distributions[j][y_train[i]] + X_train[i][j]

        ## Applying Laplace smoothing to the attribute distribution, and calculating the probabilities
        for i in range (0, num_attributes):
            for j in range (0, self.n_classes):
                self.attribute_distributions[i][j] = self.attribute_distributions[i][j] + 1
                self.attribute_distributions[i][j] = (self.attribute_distributions
                [i][j]/(np.count_nonzero(y_train == j) + 2))


        return self.attribute_distributions, self.priors




    def predict(self, inputs):
        """ Outputs a predicted label for each input in inputs.

        @params:
            inputs: a NumPy array containing inputs
        @return:
            a numpy array of predictions
        """
        # TODO
        predictions_array = np.zeros(inputs.shape[0])

        ## Looping through each example
        for i in range (0, inputs.shape[0]):

            sum_probabilities = np.copy (self.priors)

            ## Converting to log space
            sum_probabilities = np.log(sum_probabilities)

            ## Looping through each class
            for j in range (0, sum_probabilities.size):

                ## Looping through each attribute
                for k in range (0, inputs.shape[1]):

                    ## Flipping the probabilities if the input is 0
                    if (inputs[i][k] == 0):
                        sum_probabilities[j] = sum_probabilities[j] + np.log((1 - self.attribute_distributions[k][j]))

                    ## Otherwise adding by the log normal sum_probabilities
                    else:
                        sum_probabilities[j] = sum_probabilities[j] + np.log(self.attribute_distributions[k][j])

            sum_probabilities = np.exp(sum_probabilities)
            ## Setting the prediction predictions_array
            predictions_array[i] = np.argmax(sum_probabilities)

        return predictions_array



    def accuracy(self, X_test, y_test):
        """ Outputs the accuracy of the trained model on a given dataset (data).

        @params:
            X_test: 2D numpy array of examples
            y_test: numpy array of labels
        @return:
            a float number indicating accuracy (between 0 and 1)
        """

        # TODO
        predictions_array = self.predict(X_test)
        count_correct = 0
        for i in range (y_test.size):
            if y_test[i] == predictions_array[i]:
                count_correct = count_correct + 1

        return count_correct/y_test.size

    def print_fairness(self, X_test, y_test, x_sens):
        """
        ***DO NOT CHANGE what we have implemented here.***

        Prints measures of the trained model's fairness on a given dataset (data).

        For all of these measures, x_sens == 1 corresponds to the "privileged"
        class, and x_sens == 1 corresponds to the "disadvantaged" class. Remember that
        y == 1 corresponds to "good" credit.

        @params:
            X_test: 2D numpy array of examples
            y_test: numpy array of labels
            x_sens: numpy array of sensitive attribute values
        @return:

        """
        predictions = self.predict(X_test)

        # Disparate Impact (80% rule): A measure based on base rates: one of
        # two tests used in legal literature. All unprivileged lasses are
        # grouped together as values of 0 and all privileged classes are given
        # the class 1. . Given data set D = (X,Y, C), with protected
        # attribute X (e.g., race, sex, religion, etc.), remaining attributes Y,
        # and binary class to be predicted C (e.g., “will hire”), we will say
        # that D has disparate impact if:
        # P[Y^ = 1 | S != 1] / P[Y^ = 1 | S = 1] <= (t = 0.8).
        # Note that this 80% rule is based on US legal precedent; mathematically,
        # perfect "equality" would mean

        di = np.mean(predictions[np.where(x_sens==0)])/np.mean(predictions[np.where(x_sens==1)])
        print("Disparate impact: " + str(di))

        # Group-conditioned error rates! False positives/negatives conditioned on group

        pred_priv = predictions[np.where(x_sens==1)]
        pred_unpr = predictions[np.where(x_sens==0)]
        y_priv = y_test[np.where(x_sens==1)]
        y_unpr = y_test[np.where(x_sens==0)]

        # s-TPR (true positive rate) = P[Y^=1|Y=1,S=s]
        priv_tpr = np.sum(np.logical_and(pred_priv == 1, y_priv == 1))/np.sum(y_priv)
        unpr_tpr = np.sum(np.logical_and(pred_unpr == 1, y_unpr == 1))/np.sum(y_unpr)

        # s-TNR (true negative rate) = P[Y^=0|Y=0,S=s]
        priv_tnr = np.sum(np.logical_and(pred_priv == 0, y_priv == 0))/(len(y_priv) - np.sum(y_priv))
        unpr_tnr = np.sum(np.logical_and(pred_unpr == 0, y_unpr == 0))/(len(y_unpr) - np.sum(y_unpr))

        # s-FPR (false positive rate) = P[Y^=1|Y=0,S=s]
        priv_fpr = 1 - priv_tnr
        unpr_fpr = 1 - unpr_tnr

        # s-FNR (false negative rate) = P[Y^=0|Y=1,S=s]
        priv_fnr = 1 - priv_tpr
        unpr_fnr = 1 - unpr_tpr

        print("FPR (priv, unpriv): " + str(priv_fpr) + ", " + str(unpr_fpr))
        print("FNR (priv, unpriv): " + str(priv_fnr) + ", " + str(unpr_fnr))


        # #### ADDITIONAL MEASURES IF YOU'RE CURIOUS #####

        # Calders and Verwer (CV) : Similar comparison as disparate impact, but
        # considers difference instead of ratio. Historically, this measure is
        # used in the UK to evalutate for gender discrimination. Uses a similar
        # binary grouping strategy. Requiring CV = 1 is also called demographic
        # parity.

        cv = 1 - (np.mean(predictions[np.where(x_sens==1)]) - np.mean(predictions[np.where(x_sens==0)]))

        # Group Conditioned Accuracy: s-Accuracy = P[Y^=y|Y=y,S=s]

        priv_accuracy = np.mean(predictions[np.where(x_sens==1)] == y_test[np.where(x_sens==1)])
        unpriv_accuracy = np.mean(predictions[np.where(x_sens==0)] == y_test[np.where(x_sens==0)])

        return predictions
