

"""function that return classifier based on its input"""
def get_classfier_type(classifers_type):
    if classifers_type == "logistic_regression":
        return LogisticRegressionUsingGD()
    else:
        return DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)

class Adaboost:
    """ AdaBoost enemble classifier from scratch """

    def __init__(self):
        self.weak_classifiers = None
        self.classifier_weights = None
        self.errors = None
        self.sample_weights = None

    def _check_X_y(self, X, y):
        """ Validate assumptions about format of input data"""
        assert set(y) == {-1, 1}, 'Response variable must be ±1'
        return X, y

    
    def fit(self, X: np.ndarray, y: np.ndarray, iters: int, classifers_type):
        """ Fit the model using training data """
        
        X, y = self._check_X_y(X, y)
        n = X.shape[0]
      
        
        
        # init numpy arrays
        self.sample_weights = np.zeros(shape=(iters, n))
        self.weak_classifiers = np.zeros(shape=iters, dtype=object)
        self.classifier_weights = np.zeros(shape=iters)
        self.errors = np.zeros(shape=iters)

        # initialize weights uniformly
        self.sample_weights[0] = np.ones(shape=n) / n
        
        for t in range(iters):
            # fit  weak learner
            curr_sample_weights = self.sample_weights[t]
            classifier = get_classfier_type(classifers_type)
            classifier = classifier.fit(X, y, curr_sample_weights)
            
            # calculate error and classifier weight from weak learner prediction
            classifier_pred = classifier.predict(X)
            err = curr_sample_weights[(classifier_pred !=y)].sum()# / n           
            classifier_weight = np.log((1 - err) / err) / 2
                        
            # update sample weights
            new_sample_weights = (
                curr_sample_weights * np.exp(-classifier_weight * y * classifier_pred)
                )
            
            new_sample_weights /= new_sample_weights.sum()
            
            
            # If not final iteration, update sample weights for t+1
            if t+1 < iters:
                self.sample_weights[t+1] = new_sample_weights

            # save results of iteration
            self.weak_classifiers[t] = classifier
            self.classifier_weights[t] = classifier_weight
            self.errors[t] = err

        return self
    
    
    def predict(self, X):
        """ Make predictions using already fitted model """
        
        classifier_preds = np.array([classifier.predict(X) for classifier in self.weak_classifiers])
        
        return np.sign(np.dot(self.classifier_weights, classifier_preds))
 