from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import BaseCrossValidator


class CustomCV(BaseCrossValidator):
    def __init__(self, train_index_sets, valid_index_sets):
        self.train_index_sets = train_index_sets
        self.valid_index_sets = valid_index_sets

    def split(self, X, y=None, groups=None):
        for i in range(len(self.train_index_sets)):
            yield self.train_index_sets[i], self.valid_index_sets[i]
            
    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.train_index_sets)  # Return the number of splits
    
 
def perform_GridSearchCV(classifier, grid_params, X_train, y_train,
                         train_index_sets, vaildation_index_sets):
    
    # Create GridSearchCV object and fit to the training data
    custom_splitter = CustomCV(train_index_sets, vaildation_index_sets)

    grid_search = GridSearchCV(classifier, param_grid=grid_params, scoring='balanced_accuracy', 
                                  cv=custom_splitter, verbose=1)

    grid_search.fit(X_train, y_train)

    # Get the best parameters and the best score
    best_classifier = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    return {
        'best_classifier': best_classifier,
        'best_params': best_params,
        'best_score': best_score
    }