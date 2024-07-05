# Machine-Learning---Project-recommendation-

### Description
Includes three different models: Support Vector Machine (SVM), Random Forest, and Gradient Boosting Regressor. It trains each model, performs hyperparameter tuning using grid search, and evaluates them using cross-validation. Here's a summary of how each model is included and evaluated:

#### Support Vector Machine (SVM):

Defined with SVR(kernel='linear')
Hyperparameter tuning using GridSearchCV with parameters for C and epsilon
Evaluated using cross-validation and test set metrics

#### Random Forest:

Defined with RandomForestRegressor(random_state=42)
Hyperparameter tuning using GridSearchCV with parameters for n_estimators and max_depth
Evaluated using cross-validation and test set metrics

#### Gradient Boosting Regressor:

Defined with GradientBoostingRegressor(random_state=42)
Hyperparameter tuning using GridSearchCV with parameters for n_estimators, learning_rate, and max_depth
Evaluated using cross-validation and test set metrics

The code then selects the best model based on the R-squared score and uses it for making project recommendations.
