# Machine-Learning---Project-recommendation-

run `pip install -r requirements.txt --user`

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

# Detailed explanation : 


### Import Statements and Data Loading

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tabulate import tabulate
import numpy as np
import pickle

# Read data for old projects from Excel file
df = pd.read_excel('Classeur2.xlsx')
```

- **Explanation**: Imports all necessary libraries and modules at the beginning for clarity. It also loads data (`Classeur2.xlsx`) into a DataFrame (`df`) containing information about old projects.

### Define and Prepare Data for New Projects

```python
# Define the data for new projects with varying column entries
new_projects_data = {
    'Budget': [1500],
    'Duration': [3],
    'Team_Size': [4],
    'Client_Feedback': ['good'],
    'Functional_Requirements': ['tribunal verification'],
    'Technologies_Used': ['Angular, Springboot, mongo']
}

# Create the DataFrame for new projects
new_projects = pd.DataFrame(new_projects_data)
```

- **Explanation**: Sets up a DataFrame (`new_projects`) to simulate new project data with specific features such as budget, duration, team size, client feedback, functional requirements, and technologies used.

### Text Processing and Vectorization

```python
# Concatenate text features for both old and new projects
all_text = df['Functional_Requirements'] + ", " + df['Technologies_Used']
new_text = new_projects['Functional_Requirements'] + ", " + new_projects['Technologies_Used']

# Initialize TF-IDF vectorizer with English stop words removal
vectorizer = TfidfVectorizer(stop_words='english')

# Compute TF-IDF matrices
sparse_matrix_past = vectorizer.fit_transform(all_text)
sparse_matrix_new = vectorizer.transform(new_text)
```

- **Explanation**: Prepares text data by concatenating relevant columns (`Functional_Requirements` and `Technologies_Used`). It then initializes a TF-IDF vectorizer with English stop words removal and computes TF-IDF matrices (`sparse_matrix_past` for old projects and `sparse_matrix_new` for new projects).

### Model Training and Evaluation

```python
# Split data into training and testing sets for evaluation
X_train, X_test, y_train, y_test = train_test_split(sparse_matrix_past, df['Similarity_Score'], test_size=0.2, random_state=42)

# Initialize models for comparison
svm = SVR(kernel='linear')
rf = RandomForestRegressor(random_state=42)
gbr = GradientBoostingRegressor(random_state=42)

# Set up parameter grids for hyperparameter tuning
param_grid_svm = {
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1, 1]
}

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20]
}

param_grid_gbr = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Perform grid search with cross-validation to find best models
grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=5, scoring='r2')
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='r2')
grid_search_gbr = GridSearchCV(gbr, param_grid_gbr, cv=5, scoring='r2')

# Train models
grid_search_svm.fit(X_train, y_train)
grid_search_rf.fit(X_train, y_train)
grid_search_gbr.fit(X_train, y_train)

# Evaluate models using cross-validation scores
cv_scores_svm = cross_val_score(grid_search_svm.best_estimator_, X_train, y_train, cv=5)
cv_scores_rf = cross_val_score(grid_search_rf.best_estimator_, X_train, y_train, cv=5)
cv_scores_gbr = cross_val_score(grid_search_gbr.best_estimator_, X_train, y_train, cv=5)

# Print cross-validation scores for comparison
print("Cross-Validation Scores:")
print("SVM:", cv_scores_svm)
print("Random Forest:", cv_scores_rf)
print("Gradient Boosting:", cv_scores_gbr)
```

- **Explanation**: Splits data into training and testing sets for model evaluation. Initializes and tunes Support Vector Regression (SVR), Random Forest, and Gradient Boosting models using grid search with cross-validation. Evaluates models and prints their cross-validation scores for comparison.

### Model Selection and Prediction

```python
# Predict similarity scores for new projects using the best model
best_model_name = max(
    ("SVM", r2_score(y_test, grid_search_svm.best_estimator_.predict(X_test))),
    ("Random Forest", r2_score(y_test, grid_search_rf.best_estimator_.predict(X_test))),
    ("Gradient Boosting", r2_score(y_test, grid_search_gbr.best_estimator_.predict(X_test))),
    key=lambda x: x[1]
)[0]

print(f"The best model is {best_model_name}")

# Select and save the best model for recommendations
if best_model_name == "SVM":
    best_model = grid_search_svm.best_estimator_
elif best_model_name == "Random Forest":
    best_model = grid_search_rf.best_estimator_
else:
    best_model = grid_search_gbr.best_estimator_

pickle.dump(best_model, open("best_model_recommendation.sav", "wb"))

# Predict similarity scores for new projects using the best model
similarity_scores_best = best_model.predict(sparse_matrix_new)
new_projects['Predicted_Similarity_Score'] = similarity_scores_best

# Add similarity scores to the original DataFrame for sorting
df['Predicted_Similarity_Score'] = best_model.predict(sparse_matrix_past)

# Exclude the new project itself from the recommendation list
recommendations = df[df.index != df.index[-1]]

# Sort recommendations by predicted similarity score
recommendations_sorted = recommendations.sort_values(by='Predicted_Similarity_Score', ascending=False)

# Print top recommendations for new projects
print("Top Recommendations for New Projects:")
print(tabulate(recommendations_sorted.head(), headers='keys', showindex=False))
```

- **Explanation**: Determines the best-performing model based on R-squared score and selects it for making recommendations. Saves the best model using pickle for future use. Predicts similarity scores for new projects using the best model and sorts recommendations based on predicted similarity scores. Prints top recommendations for new projects.

### Additional Notes:

- Ensure proper commenting and use of markdown cells in Jupyter notebooks to explain each step clearly.
- Use consistent variable naming and logical organization of code blocks.
- Save necessary artifacts like the TF-IDF vectorizer (`vectorizer`) and the best model (`best_model_recommendation.sav`) for deployment or further analysis.

By organizing the notebook in this structured manner, with clear explanations and proper formatting, it becomes easier to follow and understand the entire process of project recommendation and model evaluation.
