# Movie Rating Prediction 🎬

## Project Overview
This project aims to predict the average rating of movies based on features such as genre, director popularity, budget, and cast rating. The problem is formulated as a regression task where the goal is to estimate the continuous average rating value for a given movie.

## Dataset
The dataset contains the following columns:
- **genre**: The category of the movie (e.g., Action, Comedy, Romance, Thriller, Drama).
- **director_popularity**: A numerical value representing the popularity score of the movie’s director.
- **budget**: The movie's budget in USD.
- **cast_rating**: A score representing the average rating of the cast members.
- **average_rating**: The target variable, representing the movie's average rating.

## Objective
- Build a regression model that accurately predicts the average rating of movies based on the given features.
- Evaluate the model performance using metrics such as R² score, Mean Squared Error (MSE), and Mean Absolute Error (MAE).

## Approach
- Perform basic exploratory data analysis (EDA) and visualization to understand feature relationships.
- Preprocess data, including encoding categorical variables.
- Train a Random Forest Regressor model.
- Validate and test the model on unseen data.
- Tune model hyperparameters to optimize performance.

## Usage
1. Load the dataset (`movie_rating_data.csv`).
2. Run the Python code (`movie_rating_prediction.py`) to train and evaluate the model.
3. Use the trained model to predict average ratings for new movie data.

## Libraries Used
- pandas
- numpy
- scikit-learn
- seaborn (for visualization)
- matplotlib (for visualization)

## Potential Improvements
- Include more features like release year, movie runtime, and critic reviews.
- Apply advanced regression models or ensemble methods.
- Perform hyperparameter tuning with grid search or random search.
- Use cross-validation for more robust evaluation.

## Author
Md Hidayat Ali

---


