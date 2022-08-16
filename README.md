**Stage 1 - Model training**
_______________________

For model training file 'Training.py' was used.
Trained pipeline is stored in 'Deploy\src\model_pipeline.pkl'.

Searching Pipeline:
1. Custom function to manual feature creation features_creation (line 29)
2. Feature extraction options [TfidfVectorizer\HashingVectorizer] (line 102)
3. Regression model options [SGDRegressor, XGBRegressor, Lasso, SVR] (line 114)

Hyperparameters for both feature vectorization and model tinning were find with GridSearch (line 143).
Test error (RMSE - 0.747) was huge in comparison with results in Kaggle dashboard (winner RMSE ~0.460),
however, it might be acceptable as the baseline model.


**Stage 2 - API service**
_______________________

Model deployment was created with python framework as microservice via flask.
Deployment script - 'Deploy\main.py'.
Capability was successfully tested with the script 'Deploy\Check_deploy.py'.







