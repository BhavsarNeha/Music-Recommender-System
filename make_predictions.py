import pandas as pd
import joblib

# Load preprocessed test data
test = pd.read_csv('test_preprocessed.csv')

# Load the trained model
model = joblib.load('music_recommender_model.pkl')

# Generate predictions
test_predictions = model.predict(test.drop(columns=['id']))

# Prepare submission file
submission = pd.DataFrame({'id': test['id'], 'target': test_predictions})
submission.to_csv('submission.csv', index=False)
