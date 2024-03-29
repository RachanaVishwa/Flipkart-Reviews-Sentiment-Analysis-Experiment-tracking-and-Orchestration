# # Machine Learning Workflow Orchestration

# Orchestration refers to the coordination and management of various tasks, resources, and processes involved in the end-to-end machine learning lifecycle. This includes:
# 
# 1. Data Preparation and Management
# 2. Model Training
# 3. Experimentation and Evaluaiton
# 4. Model Deployment
# 5. Monitor and Management
# 6. Automation of repetitive tasks

# ## Introducing Prefect
# Prefect is an open-source orchestration and observability platform that empowers developers to build and scale resilient code quickly, turning their Python scripts into resilient, recurring workflows.

# ## Why Prefect?
# 
# * Python based open source tool
# * Manage ML Pipelines
# * Schedule and Monitor the flow
# * Gives observability into failures
# * Native dask integration for scaling (Dask is used for parallel computing)

# # Refactoring the ML Workflow

from prefect import task, flow

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import joblib
from sklearn.naive_bayes import MultinomialNB
from joblib import Memory
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics

@task
def load_data(file_path):
    """
    Load data from a CSV file.
    """
    df = pd.read_csv(file_path)
    df.columns = [ col.strip().replace(' ', '_') for col in df.columns ]
    
    df['Ratings'] = pd.Series(np.select([(df['Ratings'] >= 4), (df['Ratings'] <= 3)], [1, 0]))
    
    # replace null values with nan and remove
    df.replace('',np.nan,inplace=True)
    df.dropna(inplace=True)
    
    return df


@task
def input_output(data, input, output):
    """
    Split features and target variables.
    """
    X = data[input]
    y = data[output]
    return X, y


@task
def split_train_test(X, y, test_size=0.25, random_state=42):
    """
    Split data into train and test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


@task
def train_model(X_train, y_train, **hyperparameters):
    """
    Training the machine learning model.
    """
    cachedir = '.cache'
    memory = Memory(location=cachedir, verbose=0)

    pipe = Pipeline([
        ('vectorization', CountVectorizer()),
        ('classifier', MultinomialNB())
    ], memory=memory)

    clf = GridSearchCV(estimator=pipe,
                       param_grid=hyperparameters,
                       scoring='f1',
                       cv=4,
                       return_train_score=True,
                       verbose=1
                      )

    clf.fit(X_train, y_train)
    return clf

@task
def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluating the model.
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_score = metrics.f1_score(y_train, y_train_pred)
    test_score = metrics.f1_score(y_test, y_test_pred)
    
    return train_score, test_score


#X = data_df['Review_text']

#y = pd.Series(np.select([(data_df['Ratings'] >= 4), (data_df['Ratings'] <= 3)], [1, 0]))

# Workflow

@flow(name="Multinomial Naive Bayes Training")
def workflow():
    data_path = "/Users/rachusarang/Downloads/ILR/reviews_data_dump/reviews_badminton/data.csv"
    INPUT = 'Review_text'
    OUTPUT = 'Ratings'
    HYPERPARAMETERS = {
        'vectorization': [CountVectorizer()],
        'vectorization__max_features' : [5000], 
        'classifier__alpha' : [1]
    }
    
    # Load data
    df = load_data(data_path)

    # Identify Inputs and Output
    X, y = input_output(df, INPUT, OUTPUT)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Build a model
    model = train_model(X_train, y_train, **HYPERPARAMETERS)
    
    # Evaluation
    train_score, test_score = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    print("Train Score:", train_score)
    print("Test Score:", test_score)
    

if __name__ == "__main__":
    workflow.serve(
        name="Badminton-Reviews-ML-Deployment",
        cron="30 * 15 * *" ## Runs at 00:30:00 on day-of-month 15
    )