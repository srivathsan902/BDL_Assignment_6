# Digit Classification Using FastAPI

## Overview
We have developed models for MNIST digit classification locally and saved them in our local machines. To enable users to access this model through api calls, we use FAST api module available in python. It receives images from an end user and serves to the server, processes the image, gives it to the model and sends back the prediction to the client.

## Installation
All the dependencies involved can be installed using `pip install -r requirements.txt`. Make sure to create a virtual environment to ensure there are no clashes with other projects. 

## Walkthrough Of Code
In `task_1`, a user is expected to upload only a 28x28 image and the model returns an error if otherwise. 
In `task_2` user can upload image of any size, it gets resized to 28x28 and then predicts the digit in it. Other than that, both task 1 and task 2 takes in `model_path` as a command line argument and uses it to load any model locally to itself. Then the model is used to predict images when triggerred.

To run `task_1`: put `python task_1.py MODEL` in command line.

To run `task_2`: put `python task_2.py MODEL` in command line.


## Helpful Links
* [Official Documentation for FastAPI](https://fastapi.tiangolo.com/)
* [FastAPI - Swagger UI](http://127.0.0.1:8000/docs) used in the code

