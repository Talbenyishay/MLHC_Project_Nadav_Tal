This repository contains the code for the final project in the course of machine learning for healthcare.
The project deals with the prediction of mortality, prolonged stays and readmission on the dataset of MIMIC3 using data from the first 42 hours of admissions.

The repository contains several python files:

unseen_data_evaluation.py - the main pipeline code, using it you can run our models and test new patients.

preprocessing.py - helper module for arranging the dataset with all the features for running the model.

diagnosis_processing.py - helper module for processing the diagnosis given at the beginning of an admission into features for the model.

models.py - helper module that defines all the models implemented in the project.


The Saved_models directory contains the 3 trained models for each one of the targets.

The Expected_Columns_For_Models directory contains 3 list of required features for each target in order to test using our pipeline.

The important_files directory contains files related to the processing of the diagnosis in the beginning of the admission.