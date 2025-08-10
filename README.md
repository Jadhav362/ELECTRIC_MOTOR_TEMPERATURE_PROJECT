Electric Motor Temperature Prediction
This project focuses on predicting the stator winding temperature of a Permanent Magnet Synchronous Motor (PMSM) using sensor data. The goal is to build an accurate regression model that can help in monitoring motor health, preventing overheating, and enabling predictive maintenance.

This repository contains the complete Jupyter Notebook for data exploration, model training, and evaluation. The final model, an XGBoost Regressor, demonstrated high accuracy in predicting the target temperature.

📋 Table of Contents
Project Overview

Dataset

Key Features

Technologies & Libraries

Installation

Usage

Model Performance

License

📝 Project Overview
Monitoring the temperature of electric motors is crucial for ensuring their longevity and operational efficiency. Overheating can lead to insulation degradation and catastrophic failure. This project leverages machine learning to create a predictive model based on various real-time measurements from the motor.

The process involves:

Exploratory Data Analysis (EDA) to understand feature distributions and correlations.

Data Preprocessing using a standard scaler to normalize the features.

Training and Evaluating Multiple Models, including Linear Regression, Ridge, Lasso, Random Forest, and XGBoost.

Selecting the Best Model based on performance metrics like R² Score and Root Mean Squared Error (RMSE).

The final model can be used to estimate the stator winding temperature from a new set of sensor readings.

🗂️ Dataset
The project uses the "Electric Motor Temperature" dataset, which contains measurements from a Permanent Magnet Synchronous Motor (PMSM) under various operating conditions.

The dataset includes 11 input features and 1 target variable:

Input Features:

ambient: Ambient temperature (°C)

coolant: Coolant temperature (°C)

u_d, u_q: Stator voltage components (V)

motor_speed: Motor speed (rpm)

torque: Torque (Nm)

i_d, i_q: Stator current components (A)

stator_yoke: Stator yoke temperature (°C)

stator_tooth: Stator tooth temperature (°C)

Target Variable:

stator_winding: Stator winding temperature (°C) - This is the value the model predicts.

✨ Key Features
Comprehensive EDA: Includes correlation heatmaps and distribution plots to visualize feature relationships.

Model Comparison: Systematic evaluation of 5 different regression algorithms.

High-Performance Model: The final XGBoost Regressor achieves an R² score of over 99.8%.

Feature Importance Analysis: Identifies which sensor readings are most influential in predicting the temperature.

Ready for Deployment: The notebook includes code to save the trained model (.pkl file) for use in other applications.
