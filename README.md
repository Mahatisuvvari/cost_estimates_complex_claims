#  Technical Exercise

## Overview

This project focuses on improving claims processes, specifically aimed at improving cost estimates and identifying complex claims. The repository contains models, scripts, and notebooks used for feature engineering, model building, and evaluation.

## Data
1. Please download the data from Kaggle. Follow the link https://www.kaggle.com/datasets/lucamassaron/easy-peasy-its-lemon-squeezy and place it in the data folder as data.csv.

## Installation

1. Create a conda environment and activate it:
    ```bash
    conda create --name technical_exercise python=3.11.9
    conda activate technical_exercise
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Models

- Navigate to the `scripts` directory to run different models.
    ```bash
    cd Suncorp - Technical Exercise/scripts
    ```

- Execute the main script for complex claim model:
    ```bash
    python scripts/complexclaim_model/main.py
    ```

- Execute the main script for estimates model:
    ```bash
    python scripts/estimates_model/main.py
    ```

### Notebooks

- You can explore and run the Jupyter notebooks available in the `notebooks` directory. These notebooks include:
    - `1.0.0 EDA.ipynb`: Exploratory Data Analysis.
    - `1.1 Question 1 Feature Engineering and Model Building - Test.ipynb`: Feature engineering and model building.
    - `1.2 Question 2 Complex Claims - Model.ipynb`: Model for complex claims.



