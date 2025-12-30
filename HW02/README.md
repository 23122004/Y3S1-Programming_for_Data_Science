# Predicting Job Changes for Data Scientists

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Method](#method)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Challenges & Solutions](#challenges--solutions)
- [Future Improvements](#future-improvements)
- [Contributors](#contributors)
- [Contact](#contact)
- [License](#license)

## Introduction
This project aims to predict whether a data scientist will look for a new job or not. The project is a part of the CSC17104 - Programming for Data Science course and is implemented using NumPy for data manipulation and Matplotlib/Seaborn for visualization. The core of the project is to build a machine learning model from scratch using NumPy to predict job changes.

## Dataset
The dataset is from Kaggle: [HR Analytics: Job Change of Data Scientists](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists).

### Features
The dataset contains the following features:
- `enrollee_id`: Unique ID for candidate
- `city`: City code
- `city_development_index`: Development index of the city (scaled)
- `gender`: Gender of candidate
- `relevent_experience`: Relevant experience of candidate
- `enrolled_university`: Type of University course enrolled in
- `education_level`: Education level of candidate
- `major_discipline`: Major discipline of candidate
- `experience`: Candidate total experience in years
- `company_size`: Size of current employer's company
- `company_type`: Type of current employer's company
- `last_new_job`: Difference in years between previous job and current job
- `training_hours`: training hours completed
- `target`: 0 – Not looking for job change, 1 – Looking for a job change

### Data Size and Characteristics
- The training set has 19158 instances and 14 features.
- The test set has 2129 instances and 13 features.

## Method
### Data Processing
- **Data Loading:** The data is loaded using NumPy.
- **Preprocessing:**
    - Handling missing values.
    - Normalization/Standardization of numerical features.
    - Encoding categorical features.
- **Feature Engineering:** New features will be created from existing ones to improve model performance.

### Machine Learning Model
A Logistic Regression model will be implemented from scratch using NumPy.

- **Algorithm:** Logistic Regression
- **Loss Function:** Binary Cross-Entropy
- **Optimization:** Gradient Descent

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repository.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
- **Data Exploration:** Run the `notebooks/01_data_exploration.ipynb` notebook.
- **Preprocessing:** Run the `notebooks/02_preprocessing.ipynb` notebook.
- **Modeling:** Run the `notebooks/03_modeling.ipynb` notebook.

## Results
This section will be updated with the model's performance metrics (accuracy, precision, recall, F1-score) and visualizations.

## Project Structure
```
project-name/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── visualization.py
│   └── models.py
```

## Challenges & Solutions
This section will document the challenges faced during the project, especially with NumPy, and the solutions implemented.

## Future Improvements
This section will discuss potential future improvements to the project.

## Contributors
- Nguyễn Lê Hoàng Trung (23122004)

## Contact
- 23122004@student.hcmus.edu.vn

## License
This project is licensed under the MIT License.