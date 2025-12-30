import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_numerical_distribution(data, feature_name):
    """Plots the distribution of a numerical feature."""
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 6))
    sns.histplot(data[feature_name], kde=True)
    plt.title(f'Distribution of {feature_name}')
    plt.show()

def plot_categorical_distribution(data, feature_name):
    """Plots the distribution of a categorical feature."""
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    feature_values, counts = np.unique(data[feature_name], return_counts=True)
    sns.barplot(x=feature_values, y=counts)
    plt.title(f'Distribution of {feature_name}')
    plt.xticks(rotation=45)
    plt.show()

def plot_cost_history(cost_history):
    """Plots the cost history of the model."""
    plt.plot(range(len(cost_history)), cost_history)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost History")
    plt.show()