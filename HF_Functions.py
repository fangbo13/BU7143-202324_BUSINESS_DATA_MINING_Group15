#import  Essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from collections import Counter
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,roc_auc_score, precision_score,f1_score,RocCurveDisplay, ConfusionMatrixDisplay,classification_report
import warnings


# Reference:https://github.com/azizepalali/rule_based_classification/blob/main/rule_based_classification.py
# Categoric or Numeric Data Analysis
def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols, num_cols, cat_but_car


# Reference:https://github.com/azizepalali/rule_based_classification/blob/main/rule_based_classification.py
def cat_summary(dataframe, col_name, plot=False):

    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        plt.style.use('seaborn-darkgrid')
        fig, ax = plt.subplots(1, 2)
        ax = np.reshape(ax, (1, 2))
        ax[0, 0] = sns.countplot(x=dataframe[col_name], color="green", ax=ax[0, 0])
        ax[0, 0].set_ylabel('Count')
        ax[0, 0].set_xticklabels(ax[0, 0].get_xticklabels(), rotation=-45)
        ax[0, 1] = plt.pie(dataframe[col_name].value_counts().values, labels=dataframe[col_name].value_counts().keys(),colors=sns.color_palette('bright'), shadow=True, autopct='%.0f%%')
        plt.title("Percent")
        fig.set_size_inches(10, 6)
        fig.suptitle('Analysis of Categorical Variables', fontsize=13)
        plt.show()


def correlated_map(dataframe, plot=False):  # Define correlated_ Map function
    # Used to find pairwise correlation of all columns in a data frame
    corr = dataframe.corr()
    if plot:
        # Set the size of the graph by passing the dictionary to the parameter using the key RC in the Seaborn method
        sns.set(rc={'figure.figsize': (20, 20)})
        # Draw a heat map, using CMAP to set the color of the graph to Paired_ R, annot annotates the heat map, set to True to write data values to the cell
        sns.heatmap(corr, cmap="Paired_r", annot=True, linewidths=.7)
        # Set the size of the label body of the X-axis and offset it to -30 degrees-30°
        plt.xticks(rotation=-30, size=10)
        plt.yticks(size=10)  # Set label body size for y-axis
        plt.title('Correlation Map', size=20)  # Set title and size to 20
        plt.show()


def label_encoder(dataframe, binary_col):  #Define label_ Encoder function
    labelencoder = LabelEncoder() #Instantiate labelencoder object
    dataframe[binary_col] = labelencoder.fit_transform(
        dataframe[binary_col])  #Apply le to Classification Feature Columns
    return dataframe  #Return dataframe




# Reference:https://gist.github.com/joseph-allen/14d72af86689c99e1e225e5771ce1600
def detect_outliers(df,n,features):
    outlier_indices = []
    """
    Detect outliers from given list of features. It returns a list of the indices
    according to the observations containing more than n outliers according
    to the Tukey method
    """
    # iterate over features(columns)
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col],75)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   

# def plot_confusion_matrix(y_test,y_test_LRSC):
#     acc = round(accuracy_score(y_test,y_test_LRSC), 2)
#     cm = confusion_matrix(y_test,y_test_LRSC)
#     sns.heatmap(cm, annot=True, fmt=".0f")
#     plt.title('Accuracy Score: {0}'.format(acc), size=10)
#     plt.show()
# Creating a function for prediction
