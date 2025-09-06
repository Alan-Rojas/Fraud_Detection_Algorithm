# Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import PowerTransformer
from scipy.stats import norm 
#---------------------------------------------------------------------Data Cleanning---------------------------------------------------------------
def separate_dataset_by_class(data):
    """
    This function takes a dataset with n number of columns which one of them is named Class.
    The function retruns a list of datasets separated by Class values
    """
    output = []

    for val in data["Class"].unique():
        filtered = data[data["Class"] == val]
        output.append(filtered)
    return output

def normalize(x, verbose=False):
    """
    Attempts to transform a pandas Series into a normally distributed form.
    Uses Shapiro-Wilk test for normality (p > 0.05 = normal).
    Tries log, Box-Cox, and Yeo-Johnson transformations.
    Returns the transformed Series.
    """
    x = x.dropna()  # remove NaNs for Shapiro
    x_array = x.values.reshape(-1, 1)  # for sklearn
    
    # If already normal, return original
    try:
        p_val = stats.shapiro(x)[1]
    except:
        if verbose: print("Shapiro failed (maybe constant values)")
        return x

    if p_val > 0.05:
        if verbose: print("Already normal")
        return x

    # Try transformations
    transforms = []

    # 1. Log
    if (x > 0).all():
        try:
            x_log = np.log(x)
            if stats.shapiro(x_log)[1] > 0.05:
                if verbose: print("Log transform worked")
                return pd.Series(x_log, index=x.index)
            transforms.append(pd.Series(x_log, index=x.index))
        except Exception as e:
            if verbose: print("Log failed:", e)

    # 2. Box-Cox
    if (x > 0).all():
        try:
            boxcox_transformed, _ = stats.boxcox(x)
            if stats.shapiro(boxcox_transformed)[1] > 0.05:
                if verbose: print("Box-Cox transform worked")
                return pd.Series(boxcox_transformed, index=x.index)
            transforms.append(pd.Series(boxcox_transformed, index=x.index))
        except Exception as e:
            if verbose: print("Box-Cox failed:", e)

    # 3. Yeo-Johnson (can handle zero and negative)
    try:
        pt = PowerTransformer(method='yeo-johnson')
        yeoj = pt.fit_transform(x_array).flatten()
        if stats.shapiro(yeoj)[1] > 0.05:
            if verbose: print("Yeo-Johnson transform worked")
            return pd.Series(yeoj, index=x.index)
        transforms.append(pd.Series(yeoj, index=x.index))
    except Exception as e:
        if verbose: print("Yeo-Johnson failed:", e)

    if verbose: print("No transform passed normality test. Returning last attempt.")
    
    return transforms[-1] if transforms else x

def drop_outliers(x, threshold=2):
    """
    Drops values more than `threshold` standard deviations away from the mean.
    Default is 2 std dev.
    """
    mean = np.mean(x)
    std_dev = np.std(x)
    return x[np.abs(x - mean) <= threshold * std_dev]

def clean_data():
    credit_cards = pd.read_csv("creditcard.csv")
    separated = separate_dataset_by_class(credit_cards)
    separated[0] = separated[0].sample(frac = 1).reset_index(drop = True).iloc[:separated[1].shape[0]]
    balanced_transactions = pd.concat(separated)
    return balanced_transactions

def get_training_data():
    df = clean_data()
    x = df.drop(columns=["Class"])
    y = df["Class"]

    x_0 = pd.DataFrame(index=x.index)

    for col in x.columns:

        # Step 1: Drop outliers
        col_clean = drop_outliers(x[col])

        # Step 2: Normalize
        col_normal = normalize(col_clean)
        
        # Reinsert with correct indexing (keep NaNs for now)
        x_0[col] = col_normal

    # Add back target
    x_0["Class"] = y

    # Optionally: Drop rows with any NaN introduced by outlier removal
    Clean_data = x_0.dropna().copy()
    return Clean_data

def get_all_data():
    """
    This function returns all data (separated in x and y) to be tested with the trained models.
    """
    df = pd.read_csv("creditcard.csv")
    x, y = df.drop(columns = ["Class"]), df["Class"]
    x_0 = pd.DataFrame(index=x.index)

    for col in x.columns:

        # Step 2: Normalize
        col_normal = normalize(x[col])
        
        # Reinsert with correct indexing
        x_0[col] = col_normal

    return x_0, y
