from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd

# Load the CSV file (should be in the same directory)
data = pd.read_csv("norway_new_car_sales_by_make.csv")

# Create a column "Period" with both the Year and the Month
data["Period"] = data["Year"].astype(str) + "-" + data["Month"].astype(str)
# We use the datetime formatting to make sure format is consistent
data["Period"] = pd.to_datetime(data["Period"]).dt.strftime(" % Y - % m")

# Create a pivot of the data to show the periods on columns and the car
# makers on rows
df = pd.pivot_table(
    data=data,
    values="Quantity",
    index="Make",
    columns="Period",
    aggfunc="sum",
    fill_value=0)

# Print data to Excel for reference
df.to_excel("Clean Demand.xlsx")


def import_data():
    data = pd.read_csv("norway_new_car_sales_by_make.csv")
    data["Period"] = data["Year"].astype(str) + "-" + data["Month"].astype(str)
    data["Period"] = pd.to_datetime(data["Period"]).dt.strftime("%Y-%m")
    df = pd.pivot_table(
        data=data,
        values="Quantity",
        index="Make",
        columns="Period",
        aggfunc="sum",
        fill_value=0)
    return df


def datasets(df, x_len=12, y_len=1, y_test_len=12):
    D = df.values

    periods = D.shape[1]

    # Training set creation: run through all the possible time windows
    loops = periods + 1 - x_len - y_len - y_test_len
    train = []
    for col in range(loops):
        train.append(D[:, col:col + x_len + y_len])
    train = np.vstack(train)
    X_train, Y_train = np.split(train, [x_len], axis=1)

    # Test set creation: unseen "future" data with the demand just before
    max_col_test = periods - x_len - y_len + 1
    test = []
    for col in range(loops, max_col_test):
        test.append(D[:, col:col + x_len + y_len])
    test = np.vstack(test)
    X_test, Y_test = np.split(test, [x_len], axis=1)

    # this data formatting is needed if we only predict a single period
    if y_len == 1:
        Y_train = Y_train.ravel()
    Y_test = Y_test.ravel()

    return X_train, Y_train, X_test, Y_test


df = import_data()
X_train, Y_train, X_test, Y_test = datasets(df)

# - Instantiate a Decision Tree Regressor
tree = DecisionTreeRegressor(max_depth=5, min_samples_leaf=5)

# - Fit the tree to the training data
tree.fit(X_train, Y_train)

# Create a prediction based on our model
Y_train_pred = tree.predict(X_train)

# Compute the Mean Absolute Error of the model

MAE_tree = np.mean(abs(Y_train - Y_train_pred)) / np.mean(Y_train)

# Print the results
print("Tree on train set MAE %:", round(MAE_tree * 100, 1))

Y_test_pred = tree.predict(X_test)
MAE_test = np.mean(abs(Y_test - Y_test_pred)) / np.mean(Y_test)
print("Tree on test set MAE%:", round(MAE_test * 100, 1))
