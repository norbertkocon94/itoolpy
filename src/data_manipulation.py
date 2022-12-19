from sklearn.model_selection import train_test_split

# Function that splits data into 3 sets (train, validation, test)
def split_data_classification(df, target, train_size=0.7, val_size=0.15, test_size=0.15, shuffle=True ,random_state=42) -> tuple:
    """Function that splits data into 3 sets (train, validation, test)

    Args:
        df (pd.DataFrame): Dataframe with data
        target (str): Target column name
        train_size (float, optional): Train size. Defaults to 0.7.
        val_size (float, optional): Validation size. Defaults to 0.15.
        test_size (float, optional): Test size. Defaults to 0.15.
        shuffle (bool, optional): Shuffle data. Defaults to True.
        random_state (int, optional): Random state. Defaults to 42
    Returns:
        tuple: Tuple with 3 dataframes (train, validation, test)
    """        

    # Split data into train and test
    train, test = train_test_split(df, train_size=train_size, test_size=test_size, random_state=random_state, shuffle=shuffle)

    # Split data into train and validation
    train, val = train_test_split(train, train_size=train_size, test_size=val_size, random_state=random_state, shuffle=shuffle)

    # Split target from features
    X_train, y_train = train.drop(target, axis=1), train[target]
    X_val, y_val = val.drop(target, axis=1), val[target]
    X_test, y_test = test.drop(target, axis=1), test[target]

    return X_train, y_train, X_val, y_val, X_test, y_test

# Function that splits data into 3 sets (train, validation, test) for regression
def split_data_regression(df, target, train_size=0.7, val_size=0.15, test_size=0.15, shuffle=True, random_state=42) -> tuple:
    """Function that splits data into 3 sets (train, validation, test) for regression

    Args:
        df (pd.DataFrame): Dataframe with data
        target (str): Target column name
        train_size (float, optional): Train size. Defaults to 0.7.
        val_size (float, optional): Validation size. Defaults to 0.15.
        test_size (float, optional): Test size. Defaults to 0.15.
        random_state (int, optional): Random state. Defaults to 42
        shuffle (bool, optional): Shuffle data. Defaults to True.
    Returns:
        tuple: Tuple with 3 dataframes (train, validation, test)
    """        

    # Split data into train and test
    train, test = train_test_split(df, train_size=train_size, test_size=test_size, random_state=random_state, shuffle=shuffle)

    # Split data into train and validation
    train, val = train_test_split(train, train_size=train_size, test_size=val_size, random_state=random_state, shuffle=shuffle)

    # Split target from features
    X_train, y_train = train.drop(target, axis=1), train[target]
    X_val, y_val = val.drop(target, axis=1), val[target]
    X_test, y_test = test.drop(target, axis=1), test[target]

    return X_train, y_train, X_val, y_val, X_test, y_test