import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_data():
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']

    raw_dataset = pd.read_csv(url, names=column_names, na_values='?', comment='\t',
                              sep=' ', skipinitialspace=True)

    # remove entries with missing values
    dataset = raw_dataset.dropna()
    # from sklearn import preprocessing
    # normalized_features = preprocessing.StandardScaler().fit_transform(dataset)
    # dataset = pd.DataFrame(data=normalized_features, columns=column_names)
    return dataset


def inspect_data(dataset):
    print('Dataset shape:')
    print(dataset.shape)

    print('Tail:')
    print(dataset.tail())

    print('Statistics:')
    print(dataset.describe().transpose())

    sns.pairplot(dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
    plt.show()


def split_data(dataset):
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    return train_dataset, test_dataset