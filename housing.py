import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split


import os
import tarfile
from six.moves import urllib
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join(os.path.abspath(os.getcwd()),"datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(url=HOUSING_URL, path=HOUSING_PATH, root=DOWNLOAD_ROOT):
    print(os.path.isdir(path))
    if not os.path.isdir(path):
        os.makedirs(path)
    else: return None
        
    tgz_path = os.path.join(path, "housing.tgz")
    urllib.request.urlretrieve(url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=path)
    housing_tgz.close()


import pandas as pd

def load_housing_data(path=HOUSING_PATH):
    csv_path = os.path.join(path, "housing.csv")
    return pd.read_csv(csv_path)


fetch_housing_data()
housing = load_housing_data()
print(housing.head())

print(housing.info())

print(housing["ocean_proximity"].value_counts())

print(housing.describe())

housing.hist(bins=50, figsize=(20,15))
plt.show()


import numpy as np

def split_train_test(data, test_ratio):
    #setting random seed
    np.random.seed(42)
    # shuffle indices
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    # test set indices
    test_indices = shuffled_indices[:test_set_size]
    # train set indices
    train_indices = shuffled_indices[test_set_size:]
    # return train and test sets
    return data.iloc[train_indices], data.iloc[test_indices]

train_data, test_data = split_train_test(housing, 0.2)


from zlib import crc32
import numpy as np
# not sure page 55
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


housing_with_id = housing.reset_index() # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
# could be valid alternatice id
#housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
#train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42) 
# probs best way to split^

housing ["income_cat"]= pd.cut(housing["median_income"],
                               bins = [0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels = [1,2,3,4,5])
housing["income_cat"].hist()
# 0 -> 1.5
# 1.5 -> 3.0
# 3.0 -> 4.5
# 4.5 -> 6.0
# 6.0 -> inf

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
print(split.split(housing, housing["income_cat"]))
for train_index, test_index in split.split(housing , housing["income_cat"]):
    print(train_index, test_index)
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]