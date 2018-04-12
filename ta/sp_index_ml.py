""" Lib to run ML on SP500 """
import os
import pickle
import datetime
import pandas as pd
import numpy as np
import stockstats
from pandas_datareader import data as web
from sklearn import svm, cross_validation, neighbors
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf


OUTPUT_DIR = 'sp_ml'
SPRAW_FILE = OUTPUT_DIR + '/sp_raw'
DF_FILE = OUTPUT_DIR + '/sp500_indicators'
LABELS_FILE = OUTPUT_DIR + '/sp500_labels'
TRAIN_FILE = OUTPUT_DIR + '/training'
INIT_INDICATORS = ['change', 'rate', 'middle', 'boll', 'boll_ub', 'boll_lb', 'macd', 'macds', 'macdh',
                   'kdjk', 'kdjd', 'kdjj', 'cr', 'cr-ma1', 'cr-ma2', 'cr-ma3', 'cci', 'tr', 'atr', 'um', 'dm', 'pdi', 'mdi', 'dx', 'adx', 'adxr',
                   'trix', 'vr', 'dma', 'log-ret']

   
def get_sp500(force=True, start=2000):
    """ Get is from MSTAR """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if os.path.exists(SPRAW_FILE) and not force:
        index = pd.read_csv(SPRAW_FILE, index_col='Date', parse_dates=['Date'])
    else:
        index = web.DataReader('SPX', "morningstar", datetime.datetime(start, 1, 1), datetime.date.today())
        index = index.reset_index('Symbol').drop('Symbol', 1)
        index.to_csv(SPRAW_FILE)
    return index


def get_index_indicators(force=True, index=None, delta_days=7):
    if force or index is None or not os.path.exists(DF_FILE):
        index = get_sp500()
        df = stockstats.StockDataFrame.retype(index)
        indicators = INIT_INDICATORS
        for i in range(1, delta_days+1):
            indicators.append('close_-' + str(i) + '_r')
            indicators.append('open_-' + str(i) + '_r')
            indicators.append('volume_-' + str(i) + '_r')
        df1 = df[indicators]
        df1.to_csv(DF_FILE)
    else:
        df1 = pd.read_csv(DF_FILE, index_col='Date', parse_dates=['Date'])
    return df1


def labels_helper(val):
    if val > 2:
        return 2
    elif val > 0.5:
        return 1
    elif val > -0.5:
        return 0
    else:
        return -1


def get_labels_table(force=True, index=None, forward_days=7):
    if index is None or force or not os.path.exists(LABELS_FILE):
        index = get_sp500()
        indicators = []
        #for i in range(1, forward_days+1):
            #indicators.append('close_' + str(i) + '_r')
        df = stockstats.StockDataFrame.retype(index)
        df1 = df['close_3_r']
        df1 = df1.apply(labels_helper)
        df1.to_csv(LABELS_FILE)
    else:
        df1 = pd.read_csv(LABELS_FILE, index_col='Date', parse_dates=['Date'])
    return df1


def normalize(x):
    scaler = StandardScaler()
    x_norm = scaler.fit_transform(x.values)
    x_norm = pd.DataFrame(x_norm, index=x.index, columns=x.columns)
    return x_norm


def rebalance(unbalanced_data):
    # Separate majority and minority classes
    data_minority = unbalanced_data[unbalanced_data.close_3_r==0]
    data_majority = unbalanced_data[unbalanced_data.close_3_r==1]
    # Upsample minority class
    n_samples = len(data_majority)
    data_minority_upsampled = resample(data_minority, replace=True, n_samples=n_samples, random_state=5)
    # Combine majority class with upsampled minority class
    data_upsampled = pd.concat([data_majority, data_minority_upsampled])
    data_upsampled.sort_index(inplace=True)
    # Display new class counts
    data_upsampled.close_3_r.value_counts()
    return data_upsampled


def get_pairs(force=True, features=None, labels=None, labels_days=7, test_size=0.25, validation=60):
    X = get_index_indicators(force, features, labels_days)
    y = get_labels_table(force, labels, labels_days)
    nan = 0
    X.fillna(nan, inplace=True)
    y.fillna(nan, inplace=True)
    X.replace([np.inf, -np.inf], nan, inplace=True)
    index_slice = int(np.floor(len(X.index) * test_size)) - validation
    data_train = X.join(y)
    data_train = rebalance(data_train)
    y = data_train.close_3_r
    X = normalize(data_train.drop('close_3_r', axis=1))
    X_train = X.iloc[:-index_slice]
    y_train = y.iloc[:-index_slice]
    X_test = X.iloc[-index_slice:-validation]
    y_test = y.iloc[-index_slice:-validation]
    X_validation = normalize(X.iloc[-validation:-labels_days])
    y_validation = y.iloc[-validation:-labels_days]
    return (X_train, y_train), (X_test, y_test), (X_validation, y_validation)


def train_model(X, y, force=True):
    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])
    clf.fit(X, y)
    pickle.dump(clf, open(TRAIN_FILE, 'wb'))
    return clf


def train_model2(X, y):
    model = MLPClassifier(random_state=5)
    model.fit(X, y)
    model_data = {'hidden_layer_sizes': [10, 50, 100],
                  'activation': ['identity', 'logistic', 'tanh', 'relu'],
                  'solver': ['lbfgs', 'sgd', 'adam'],
                  'learning_rate': ['constant', 'invscaling', 'adaptive'],
                  'max_iter': [200, 400, 800],
                  'random_state': [5]}
    best = GridSearchCV(model, model_data, scoring='f1').fit(X, y)
    model = best.best_estimator_
    return model


def scores(model, X, y):
    X = normalize(X)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print("Accuracy Score: {0:0.2f} %".format(acc * 100))
    #f1 = f1_score(y, y_pred)
    #auc = roc_auc_score(y, y_pred)
    #print("F1 Score: {0:0.4f}".format(f1))
    #print("Area Under ROC Curve Score: {0:0.4f}".format(auc))
    return y_pred

def train_using_tf(X_train, y_train, X_test, y_test):
    # Number of epochs and batch size
    epochs = 10
    batch_size = 10
    sigma = 1
    
    #data = X.join(y).values
    #scaler = MinMaxScaler()
    #scaler.fit(data)
    #scaled = scaler.transform(data)
    #X_train = scaled[:, :-1]
    #y_train = scaled[:, -1]
    #scaled_test = scaler.transform(data)
    #X_test = scaled_test[:, :-1]
    #y_test = scaled_test[:, -1]
    n_indicators = X_train.shape[1]
    X = tf.placeholder(dtype=tf.float32, shape=[None, n_indicators])
    Y = tf.placeholder(dtype=tf.float32, shape=[None])
    n_neurons_1 = 1024
    n_neurons_2 = 512
    n_neurons_3 = 256
    n_neurons_4 = 128
    n_target = 1

    # Initializers
    weight_initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG", uniform=True, factor=sigma)
    bias_initializer = tf.zeros_initializer()

    # Layers: Variables for hidden weights and biases
    W_hidden_1 = tf.Variable(weight_initializer([n_indicators, n_neurons_1]))
    bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
    W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
    bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
    W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
    bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
    W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
    bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))
    # Output layer: Variables for output weights and biases
    W_out = tf.Variable(weight_initializer([n_neurons_4, n_target]))
    bias_out = tf.Variable(bias_initializer([n_target]))

    # Hidden layer
    hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
    hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
    hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
    hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))
    # Output layer (must be transposed)
    out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))
    # Cost function
    mse = tf.reduce_mean(tf.squared_difference(out, Y))
    # Optimizer
    opt = tf.train.AdamOptimizer().minimize(mse)
    # Make Session
    net = tf.Session()
    # Run initializer
    net.run(tf.global_variables_initializer())
    # Setup interactive plot
    #plt.ion()
    #fig = plt.figure()
    #ax1 = fig.add_subplot(111)
    #line1, = ax1.plot(y_test)
    #line2, = ax1.plot(y_test*0.5)
    #plt.show()
    for e in range(epochs):
        # Shuffle training data
        shuffle_indices = np.random.permutation(np.arange(len(y_train)))
        X_train = X_train.iloc[shuffle_indices]
        y_train = y_train.iloc[shuffle_indices]
    
        # Minibatch training
        for i in range(0, len(y_train) // batch_size):
            start = i * batch_size
            batch_x = X_train[start:start + batch_size]
            batch_y = y_train[start:start + batch_size]
            # Run optimizer with batch
            net.run(opt, feed_dict={X: batch_x, Y: batch_y})
    # Print final MSE after Training
    mse_final = net.run(mse, feed_dict={X: X_test, Y: y_test})
    print(mse_final)
    pred = net.run(out, feed_dict={X: X_test})
    return pred, y_test


def test_model(model, X, y):
    confidence = model.score(X, y)
    print "Confidence={} , Tested on : {} - {}".format(confidence, X.index[0], X.index[-1])
    return confidence

sp = get_sp500(True, 2000)
ndx_i = get_index_indicators(True, sp, 3)
lbls = get_labels_table(True, sp, 3)
train, test, validation = get_pairs(True, ndx_i, lbls, 3, 0.25, 60)

#model = train_model(train[0], train[1], True)
#confidence = test_model(model, test[0], test[1])
#v_t = pd.DataFrame({'predict': model.predict(validation[0]), 'real': validation[1]})

#pred, y_test = train_using_tf(train[0], train[1], test[0], test[1])
#df = pd.DataFrame({'pred': pred[0], 'real': y_test})

model = train_model2(train[0], train[1])
scores(model, test[0], test[1])
y_pred = scores(model, validation[0], validation[1])
v_t = pd.DataFrame({'predict': y_pred, 'real': validation[1]})
