from datetime import date, timedelta
from pandas_datareader import data
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge, LassoLars
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import math
import numpy as np

#define a period of 5 years of data
end = date.today()
start = date(end.year - 5, end.month, end.day)

#getting  data from Yahoo Finance
df = data.DataReader("GOOG", 'yahoo', start, end)

#feature engineering
dfreg = df.loc[:,['Adj Close','Volume']]
dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

#Pre-processing data

#drop missing value
dfreg.fillna(value=-99999, inplace=True)

#separate 1 percent of the data to forecast
forecast_out = int(math.ceil(0.01 * len(dfreg)))

#separating the label here, we want to predict the AdjClose
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'], 1))

#scale the X so that everyone can have the same distribution for linear regression
X = preprocessing.scale(X)

#finally We want to find Data Series of late X and early X (train) for model generation and evaluation
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

#separate label and identify it as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]

# saving the last date of the data 
last_date = dfreg.iloc[-1].name

#separation of training and testing of model by cross validation train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#building models 
classifiers = []
classifiers.append((LinearRegression(n_jobs=-1), 'Linear Regression'))
classifiers.append((Ridge(alpha=.5), 'Ridge Regression'))
classifiers.append((BayesianRidge(), 'Bayesian Regression'))
classifiers.append((KNeighborsRegressor(n_neighbors=2), 'KNeighbors Regression'))
classifiers.append((LassoLars(alpha=.1), 'Lasso Regression'))

#testing 
best_acc = 0
for clf, name in classifiers:
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print("Confidence of %s is %s"%(name, acc))
    if best_acc < acc:
        best_clf = clf
        best_acc = acc

#plotting prediction
print("\nPlottig data for %s"%best_clf)
forecast_set = best_clf.predict(X_lately)
dfreg['Forecast'] = np.nan
next_unix = last_date + timedelta(days=1)

for i in forecast_set:
    next_date = next_unix
    next_unix += timedelta(days=1)
    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]

# Adjusting the size of matplotlib
mpl.rc('figure', figsize=(16, 8))
# Adjusting the style of matplotlib
style.use('ggplot')
dfreg['Adj Close'].tail(500).plot()
dfreg['Forecast'].tail(500).plot()
plt.legend(loc=2)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()