'''Test file '''

import numpy as np

clf = MLPClassifier(activation="relu", solver="sgd", max_iter = 100, random_state=64)
clf.fit(X_train, t_train)
t_predict = clf.predict(X_test)
acs_unscaled_relu = accuracy_score(t_test, t_predict)

clf = MLPClassifier(activation="logistic", solver="sgd", max_iter = 100, random_state=64)
clf.fit(X_train, t_train)
t_predict = clf.predict(X_test)
acs_unscaled_sigmoid = accuracy_score(t_test, t_predict)


scaler = StandardScaler()  # Utilizing scikit's standardscaler
scaler_x = scaler.fit(X_train)  # Scaling x-data
X_train = scaler_x.transform(X_train)
X_test = scaler_x.transform(X_test)

clf = MLPClassifier(activation="relu", solver="sgd", max_iter = 100, random_state=64)
clf.fit(X_train, t_train)
t_predict = clf.predict(X_test)
acs_scaled_relu = accuracy_score(t_test, t_predict)
print(acs_scaled_relu)

clf = MLPClassifier(activation="logistic", solver="sgd", max_iter = 100, random_state=64)
clf.fit(X_train, t_train)
t_predict = clf.predict(X_test)
acs_scaled_sigmoid = accuracy_score(t_test, t_predict)
print(acs_scaled_sigmoid)

inputs = X
temp1=np.reshape(inputs[:,1],(len(inputs[:,1]),1))
temp2=np.reshape(inputs[:,2],(len(inputs[:,2]),1))
X=np.hstack((temp1,temp2))
temp=np.reshape(inputs[:,5],(len(inputs[:,5]),1))
X=np.hstack((X,temp))
temp=np.reshape(inputs[:,8],(len(inputs[:,8]),1))
X=np.hstack((X,temp))
print(X.shape)
del temp1,temp2,temp

X_train, X_test, t_train, t_test = train_test_split(X,targets, test_size=0.2)
scaler = StandardScaler()  # Utilizing scikit's standardscaler
scaler_x = scaler.fit(X_train)  # Scaling x-data
X_train = scaler_x.transform(X_train)
X_test = scaler_x.transform(X_test)

clf = MLPClassifier(activation="relu", solver="sgd", max_iter = 100, random_state=64)
clf.fit(X_train, t_train)
t_predict = clf.predict(X_test)
acs_dim_relu = accuracy_score(t_test, t_predict)

clf = MLPClassifier(activation="logistic", solver="sgd", max_iter = 100, random_state=64)
clf.fit(X_train, t_train)
t_predict = clf.predict(X_test)
acs_dim_sigmoid = accuracy_score(t_test, t_predict)

fig = plt.figure()
data = ['Original', 'Scaled', 'Reduced dimensonality']
acs_relu = [acs_unscaled_relu, acs_scaled_relu, acs_dim_relu]
acs_sigmoid = [acs_unscaled_sigmoid, acs_scaled_sigmoid, acs_dim_sigmoid]
plt.bar(data, acs_relu)
plt.bar(data, acs_sigmoid)
plt.ylabel('Accuracy score')
plt.legend(labels=['RELU', 'Sigmoid'])
plt.title("Comparrison of Original, scaled and dimension reduced data")
plt.show()
