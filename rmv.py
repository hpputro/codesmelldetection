import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn_rvm import EMRVC
from MyModel import MyModel

import time
import psutil
import sklearn.model_selection as ms
import sklearn.metrics as met


def classify(file):
    print(file)
    df = pd.read_csv(file)
    X = df.drop(['idx', 'label'], axis=1).values
    y = df['label'].values
    print(X.shape)

    model = []
    #model.append(MyModel("SVM rbf", SVC()))
    #model.append(MyModel("SVM linear", SVC(kernel='linear')))
    #model.append(MyModel("SVM sigmoid", SVC(kernel='sigmoid')))
    #model.append(MyModel("SVM poly 2", SVC(kernel='poly', degree=2)))
    #model.append(MyModel("SVM poly 3", SVC(kernel='poly')))

    model.append(MyModel("RVM rbf", EMRVC()))
    model.append(MyModel("RVM linear", EMRVC(kernel='linear')))
    model.append(MyModel("RVM sigmoid", EMRVC(kernel='sigmoid')))
    model.append(MyModel("RVM poly 2", EMRVC(kernel='poly', degree=2)))
    model.append(MyModel("RVM poly 3", EMRVC(kernel='poly')))

    print("prepare")
    kfold = ms.KFold(n_splits=4, random_state=0, shuffle=True)
    for train_ndx, test_ndx in kfold.split(X):
        train_X, test_X, train_y, test_y = X[train_ndx], X[test_ndx], y[train_ndx], y[test_ndx]

        for md in model:
            print(md.name)

            st = time.time()
            sm = psutil.virtual_memory().used
            sp = psutil.cpu_percent(interval=1)

            md.clf.fit(train_X, train_y)
            test_predict = md.clf.predict(test_X)

            et = time.time()
            em = psutil.virtual_memory().used
            ep = psutil.cpu_percent(interval=1)

            md.time.append(et - st)
            md.memo.append(abs(em - sm))
            md.proc.append(abs(ep - sp))
            md.accu.append(met.accuracy_score(test_y, test_predict))
            md.prec.append(met.precision_score(test_y, test_predict))
            md.recc.append(met.recall_score(test_y, test_predict))
            md.f1.append(met.f1_score(test_y, test_predict))
            mat = (met.confusion_matrix(test_y, test_predict))
            md.mat += mat

    print(file)
    for md in model:
        print(md.name)
        print(np.mean(md.time))
        print(np.mean(md.memo))
        print(np.mean(md.proc))
        print(np.mean(md.accu))
        print(np.mean(md.prec))
        print(np.mean(md.recc))
        print(np.mean(md.f1))
        print(md.mat)
        print()


#classify("sc2vector.csv")
#classify("sc2tfidf.csv")
#classify("sc2tfidf_vector.csv")
#classify("ts2vector.csv")
#classify("ts2tfidf.csv")
#classify("ts2tfidf_vector.csv")
#classify("sc_ts2vector.csv")
#classify("sc_ts2tfidf.csv")
classify("sc_ts2tfidf_vector.csv")
