import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, plot_roc_curve


def importData():
    dataset = pd.read_csv("data.csv")

    dataset = dataset.drop(["id", "Unnamed: 32"], axis=1)

    M = dataset[dataset.diagnosis == "M"]

    B = dataset[dataset.diagnosis == "B"]

    return dataset


def normalizeData(dataset):
    x = dataset.drop(["diagnosis"], axis=1)
    y = dataset.diagnosis.values

    x = (x - np.min(x)) / (np.max(x) - np.min(x))

    return train_test_split(
        x, y, test_size=0.3, random_state=16)


def accuracy(tn, fp, fn, tp):
    acc = (tp + tn) / (tp + tn + fp + fn)
    print("Doğruluk oranı(Accuracy): {0}\n".format(acc))
    return acc


def specificity(tn, fp):
    specificity = tn / (tn + fp)
    print("Özgüllük(Specificity): {0}\n".format(specificity))
    return specificity


def sensitivity(tp, fn):
    sensitivity = tp / (tp + fn)
    print("Duyarlılık(Sensitivity): {0}\n".format(sensitivity))
    return sensitivity


def precision(tp, fp):
    precision = tp / (tp + fp)
    print("Hassasiyet(Precision): {0}\n".format(precision))
    return precision


def recall(tp, fn):
    recall = tp / (tp + fn)
    print("Hatırlama(Recall): {0}\n".format(recall))
    return recall


def f_measure(tp, fp, fn):
    f_measure = 2*tp/(2*tp + fn + fp)
    print("F1(F-measure): {0}\n".format(f_measure))
    return f_measure


def mean_std(estimator, X, y):
    df = cross_val_score(nb, data_train, result_train,
                         cv=10, scoring="accuracy")
    print(df.mean())
    print("standart sapma: {0}\n".format(df.std()))

    return df.mean(), df.std()


def drawROC(estimator, X, y):
    plot_roc_curve(estimator, X, y)
    plt.show()


def learningCurve(estimator=None, title="GaussianNB", X=None, y=None, cv=None,
                  n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    _, axes = plt.subplots(figsize=(7, 5))

    axes.set_title(title)

    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.1,
                      color="r")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
              label="Training score")
    axes.legend(loc="best")
    plt.show()


def confusionMatrix(tn, fp, fn, tp):
    confusion_matrix = [[tp, fn], [fp, tn]]
    print("Karışıklık Matrisi(Confusion Matrix): {0}\n".format(
        confusion_matrix))
    return confusion_matrix


def initial_info():
    print("Doğruluk (Accuracy) değeri için {0} giriniz.".format(1))
    print("Duyarlılık (Sensitivity) değeri için {0} giriniz.".format(2))
    print("Özgüllük (Specificity) değeri için {0} giriniz.".format(3))
    print("Hassasiyet (Precision) değeri için {0} giriniz.".format(4))
    print("Hatırlama (Recall) değeri için {0} giriniz.".format(5))
    print("F1 (F-measure) değeri için {0} giriniz.".format(6))
    print("Öğrenme eğrisi (Learning Curve) için {0} giriniz.".format(7))
    print("ROC eğrisi için {0} giriniz.".format(8))
    print("Karışıklık Matrisi(Confusion Matrix) için {0} giriniz.".format(9))
    print("Çıkmak için '{0}' giriniz.\n".format('q'))
    print("İsteğiniz: ", end='')


def main():
    dataset = importData()

    data_train, data_test, result_train, result_test = normalizeData(dataset)

    nb = GaussianNB()
    nb.fit(data_train, result_train)

    result_prediction = nb.predict(data_test)

    tn, fp, fn, tp = confusion_matrix(result_test, result_prediction).ravel()
    while True:
        initial_info()
        expected = input()

        if expected == "q":
            sys.exit()
        expected = int(expected)
        if expected == 1:
            accuracy(tn, fp, fn, tp)
        elif expected == 2:
            sensitivity(tp, fn)
        elif expected == 3:
            specificity(tn, fp)
        elif expected == 4:
            precision(tp, fp)
        elif expected == 5:
            recall(tp, fn)
        elif expected == 6:
            f_measure(tp, fp, fn)
        elif expected == 7:
            learningCurve(estimator=nb, X=data_train, y=result_train)
        elif expected == 8:
            drawROC(nb, data_test, result_test)
        elif expected == 9:
            confusionMatrix(tn, fp, fn, tp)
        else:
            print("İstek geçersiz, lütfen geçerli bir veri giriniz.\n")


if __name__ == "__main__":
    main()
