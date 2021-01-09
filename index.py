import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

from itertools import cycle
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, plot_roc_curve, auc, roc_curve


# ./data.csv dosyasını dahil eder ve id sütununu siler
def importData():
    dataset = pd.read_csv("data.csv")

    dataset = dataset.drop(["id"], axis=1)

    return dataset

# dataset değişkenini 0 ve 1 sayıları arasında normalize eder
def normalizeData(dataset):
    x = dataset.drop(["diagnosis"], axis=1)
    y = dataset.diagnosis.values

    x = (x - np.min(x)) / (np.max(x) - np.min(x))

    return train_test_split(
        x, y, test_size=0.3, random_state=16)

# Karışıklık matrisine göre doğruluk değeri hesaplar
def accuracy(tn, fp, fn, tp):
    acc = (tp + tn) / (tp + tn + fp + fn)
    print("Doğruluk oranı(Accuracy): {0}\n".format(acc))
    return acc

# tn ve fp değerlerine göre özgüllük değeri hesaplar
def specificity(tn, fp):
    specificity = tn / (tn + fp)
    print("Özgüllük(Specificity): {0}\n".format(specificity))
    return specificity

# tp ve fn değerlerine göre duyarlılık değeri hesaplar
def sensitivity(tp, fn):
    sensitivity = tp / (tp + fn)
    print("Duyarlılık(Sensitivity): {0}\n".format(sensitivity))
    return sensitivity

# tp ve fp değerlerine göre hassasiyet değeri hesaplar
def precision(tp, fp):
    precision = tp / (tp + fp)
    print("Hassasiyet(Precision): {0}\n".format(precision))
    return precision

# tp ve fn değerlerine göre hatırlama değeri hesaplar
def recall(tp, fn):
    recall = tp / (tp + fn)
    print("Hatırlama(Recall): {0}\n".format(recall))
    return recall

# tp,fp ve fn değerlerine göre f1 değeri hesaplar
def f_measure(tp, fp, fn):
    f_measure = 2*tp/(2*tp + fn + fp)
    print("F1(F-measure): {0}\n".format(f_measure))
    return f_measure

# sınıflandırıcı ve dataset verilerinin 10-fold yöntemiyle ortalama doğruluğunu ve standat sapmasını hesaplar
def mean_std(estimator, X, y):

    df = cross_val_score(estimator, X, y,
                         cv=10, scoring="accuracy")
    print("Accuracy(mean): {0}".format(df.mean()))
    print("Standart sapma: {0}\n".format(df.std()))

    return df.mean(), df.std()

# Verisetinin GaussianNB sınıflandırıcısına göre ROC eğrisini çizer
def drawROC(dataset):
    # Veriseti normalize ediliyor
    X = dataset.drop(['diagnosis'],axis=1)
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    dataset["diagnosis"].replace({"B": 0, "M": 1}, inplace=True)
    y = dataset.diagnosis
    n_samples, n_features = X.shape

    # Veri seti numpy türünden dizilere dönüştürülüyor
    random_state = np.random.RandomState(0)
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
    y = y.to_numpy()
    
    # Sınıflandırıcı nitelikleri ayarlanıyor
    cv = StratifiedKFold(n_splits=10)
    classifier = GaussianNB()

    # Başlangıç değerleri atanıyor
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    aucs = []
    colors = cycle(['cyan', 'indigo', 'seagreen', 'red', 'blue', 'darkorange'])
    lw = 2
    i = 0

    # AUC değerleri hesaplanıyor
    for (train, test), color in zip(cv.split(X, y), colors):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0

        # FPR ve TPR ile AUC değeri hesaplanıyor
        roc_auc = auc(fpr, tpr)
        # AUC dizisine yeni değer ekleniyor
        aucs.append(roc_auc)
        # Plotlara yeni ROC eğrisi ekleniyor
        plt.plot(fpr, tpr, lw=.5, color=color, linestyle="--",
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1

    # Plota köşegen ekleniyor
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='k',
             label='Luck')

    # Ortalama TPR değeri hesaplanıyor
    mean_tpr /= cv.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    # Ortalama AUC değeri hesapalnıyor
    mean_auc = auc(mean_fpr, mean_tpr)
    # Veriler ekrana yazdırılıyor
    print("AUC (mean): {0}".format(mean_auc))
    print("Standart sapma: {0}\n".format(np.std(aucs)))
    # Plota ortalama ROC eğrisi ekleniyor
    plt.plot(mean_fpr, mean_tpr, color='g',
             label='Mean ROC (AUC = %0.2f)' % mean_auc, lw=lw)

    # Plot verileri ekleniyor
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

# Sınıflandırıcının öğrenme eğrisini çizer
def learningCurve(estimator=None, title="GaussianNB", X=None, y=None, cv=2,
                  n_jobs=None, train_sizes=np.linspace(.1, .9, 9)):
    # Plot verileri giriliyor
    _, axes = plt.subplots(figsize=(7, 5))

    axes.set_title(title)

    axes.set_xlabel("Training examples")
    axes.set_ylabel("Accuracy(mean)")

    # Öğrenme eğrisi için veriler hesaplanıyor
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True, random_state=16)
    
    # Ortalama eğitim doğruluğu hesaplanıyor
    train_scores_mean = np.mean(train_scores, axis=1)
    # EĞitim doğruluğunun standart sapması hesaplanıyor
    train_scores_std = np.std(train_scores, axis=1)
    # Plot learning curve
    axes.grid()
    # Standart sapmalar grafiğe dahil ediliyor
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.1,
                      color="r")
    
    # Grafik için tanımlayıcı ekleniyor
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
              label="Training score")
    axes.legend(loc="best")
    # Grafik çizdiriliyor
    plt.show()

# Karışıklık matrisinin verilerini kabul görülen forma getirir
def confusionMatrix(tn, fp, fn, tp):
    confusion_matrix = [[tp, fn], [fp, tn]]
    print("Karışıklık Matrisi(Confusion Matrix): {0}\n".format(
        confusion_matrix))
    return confusion_matrix

# Kullanıcı için yönlendirme menüsü
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
    print("10-Fold doğrulama için {0} giriniz.".format(10))
    print("Çıkmak için '{0}' giriniz.\n".format('q'))
    print("İsteğiniz: ", end='')


def main():
    # Veriseti dahil ediliyor
    dataset = importData()

    # Veri seti normalize ediliyor
    data_train, data_test, result_train, result_test = normalizeData(dataset)
    
    # Yeni bir sınıflandırıcı oluşturuluyor ve model eğitiliyor
    nb = GaussianNB()
    nb.fit(data_train, result_train)

    # Test verisetinin sınıfları tahmin ediliyor
    result_prediction = nb.predict(data_test)

    # Tahminler ile doğrular karşılaştırılıyor ve karmaşıklık matrisi oluşturuluyor
    tn, fp, fn, tp = confusion_matrix(result_test, result_prediction).ravel()
    while True:
        # Menü çağırılıyor
        initial_info()
        # Kullanıcıdan bir alan seçmesi isteniyor
        expected = input()

        # İstenen alana göre kullanıcıya sonuçlar gösteriliyor
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
            learningCurve(estimator=GaussianNB(), X=dataset.drop(["diagnosis"], axis=1), y=dataset.diagnosis)
        elif expected == 8:
            drawROC(dataset)
        elif expected == 9:
            confusionMatrix(tn, fp, fn, tp)
        elif expected == 10:
            mean_std(nb, dataset.drop(['diagnosis'],axis=1), dataset.diagnosis)
        else:
            print("İstek geçersiz, lütfen geçerli bir veri giriniz.\n")


if __name__ == "__main__":
    main()
