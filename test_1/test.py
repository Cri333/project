import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit, cross_val_score, KFold, \
    LeaveOneGroupOut, GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn import metrics
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.model_selection import cross_val_score, LeaveOneOut
import warnings
from sklearn.exceptions import ConvergenceWarning
from numpy import mean, std
import seaborn as sns
from sklearn.base import clone
# Ignora gli avvisi di convergenza
warnings.filterwarnings('ignore', category=ConvergenceWarning)


'''classificatori complete'''
def naive_bayes():
    # Definisci i parametri da ottimizzare per Naive Bayes
    parametri_nb = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}
    # Crea un classificatore Naive Bayes
    nb = GaussianNB()
    # Crea l'oggetto GridSearchCV per Naive Bayes
    grid_nb = GridSearchCV(nb, parametri_nb, refit=True)
    return grid_nb


def logistic_regression():
    # Definisci i parametri da ottimizzare per Logistic Regression
    parametri_lr = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
    # Crea un classificatore Logistic Regression
    lr = LogisticRegression(solver='liblinear')
    # Crea l'oggetto GridSearchCV per Logistic Regression
    grid_lr = GridSearchCV(lr, parametri_lr, refit=True)
    return grid_lr



def kn():
    # Definisci i parametri da ottimizzare per K Neighbors Classifier
    parametri_kn = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
    # Crea un classificatore K Neighbors Classifier
    kn = KNeighborsClassifier(leaf_size=1, metric='manhattan')
    # Crea l'oggetto GridSearchCV per K Neighbors Classifier
    grid_kn = GridSearchCV(kn, parametri_kn, refit=True)
    return grid_kn


def svm():
    # Definisci i parametri da ottimizzare per SVM
    parametri_svm = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001]}
    # Crea un classificatore SVM con kernel RBF
    svm = SVC(kernel='rbf', probability=True)
    # Crea l'oggetto GridSearchCV per SVM
    grid_svm = GridSearchCV(svm, parametri_svm, refit=True)
    return grid_svm


def decision_tree():
    # Definisci i parametri da ottimizzare per Decision Tree Classifier
    parametri_dt = {'max_depth': [None, 2, 4, 6], 'min_samples_split': [2, 3]}
    # Crea un classificatore Decision Tree Classifier
    dt = DecisionTreeClassifier(criterion='entropy', splitter='best')
    # Crea l'oggetto GridSearchCV per Decision Tree Classifier
    grid_dt = GridSearchCV(dt, parametri_dt, refit=True)
    return grid_dt


def random_forest():
    # Definisci i parametri da ottimizzare per Random Forest Classifier
    parametri_rf = {'n_estimators': [10, 20, 30, 40]}
    # Crea un classificatore Random Forest Classifier
    rf = RandomForestClassifier(max_features='sqrt', criterion='entropy', max_depth=None,
                                min_samples_split=2, min_samples_leaf=1)
    # Crea l'oggetto GridSearchCV per Random Forest Classifier
    grid_rf = GridSearchCV(rf, parametri_rf, refit=True)
    return grid_rf


# Crea un dizionario di classificatori
classifiers_complete = {
    "Naive Bayes": naive_bayes(),
    "Logistic Regression": logistic_regression(),
    "KN": kn(),
    "SVM": svm(),
    "Decision Tree": decision_tree(),
    "Random Forest": random_forest()
}


'''classificatori semplici'''
def define_ilknn():
    # ILK-NN con distanza euclidea e K=1
    ilknn = KNeighborsClassifier(n_neighbors=1, metric='euclidean', weights='distance')
    return ilknn

def define_svm():
    # SVM con kernel Radial Basis Function (RBF)
    # I valori ottimali di C e gamma devono essere trovati tramite la ricerca nella griglia
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)  # Valori di C e gamma sono esempi
    return svm

def define_ann():
    # ANN con architettura superficiale
    # Il numero di strati nascosti e il numero di neuroni in ciascuno di essi vengono selezionati mediante ricerca a griglia
    # Se H1 = 0 o H2 = 0, la ANN risultante ha un solo strato nascosto. Se H1 = H2 = 0, la ANN risultante non ha livelli nascosti
    ann = MLPClassifier(hidden_layer_sizes=(50,))  # Il valore 50 è un esempio
    return ann


def train_stacked_linear():
    # Creazione dei classificatori base
    knn = define_ilknn()
    svm = define_svm()
    ann = define_ann()

    # Creazione del classificatore stacked
    lr = LogisticRegression()
    classifiers = [('KNN', knn), ('SVM', svm), ('ANN', ann)]

    stacked = StackingClassifier(estimators=classifiers, final_estimator=lr)
    return stacked

from sklearn.ensemble import VotingClassifier

def train_stacked_majority():
    # Creazione dei classificatori base
    knn = define_ilknn()
    svm = define_svm()
    ann = define_ann()

    # Creazione del classificatore di voto
    vote = VotingClassifier(estimators=[('KNN', knn), ('SVM', svm), ('ANN', ann)], voting='hard')

    # Creazione del classificatore stacked
    stacked = StackingClassifier(estimators=[('KNN', knn), ('SVM', svm), ('ANN', ann)], final_estimator=vote)
    return stacked



# Crea un dizionario di classificatori
classifiers = {
    "KN semplice": define_ilknn(),
    "SVM semplice": define_svm(),
    "AN semplice": define_ann(),
   # "Stack_linear": train_stacked_linear(),
  #"Stack_majority": train_stacked_majority()

}


'''lettura e filtraggio dati'''
# Leggi i dati
df = pd.read_csv('driver_stress_Time_train.csv')


#df = df.dropna(axis=1)


# Definisci le caratteristiche e le etichette
X = df.drop(['driver', 'Time_Interval', 'stress'], axis=1)
#X = df.filter(regex='^(Heart.Rate|Breathing.Rate|Perinasal.Perspiration|Palm.EDA)')

X_HR = X.filter(regex='^Heart.Rate')
X_BR = X.filter(regex='^Breathing.Rate')
X_PP = X.filter(regex='^Perinasal.Perspiration')
X_Palm_EDA = X.filter(regex='^Palm.EDA')
X_SPEED = X.filter(regex='^Speed')
X_ACC = X.filter(regex='^Acceleration')
X_BRA = X.filter(regex='^Breaking')
X_STEE = X.filter(regex='^Steering')
y = df['stress']  # Etichette


# Crea l'array 'groups'
le = LabelEncoder()
groups = le.fit_transform(df['driver'])




'''5 fold cross validation'''
def five_fold_cross_validation(X, y, groups, classifiers):
    #cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv = GroupKFold(n_splits=5)

    scores_dict = {}

    for train_index, test_index in cv.split(X, y, groups):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        print(f"Driver unici per l'addestramento: {np.unique(groups[train_index])}")
        print(f"Driver unici per il test: {np.unique(groups[test_index])}")
        # Stampa i valori di 'driver' per le righe del set di addestramento e del set di test
        print("Driver nel set di addestramento:", df['driver'].iloc[train_index].unique())
        print("Driver nel set di test:", df['driver'].iloc[test_index].unique())
        # Stampa alcuni dei tuoi dati di addestramento e di test
        print("Dati di addestramento:", X_train.head())
        print("Dati di test:", X_test.head())

        for name, classifier in classifiers.items():
            pipe = Pipeline([('scaler', StandardScaler()), ('clf', classifier)])
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='f1_micro',  groups=groups[train_index])
            scores_dict[name] = scores
            print(f"Punteggi di validazione incrociata per {name}: {scores}")
            print(f"Media dei punteggi di validazione incrociata per {name}: {scores.mean()}")
            print(f"Deviazione standard dei punteggi di validazione incrociata per {name}: {np.std(scores)}")

    return scores_dict


'''leave one subject out (LOO)'''

def leave_one_subject_out_validation(X, y, groups, classifiers):
    loo = LeaveOneGroupOut()

    scores_dict = {}

    for train_index, test_index in loo.split(X, y, groups):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Stampa i valori di 'driver' per le righe del set di addestramento e del set di test
        print("Driver nel set di addestramento:", df['driver'].iloc[train_index].unique())
        print("Driver nel set di test:", df['driver'].iloc[test_index].unique())

        # Stampa alcuni dei tuoi dati di addestramento e di test
        print("Dati di addestramento:", X_train.head())
        print("Dati di test:", X_test.head())



        for name, classifier in classifiers.items():
            pipe = Pipeline([('scaler', StandardScaler()), ('clf', classifier)])
            scores = cross_val_score(pipe, X_train, y_train, cv=loo, scoring='f1_micro',  groups=groups[train_index])
            scores_dict[name] = scores
            print(f"Punteggi di validazione incrociata per {name}: {scores}")
            print(f"Media dei punteggi di validazione incrociata per {name}: {scores.mean()}")
            print(f"Deviazione standard dei punteggi di validazione incrociata per {name}: {np.std(scores)}")

    return scores_dict


#f1_weighted
#f1_micro

def leave_one_subject_out_validation_2(X, y, groups, classifiers):
    loo = LeaveOneGroupOut()

    scores_dict = {}

    for train_index, test_index in loo.split(X, y, groups):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Stampa i valori di 'driver' per le righe del set di addestramento e del set di test
        print("Driver nel set di addestramento:", df['driver'].iloc[train_index].unique())
        print("Driver nel set di test:", df['driver'].iloc[test_index].unique())

        # Stampa alcuni dei tuoi dati di addestramento e di test
        print("Dati di addestramento:", X_train.head())
        print("Dati di test:", X_test.head())

        for name, classifier in classifiers.items():
            # Reinizializza il tuo modello per ogni iterazione
            classifier = clone(classifier)

            pipe = Pipeline([('scaler', StandardScaler()), ('clf', classifier)])
            scores = cross_val_score(pipe, X_train, y_train, cv=loo, scoring='f1_micro',  groups=groups[train_index])
            scores_dict[name] = scores
            print(f"Punteggi di validazione incrociata per {name}: {scores}")
            print(f"Media dei punteggi di validazione incrociata per {name}: {scores.mean()}")
            print(f"Deviazione standard dei punteggi di validazione incrociata per {name}: {np.std(scores)}")

    return scores_dict



    '''for name, classifier in classifiers.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', classifier)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        score = f1_score(y_test, y_pred, average='micro')
        scores_dict[name] = score
        print(f"Punteggio F1 per {name}: {score}")

    return scores_dict'''


def leave_one_subject_out_single_driver_test(X, y, groups, classifiers, test_driver):
    loo = LeaveOneGroupOut()
    # Trova l'indice del driver di test
    test_index = np.where(groups == test_driver)
    train_index = np.where(groups != test_driver)

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Stampa i valori di 'driver' per le righe del set di addestramento e del set di test
    print("Driver nel set di addestramento:", df['driver'].iloc[train_index].unique())
    print("Driver nel set di test:", df['driver'].iloc[test_index].unique())

    scores_dict = {}
    for name, classifier in classifiers.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', classifier)])
        scores = cross_val_score(pipe, X_train, y_train, cv=loo, scoring='f1_micro', groups=groups[train_index])
        scores_dict[name] = scores
        print(f"Punteggi di validazione incrociata per {name}: {scores}")
        print(f"Media dei punteggi di validazione incrociata per {name}: {scores.mean()}")
        print(f"Deviazione standard dei punteggi di validazione incrociata per {name}: {np.std(scores)}")

    return scores_dict


'''creazione boxplot per loo e 5 fold cross validation'''
def create_signal_boxplot(groups):
    results = {}
    for signal_name, X_resampled in [('ALL', X), ('HR', X_HR), ('BR', X_BR), ('PP', X_PP),
                                    ('Palm_EDA', X_Palm_EDA), ('SPEED', X_SPEED), ('ACC', X_ACC),
                                     ('BRA', X_BRA), ('STEE', X_STEE)]:
        scores = leave_one_subject_out_validation(X_resampled, y, groups, classifiers)
        mean_scores = {name: np.mean(scores[name]) for name in classifiers.keys() if name in scores}
        results[signal_name] = mean_scores
    df_results = pd.DataFrame(results)

    # Crea il boxplot
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df_results)
    plt.title('leave one out')
    plt.ylabel('Punteggio loo')
    plt.xlabel('Segnale')
    plt.show()

    # Crea il boxplot
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df_results.T)
    plt.title('leave one out')
    plt.ylabel('Punteggio loo')
    plt.xlabel('Classificatore')
    plt.show()



'''matrice di confusione e divisione set dati StratifiedShuffleSplit '''

def matrice_di_confusione():
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=42)

    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Stampa i valori di 'driver' per le righe del set di addestramento e del set di test
        print("Driver nel set di addestramento:", df['driver'].iloc[train_index].unique())
        print("Driver nel set di test:", df['driver'].iloc[test_index].unique())



    f, axes = plt.subplots(1, 3, figsize=(20, 10), sharey='row')
    plt.subplots_adjust(wspace=0.40, hspace=0.1)

    for i, (key, classifier) in enumerate(classifiers.items()):
        pipe = Pipeline([('Scl', StandardScaler()), ('clf', classifier)])
        Y_pred = pipe.fit(X_train, y_train).predict(X_test)
        cf_matrix = metrics.confusion_matrix(y_test, Y_pred)
        disp = metrics.ConfusionMatrixDisplay(cf_matrix,
                                              display_labels=['NOSTR', 'STRC', 'STRE', 'STRS'])
        #display_labels = ['Rilassato', 'Stressato']) per class binaria
        # Cambia il colore della matrice di confusione
        disp.plot(cmap='coolwarm', ax=axes[i], xticks_rotation=45)
        # Cambia il titolo del grafico
        disp.ax_.set_title(key, fontsize=12, fontweight='bold')
        # Stampa il rapporto di classificazione
        print(key + " Rapporto: \n" + metrics.classification_report(y_test, Y_pred))
        # Calcola l'accuratezza del modello
        accuracy = pipe.score(X_test, y_test)
        # Rimuovi la barra dei colori
        disp.im_.colorbar.remove()
        # Imposta l'etichetta dell'asse x
        disp.ax_.set_xlabel('Accuracy: {:.00%}\n'.format(accuracy), fontsize=12)
        # Rimuovi l'etichetta dell'asse y per tutti tranne il primo grafico
        if i != 0:
            disp.ax_.set_ylabel('')
    # Aggiungi un'etichetta all'asse x del grafico
    f.text(0.4, 0.1, 'Classi previste', ha='left')
    # Mostra il grafico
    plt.show()


'''matrice di confusione e divisione dati con StratifiedShuffleSplit. Applicazione di SMOTE ai dati di train'''
def matrice_di_confusione_smote():
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=42)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Stampa i valori di 'driver' per le righe del set di addestramento e del set di test
        print("Driver nel set di addestramento:", df['driver'].iloc[train_index].unique())
        print("Driver nel set di test:", df['driver'].iloc[test_index].unique())

    # Applica SMOTE ai dati di addestramento
    over = SMOTE()
    X_resampled, y_resampled = over.fit_resample(X_train, y_train)

    # Stampa il conteggio dei valori per y_resampled
    print(y_resampled.value_counts())


    f, axes = plt.subplots(1, 3, figsize=(20, 10), sharey='row')
    plt.subplots_adjust(wspace=0.40, hspace=0.1)

    for i, (key, classifier) in enumerate(classifiers.items()):
        pipe = Pipeline([('Scl', StandardScaler()), ('clf', classifier)])
        Y_pred = pipe.fit(X_resampled, y_resampled).predict(X_test)
        cf_matrix = metrics.confusion_matrix(y_test, Y_pred)
        disp = metrics.ConfusionMatrixDisplay(cf_matrix,
                                              display_labels=['NOSTR', 'STRC', 'STRE', 'STRS'])
        # display_labels = ['Rilassato', 'Stressato']) per class binaria
        # Cambia il colore della matrice di confusione
        disp.plot(cmap='coolwarm', ax=axes[i], xticks_rotation=45)
        # Cambia il titolo del grafico
        disp.ax_.set_title(key, fontsize=12, fontweight='bold')
        # Stampa il rapporto di classificazione
        print(key + " Rapporto: \n" + metrics.classification_report(y_test, Y_pred))
        # Calcola l'accuratezza del modello
        accuracy = pipe.score(X_test, y_test)
        # Rimuovi la barra dei colori
        disp.im_.colorbar.remove()
        # Imposta l'etichetta dell'asse x
        disp.ax_.set_xlabel('Accuracy: {:.00%}\n'.format(accuracy), fontsize=12)
        # Rimuovi l'etichetta dell'asse y per tutti tranne il primo grafico
        if i != 0:
            disp.ax_.set_ylabel('')
    # Aggiungi un'etichetta all'asse x del grafico
    f.text(0.4, 0.1, 'Classi previste', ha='left')
    # Mostra il grafico
    plt.show()

'''analisi PCA varianza 95-99'''
# Funzioni PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def stampa_pca(X, y, n_components):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=36)
    X_pca = pca.fit_transform(X_scaled)

    map = pd.DataFrame(X_pca)



    corr = map.corr()
    print(corr)

    plt.figure(figsize=(20, 20))
    sns.heatmap(corr, cmap='Greens', annot=True, fmt=".2f")
    plt.title('Correlezione tra componenti principali')
    plt.show()

    plot = plt.scatter(X_pca[:, 1], X_pca[:, 0], c=y)
    plt.legend(handles=plot.legend_elements()[0], labels=['Non stressato', 'Stressato'])
    plt.xlabel('PC2')
    plt.ylabel('PC1')
    plt.show()



def calcolo_componenti_pca(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA().fit(X_scaled)

    plt.rcParams["figure.figsize"] = (20, 6)  # Aumenta la dimensione del grafico

    fig, ax = plt.subplots()
    xi = np.arange(1, min(61, X.shape[1] + 1), step=1)  # Usa il minimo tra 50 e il numero di colonne in X
    y = np.cumsum(pca.explained_variance_ratio_[:60])  # Considera solo le prime 50 componenti

    plt.ylim(0.0, 1.1)
    plt.plot(xi, y, marker='o', linestyle='--', color='b')

    plt.xlabel('Numero di Componenti')
    plt.xticks(np.arange(0, min(61, X.shape[1] + 1), step=1), rotation=90)  # Ruota le etichette di 90 gradi
    plt.ylabel('Varianza cumulativa (%)')
    plt.title('Il numero di componenti necessari per spiegare la varianza')

    plt.axhline(y=0.99, color='r', linestyle='-')
    plt.text(0.5, 0.92, '99% cut-off threshold', color='red', fontsize=16)

    ax.grid(axis='x')
    plt.show()



def feature_pca(X, n_components):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components).fit(X_scaled)

    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)

    plt.xlabel('PCA Feature')
    plt.ylabel('Explained Variance')
    plt.title('Feature Explained Variance')
    plt.show()

def matrice_di_confusione_pca(X, y):
    # Normalizza i dati
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Applica la PCA
    pca = PCA(n_components=60)
    X_pca = pca.fit_transform(X_scaled)

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=42)

    for train_index, test_index in sss.split(X_pca, y):
        X_train, X_test = X_pca[train_index], X_pca[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Stampa i valori di 'driver' per le righe del set di addestramento e del set di test
        print("Driver nel set di addestramento:", df['driver'].iloc[train_index].unique())
        print("Driver nel set di test:", df['driver'].iloc[test_index].unique())

    f, axes = plt.subplots(1, 3, figsize=(20, 10), sharey='row')
    plt.subplots_adjust(wspace=0.40, hspace=0.1)

    for i, (key, classifier) in enumerate(classifiers.items()):
        pipe = Pipeline([('Scl', StandardScaler()), ('clf', classifier)])
        Y_pred = pipe.fit(X_train, y_train).predict(X_test)
        cf_matrix = metrics.confusion_matrix(y_test, Y_pred)
        disp = metrics.ConfusionMatrixDisplay(cf_matrix,
                                              display_labels=['Rilassato', 'Stressato'])
        #display_labels = ['Rilassato', 'Stressato']) per class binaria
        # Cambia il colore della matrice di confusione
        disp.plot(cmap='coolwarm', ax=axes[i], xticks_rotation=45)
        # Cambia il titolo del grafico
        disp.ax_.set_title(key, fontsize=12, fontweight='bold')
        # Stampa il rapporto di classificazione
        print(key + " Rapporto: \n" + metrics.classification_report(y_test, Y_pred))
        # Calcola l'accuratezza del modello
        accuracy = pipe.score(X_test, y_test)
        # Rimuovi la barra dei colori
        disp.im_.colorbar.remove()
        # Imposta l'etichetta dell'asse x
        disp.ax_.set_xlabel('Accuracy: {:.00%}\n'.format(accuracy), fontsize=12)
        # Rimuovi l'etichetta dell'asse y per tutti tranne il primo grafico
        if i != 0:
            disp.ax_.set_ylabel('')
    # Aggiungi un'etichetta all'asse x del grafico
    f.text(0.4, 0.1, 'Classi previste', ha='left')
    # Mostra il grafico
    plt.show()

'''analisi dei driver con prestazioni migliori e peggiori'''
def driver_unici_punteggi():
    # Creare un DataFrame vuoto per memorizzare i risultati
    results = pd.DataFrame()

    for driver in df['driver'].unique():
        # Crea set di addestramento e di test
        train = df[df['driver'] != driver]
        test = df[df['driver'] == driver]

        X_train = train.drop(['driver', 'Time_Interval', 'stress'], axis=1)
        y_train = train['stress']
        X_test = test.drop(['driver', 'Time_Interval', 'stress'], axis=1)
        y_test = test['stress']

        for i, (key, classifier) in enumerate(classifiers.items()):
            pipe = Pipeline([('Scl', StandardScaler()), ('clf', classifier)])
            Y_pred = pipe.fit(X_train, y_train).predict(X_test)
            # Calcola l'accuratezza del modello
            accuracy = pipe.score(X_test, y_test)
            # Aggiungi l'accuratezza al DataFrame dei risultati
            results.loc[driver, key] = accuracy

        # Calcola la media delle accuratezze per ciascun driver
    results['Media'] = results.mean(axis=1)

    # Stampa il DataFrame dei risultati
    print(results)

    # Stampa i driver con le prestazioni medie migliori e peggiori
    print('Driver con le prestazioni medie migliori:', results['Media'].idxmax())
    print('Driver con le prestazioni medie peggiori:', results['Media'].idxmin())

def driver_migliore_e_peggiore():
    # Creare un DataFrame vuoto per memorizzare i risultati
    results = pd.DataFrame()

    for driver in ['T031', 'T060']:
        # Crea set di addestramento e di test
        train = df[df['driver'] != driver]
        test = df[df['driver'] == driver]

        X_train = train.drop(['driver', 'Time_Interval', 'stress'], axis=1)
        y_train = train['stress']
        X_test = test.drop(['driver', 'Time_Interval', 'stress'], axis=1)
        y_test = test['stress']

        for i, (key, classifier) in enumerate(classifiers.items()):
            pipe = Pipeline([('Scl', StandardScaler()), ('clf', classifier)])
            Y_pred = pipe.fit(X_train, y_train).predict(X_test)
            # Calcola l'accuratezza del modello
            accuracy = pipe.score(X_test, y_test)
            # Aggiungi l'accuratezza al DataFrame dei risultati
            results.loc[driver, key] = accuracy

            # Crea una matrice di confusione per il classificatore
            cm = confusion_matrix(y_test, Y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.title(f'Matrice di confusione per {key} su {driver}')
            plt.show()

    # Stampa il DataFrame dei risultati
    print(results)


'''ROC'''
def draw_roc_curve():
    ss = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=42)
    for train_index, test_index in ss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Stampa i valori di 'driver' per le righe del set di addestramento e del set di test
        print("Driver nel set di addestramento:", df['driver'].iloc[train_index].unique())
        print("Driver nel set di test:", df['driver'].iloc[test_index].unique())

    bayes = Pipeline([('scl', StandardScaler()), ('nb', classifiers_complete["Naive Bayes"])])
    logreg = Pipeline([('scl', StandardScaler()), ('lr', classifiers_complete["Logistic Regression"])])
    kn = Pipeline([('scl', StandardScaler()), ('kn', classifiers_complete["KN"])])
    svm = Pipeline([('scl', StandardScaler()), ('svm', classifiers_complete["SVM"])])
    dectree = Pipeline([('scl', StandardScaler()), ('dt', classifiers_complete["Decision Tree"])])
    randf = Pipeline([('scl', StandardScaler()), ('rf', classifiers_complete["Random Forest"])])

    bayes.fit(X_train, y_train)
    logreg.fit(X_train, y_train)
    kn.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    dectree.fit(X_train, y_train)
    randf.fit(X_train, y_train)

    pred_prob1 = bayes.predict_proba(X_test)
    pred_prob2 = logreg.predict_proba(X_test)
    pred_prob3 = kn.predict_proba(X_test)
    pred_prob4 = svm.predict_proba(X_test)
    pred_prob5 = dectree.predict_proba(X_test)
    pred_prob6 = randf.predict_proba(X_test)

    fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:, 1])
    fpr2, tpr2, thresh2 = roc_curve(y_test, pred_prob2[:, 1])
    fpr3, tpr3, thresh3 = roc_curve(y_test, pred_prob3[:, 1])
    fpr4, tpr4, thresh4 = roc_curve(y_test, pred_prob4[:, 1])
    fpr5, tpr5, thresh5 = roc_curve(y_test, pred_prob5[:, 1])
    fpr6, tpr6, thresh6 = roc_curve(y_test, pred_prob6[:, 1])

    random_probs = [0 for i in range(len(y_test))]
    p_fpr, p_tpr, _ = roc_curve(y_test, random_probs)

    auc_score1 = cross_val_score(bayes, X, y, cv=ss, n_jobs=-1, scoring='roc_auc').mean()
    auc_score2 = cross_val_score(logreg, X, y, cv=ss, n_jobs=-1, scoring='roc_auc').mean()
    auc_score3 = cross_val_score(kn, X, y, cv=ss, n_jobs=-1, scoring='roc_auc').mean()
    auc_score4 = cross_val_score(svm, X, y, cv=ss, n_jobs=-1, scoring='roc_auc').mean()
    auc_score5 = cross_val_score(dectree, X, y, cv=ss, n_jobs=-1, scoring='roc_auc').mean()
    auc_score6 = cross_val_score(randf, X, y, cv=ss, n_jobs=-1, scoring='roc_auc').mean()

    plt.plot(fpr1, tpr1, color='orange', label='Naive Bayes (AUC = {:.00%})'.format(auc_score1))
    plt.plot(fpr2, tpr2, color='green', label='Logistic Regression (AUC = {:.00%})'.format(auc_score2))
    plt.plot(fpr3, tpr3, color='red', label='KN (AUC = {:.00%})'.format(auc_score3))
    plt.plot(fpr4, tpr4, color='brown', label='SVM (AUC = {:.00%})'.format(auc_score4))
    plt.plot(fpr5, tpr5, color='black', label='Decision Tree (AUC = {:.00%})'.format(auc_score5))
    plt.plot(fpr6, tpr6, color='gray', label='Random Forest (AUC = {:.00%})'.format(auc_score6))

    plt.plot(p_fpr, p_tpr, color='blue')
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')

    plt.legend(loc='best')
    plt.savefig('ROC', dpi=300)
    plt.show()

def draw_roc_curve_2():
    ss = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=42)
    for train_index, test_index in ss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Stampa i valori di 'driver' per le righe del set di addestramento e del set di test
        print("Driver nel set di addestramento:", df['driver'].iloc[train_index].unique())
        print("Driver nel set di test:", df['driver'].iloc[test_index].unique())

    kn = Pipeline([('scl', StandardScaler()), ('kn', classifiers["KN semplice"])])
    svm = Pipeline([('scl', StandardScaler()), ('svm', classifiers["SVM semplice"])])
    an = Pipeline([('scl', StandardScaler()), ('an', classifiers["AN semplice"])])




    kn.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    an.fit(X_train, y_train)



    pred_prob1 = kn.predict_proba(X_test)
    pred_prob2 = svm.predict_proba(X_test)
    pred_prob3 = an.predict_proba(X_test)




    fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:, 1])
    fpr2, tpr2, thresh2 = roc_curve(y_test, pred_prob2[:, 1])
    fpr3, tpr3, thresh3 = roc_curve(y_test, pred_prob3[:, 1])




    random_probs = [0 for i in range(len(y_test))]
    p_fpr, p_tpr, _ = roc_curve(y_test, random_probs)

    auc_score1 = cross_val_score(kn, X, y, cv=ss, n_jobs=-1, scoring='roc_auc').mean()
    auc_score2 = cross_val_score(svm, X, y, cv=ss, n_jobs=-1, scoring='roc_auc').mean()
    auc_score3 = cross_val_score(an, X, y, cv=ss, n_jobs=-1, scoring='roc_auc').mean()




    plt.plot(fpr1, tpr1, color='orange', label='KN (AUC = {:.00%})'.format(auc_score1))
    plt.plot(fpr2, tpr2, color='green', label='SVM (AUC = {:.00%})'.format(auc_score2))
    plt.plot(fpr3, tpr3, color='red', label='AN (AUC = {:.00%})'.format(auc_score3))




    plt.plot(p_fpr, p_tpr, color='blue')
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')

    plt.legend(loc='best')
    plt.savefig('ROC', dpi=300)
    plt.show()



#create_signal_boxplot(groups)
#matrice_di_confusione_smote()
#matrice_di_confusione()
draw_roc_curve_2()


'''
# Testa singolarmente il driver x
test_driver = le.transform(['T060'])[0]
scores = single_driver_test(X, y, groups, classifiers, test_driver)'''





