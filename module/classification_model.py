# ==== IMPORTS ====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.callbacks import EarlyStopping

import xgboost as xgb
from skopt import BayesSearchCV

# ==== REGRESSIONE LOGISTICA =====
# Analisi dei coefficienti della Regressione Logistica
def log_reg_coefficients_summary(log_reg_X_train, y_train_smote): 
    # Aggiungo una colonna di bias per l'intercetta
    X_train_with_intercept = sm.add_constant(log_reg_X_train)

    # Inizializzo il modello
    model = sm.Logit(y_train_smote, X_train_with_intercept)
    result = model.fit()

    print(result.summary())

    return result

# Creazione di un dataframe che contiene i coefficienti della Regressione Logistica, funzionale alla visualizzazione grafica
def log_reg_coefficients_df(result):
# Creo un dataframe con coefficienti, p-value ed errori standard
    model_summary_df = pd.DataFrame({
        "Coefficiente": result.params,
        "P-value": result.pvalues,
        "Errore Standard": result.bse,
        "Intervallo di Confidenza Inferiore": result.conf_int()[0],
        "Intervallo di Confidenza Superiore": result.conf_int()[1]
    })

    model_summary_df = model_summary_df.reset_index().rename(columns = {'index':'Features'})

    return model_summary_df

# K-NEAREST NEIGHBORS
# Ottimizzazione degli iperparametri di KNN con GridSearchCV
def tune_knn_hyperparameters(model, model_X_train, y_train_smote):
    # Configuro la griglia di ricerca con Cross-validation
    param_grid = {
        'n_neighbors': np.arange(1, round(np.sqrt(len(model_X_train)))),
        'metric': ['manhattan']
        }

    model_cv = GridSearchCV(model, param_grid, cv=5)

    # Eseguo Gird Search per trovare la configurazione ottimale del modello
    model_cv.fit(model_X_train, y_train_smote)

    # Stampo gli iperparametri e la loro accuracy
    print("Best K:", model_cv.best_params_['n_neighbors'])
    print("Best Metric:", model_cv.best_params_['metric'])
    print("Best Accuracy:", model_cv.best_score_)

    return model_cv

# ==== DECISION TREE ====
# Ottimizzazione della potatura dell'albero decisionale con cross-validation (Post-pruning x K-Fold Cross-validation)
def tune_decision_tree_pruning(dt_model, dt_X_train, y_train_smote):
    path = dt_model.cost_complexity_pruning_path(dt_X_train, y_train_smote)
    ccp_alphas = path.ccp_alphas

    best_model = None
    best_score = 0
    for ccp_alpha in ccp_alphas:
        model_temp = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
        scores = cross_val_score(model_temp, dt_X_train, y_train_smote, cv=5)
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            best_model = model_temp

    print(f'Best cross-validated accuracy: {best_score:.2f}')

    return best_score, best_model

# ==== RANDOM FORESTS =====
# Ottimizzazione degli iperparametri di RF con GridSearchCV
def tune_rf_hyperparameters(model, model_X_train, y_train_smote):
    # Configuro la griglia di ricerca con Cross-validation
    params_grid = {
            'n_estimators' : [100,350,500],
            'max_features' : ['log2','sqrt'],
            'min_samples_leaf' : [2,10,30]         
    }

    model_cv = GridSearchCV(estimator=model,
                       param_grid=params_grid,
                       scoring='accuracy',
                       cv=5,
                       # verbose=1,
                       n_jobs=-1)


    # Eseguo Gird Search per trovare la configurazione ottimale del modello
    model_cv.fit(model_X_train, y_train_smote)

    # Stampo gli iperparametri e la loro accuracy
    print("Best N. Estimators:", model_cv.best_params_['n_estimators'])
    print("Best Max Features:", model_cv.best_params_['max_features'])
    print("Best Min Samples Leaf:", model_cv.best_params_['min_samples_leaf'])
    print("Best Accuracy:", model_cv.best_score_)

    return model_cv

# ==== TREE CLASSIFIER =====
# Creazione del dataframe con le Features e la relativa importanza, funzionale alla visualizzazione grafica
def feature_importance_df(model, model_X_train):

    # Salvo le importanze associate alle Features
    importances = model.feature_importances_

    # Inizializzo e riempio il dataset
    feature_importance_df = pd.DataFrame({'Feature': model_X_train.columns, 'Importance': importances})

    # Ordino il dataset in ordine decrescente in base all'Importanza delle Features
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    return feature_importance_df

# ==== MATRICE DI CONFUSIONE =====
# Analisi dei coefficienti della matrice di confusione
def conf_matrix_coefficients_analysis(y_test, y_pred, conf_matrix):
    # Precisione
    precision = precision_score(y_test, y_pred)

    # Recall
    recall = recall_score(y_test, y_pred)

    # F1-score
    f1 = f1_score(y_test, y_pred)

    # Specificità
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp)

    print(f'Precisione: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-score: {f1}')
    print(f'Specificità: {specificity}')

    return precision, recall, f1, tn, fp, fn, tp, specificity

# Creazione del dataframe che contiene le metriche della matrice di confusione, funzionale alla visualizzazione grafica
def conf_matrix_coefficients_to_df(precision, recall, f1, specificity):
    # Inizializzo un dizionario che contiene i valori del dataframe
    conf_matrix_coefficients_data = {
        'Metriche': ['Precisione', 'Recall', 'F1-score', 'Specificità'], 
        'Valore': [precision, recall, f1, specificity]
    }

    # Trasformo il dizionario in un dataframe
    conf_matrix_coefficients_df = pd.DataFrame(conf_matrix_coefficients_data)

    return conf_matrix_coefficients_df

# ==== TEST ACCURACY ====
# Calcolo dell'accuracy del modello sui dati di Test
def model_accuracy(model, model_X_train, X_test, y_train_smote, y_test):
    model.fit(model_X_train, y_train_smote)

    y_pred = model.predict(X_test)

    y_proba = model.predict_proba(X_test)[:, 1]

    test_accuracy = accuracy_score(y_test, y_pred)

    print("Test Accuracy:", test_accuracy)

    return y_pred, y_proba, test_accuracy

# ==== AUC-ROC E INDICE DI GINI DEL MODELLO ====
# Calcolo dell'AUC-ROC e dell'Indice di Gini del modello
def model_auc_roc_gini(model_name, y_pred, y_test):

    auc = roc_auc_score(y_test, y_pred)
    gini = 2 * auc - 1 # 

    print(f"{model_name} - AUC-ROC: {auc}")
    print(f"{model_name} - Indice di Gini: {gini}") 

    return auc, gini

# ==== CONFRONTRO DEI MODELLI ====
# Creazione di un dataset che contiene i Test accuracy, Recall, F1-score, Specificità, AUC-ROC, Indice di Gini dei modeli, funzionale alla visualizzazione grafica
def model_metrics_df(model, test_accuracy, recall, f1, precision, auc, gini):
    # Inizializzo un dataset vuoto che contiene il modello e le relative metriche
    model_comparison_df = pd.DataFrame(columns=[
            'Model', 
            'Test Accuracy',
            'Recall',
            'F1-score',
            'Precision',
            'AUC-ROC',
            'Indice di Gini'
            ])
    
    for i in range (0,len(model)):
        model_comparison_df.loc[i] = [
            model[i],
            test_accuracy[i],
            recall[i],
            f1[i],
            precision[i],
            auc[i],
            gini[i]
        ]

    model_comparison_df_long = pd.melt(
    model_comparison_df,
    id_vars=['Model'], 
    var_name='Metric', 
    value_name='Value' 
    )

    return model_comparison_df_long

# ==== PRINCIPAL COMPONENT ANALYSIS =====
# Esecuzione della PCA
def pca_transformation(X_train_standard, X_test_standard):
    pca = PCA(n_components=0.75) # Mantengo il 75% della varianza
    pca_X_train = pca.fit_transform(X_train_standard)

    # Trasformo il test set con lo stesso modello PCA
    pca_X_test = pca.transform(X_test_standard)

    # Verifico il numero di componenti principali
    print(f"Numero di componenti principali selezionati: {pca.n_components_}")
    print(f"Varianza spiegata da ciascuna componente: {pca.explained_variance_ratio_}")
    print(f"Varianza totale spiegata: {sum(pca.explained_variance_ratio_):.2f}")

    return pca, pca_X_train, pca_X_test

# Nome delle componenti principali
def print_pca_features(pca, X_train_standard):
    # Estraggo i nomi delle Feature per ciascuna componente principale
    feature_names = X_train_standard.columns  # Nomi delle Feature originali
    components = pca.components_  # Componenti principali

    # Per ogni componente principale, estraggo e stampo le Features
    for i, component in enumerate(components):
        print(f"\nComponente Principale {i+1}:")
        # Associo i nomi delle Features al contributo della componente
        feature_contributions = list(zip(feature_names, component))
        # Ordino per valore assoluto decrescente per evidenziare le features più importanti
        feature_contributions = sorted(feature_contributions, key=lambda x: abs(x[1]), reverse=True)
        
        # Stampo le Features con il rispettivo contributo
        for feature, contribution in feature_contributions:
            print(f"{feature}: {contribution:.4f}")

# ==== ANN - MLP =====
# Configurazione del modello
def set_ann(ann_X_train, y_train_smote, optimizer='adam', neurons=128):

    # Definisco il modello
    model = Sequential([
        Dense(neurons, activation='relu', input_shape=(ann_X_train.shape[1],)),
        Dense(1, activation='sigmoid')  # Output layer per classificazione binaria
    ])

    # Definisco le metriche da monitorare
    metrics = ['accuracy', Precision(name="precision"), Recall(name="recall"), AUC(name="auc")]

    # Compilo il modello
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)

    # Callback per l'early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Addestramento del modello
    history = model.fit(
        ann_X_train, y_train_smote,
        validation_split=0.2,
        epochs=100,  # Numero massimo di epoche
        batch_size=32,
        verbose=1,
        callbacks=[early_stopping]
    )   

    return model, history

# Calcolo dell'accuracy del modello
def ann_accuracy(model, X_test, y_test):

    y_proba = model.predict(X_test)
    y_pred = (y_proba > 0.4).astype(int)

    test_accuracy = accuracy_score(y_test, y_pred)

    print("Test Accuracy:", test_accuracy)

    return y_proba, y_pred, test_accuracy

# ==== ANN -XGBoost =====
# Configurazione del modello
def set_xgb(ann_X_train, y_train_smote):

    # Definisco il modello
    xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',  # Per classificazione binaria
    eval_metric='logloss',  # Valutazione del modello basata sulla log loss
    scale_pos_weight=1,  # Usa questo parametro per gestire un dataset sbilanciato, se necessario
    use_label_encoder=False  # Evita avvisi di deprecazione
    )   

    # Addestro il modello
    xgb_model.fit(ann_X_train, y_train_smote)

    return xgb_model

# Tuning degli iperparameteri di XGBoost
def tune_xgb_hyperparameters(xgb_model, ann_X_train, y_train_smote):

    # Definisco la griglia di iperparametri per l'ottimizzazione bayesiana
    param_space = {
        'max_depth': (3, 10),
        'learning_rate': (0.01, 0.3, 'uniform'),
        'n_estimators': (50, 500),
        'subsample': (0.5, 1.0, 'uniform'),
        'colsample_bytree': (0.5, 1.0, 'uniform'),
        'gamma': (0, 5),
        'reg_alpha': (0, 1),
        'reg_lambda': (0, 1)
    }
    # Configuro l'ottimizzazione bayesiana con cross-validation
    model_cv = BayesSearchCV(
        xgb_model,
        param_space,
        n_iter=50,  # Numero di iterazioni
        cv=5,  # Cross-validation a 5 fold
        n_jobs=-1,  # Utilizzo di tutte le CPU disponibili
    )

    # Eseguo la ricerca bayesiana
    model_cv.fit(ann_X_train, y_train_smote)

    # Stampo gli iperparametri ottimali e la loro accuratezza
    print("Best Parameters:", model_cv.best_params_)
    print("Best Accuracy:", model_cv.best_score_)

    return model_cv
