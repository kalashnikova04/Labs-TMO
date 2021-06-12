import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

import seaborn as sns
import matplotlib.pyplot as plt


@st.cache(allow_output_mutation=True)
def load_data():
    '''
    Загрузка данных
    '''
    data = pd.read_csv("weatherAUS.csv", sep=',')
    return data


@st.cache(allow_output_mutation=True)
def preprocess_data(data_in):
    '''
    Масштабирование признаков, функция возвращает X и y для кросс-валидации
    '''
    data = data_in.copy()
    data = data.drop(['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm'], axis=1)
    data = data.dropna(axis=0, how='any')
    # кодировка признаков бинарной классификации
    le = LabelEncoder()
    cat_le = ['RainToday', 'RainTomorrow']
    for i in cat_le:
        data.loc[:, i] = le.fit_transform(data[i])
    data = data.drop(['Date', 'WindDir3pm','WindDir9am', 'WindGustDir', 'Location'], axis = 1)

    # МАСШТАБИРОВАНИЕ
    sc1 = MinMaxScaler()
    data[:] = sc1.fit_transform(data)

    return data[:]


class MetricLogger:

    def __init__(self):
        self.df = pd.DataFrame(
            {'metric': pd.Series([], dtype='str'),
             'alg': pd.Series([], dtype='str'),
             'value': pd.Series([], dtype='float')})

    def add(self, metric, alg, value):
        """
        Добавление значения
        """
        # Удаление значения если оно уже было ранее добавлено
        self.df.drop(self.df[(self.df['metric'] == metric) & (self.df['alg'] == alg)].index, inplace=True)
        # Добавление нового значения
        temp = [{'metric': metric, 'alg': alg, 'value': value}]
        self.df = self.df.append(temp, ignore_index=True)

    def get_data_for_metric(self, metric, ascending=True):
        """
        Формирование данных с фильтром по метрике
        """
        temp_data = self.df[self.df['metric'] == metric]
        temp_data_2 = temp_data.sort_values(by='value', ascending=ascending)
        return temp_data_2['alg'].values, temp_data_2['value'].values

    def plot(self, str_header, metric, ascending=True, figsize=(5, 5)):
        """
        Вывод графика
        """
        array_labels, array_metric = self.get_data_for_metric(metric, ascending)
        fig, ax1 = plt.subplots(figsize=figsize)
        pos = np.arange(len(array_metric))
        rects = ax1.barh(pos, array_metric,
                         align='center',
                         height=0.5,
                         tick_label=array_labels)
        ax1.set_title(str_header)
        for a, b in zip(pos, array_metric):
            plt.text(0.3, a - 0.1, str(round(b, 3)), color='white')
        plt.show()


# Отрисовка ROC-кривой
def draw_roc_curve(y_true, y_score, ax, pos_label=1, average='micro'):
    fpr, tpr, thresholds = roc_curve(y_true, y_score,
                                     pos_label=pos_label)
    roc_auc_value = roc_auc_score(y_true, y_score, average=average)
    #plt.figure()
    lw = 2
    ax.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_value)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_xlim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")


models_list1 = ['LogR', 'KNN_5', 'Tree', 'RF', 'GB']
models_list2 = ['LogR_grid', 'KNN_grid', 'Tree_grid', 'RF_grid', 'GB_grid']
clas_models = {'LogR': LogisticRegression(),
               'KNN_5': KNeighborsClassifier(n_neighbors=5),
               'Tree': DecisionTreeClassifier(),
               'RF': RandomForestClassifier(),
               'GB': GradientBoostingClassifier(),
               }


def train_model(models_select,  clas, X_train, X_test, y_train, y_test, clasMetricLogger):
    current_models_list = []
    roc_auc_list = []
    for model_name in models_select:
        model = clas[model_name]
        model.fit(X_train, y_train)
    # Предсказание значений
        Y_pred = model.predict(X_test)
    # Предсказание вероятности класса "1" для roc auc
        Y_pred_proba_temp = model.predict_proba(X_test)
        Y_pred_proba = Y_pred_proba_temp[:, 1]

        precision = precision_score(y_test.values, Y_pred)
        recall = recall_score(y_test.values, Y_pred)
        f1 = f1_score(y_test.values, Y_pred)
        roc_auc = roc_auc_score(y_test.values, Y_pred_proba)

        clasMetricLogger.add('precision', model_name, precision)
        clasMetricLogger.add('recall', model_name, recall)
        clasMetricLogger.add('f1', model_name, f1)
        clasMetricLogger.add('roc_auc', model_name, roc_auc)

        current_models_list.append(model_name)
        roc_auc_list.append(roc_auc)

        fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
        draw_roc_curve(y_test.values, Y_pred_proba, ax[0])
        plot_confusion_matrix(model, X_test, y_test.values, ax=ax[1], display_labels=['0', '1'],
                              cmap=plt.cm.Blues, normalize='true')
        fig.suptitle(model_name)
        st.pyplot(fig)

    #    if len(roc_auc_list)>1:
    #        ra = {'roc-auc': roc_auc_list}
    #        df_ra = pd.DataFrame(data=ra, index=current_models_list)
    #        st.bar_chart(df_ra)


st.sidebar.header('Модели машинного обучения')
models_select = st.sidebar.multiselect('Выберите модели для обучения с заданными гиперпараметрами:', models_list1)
models_select2 = st.sidebar.multiselect('Выберите модели для обучения с побором гиперпараметров:', models_list2)

data = load_data()

if st.checkbox('Показать основные характеристики датасета'):
    st.subheader('Первые пять строк датасета:')
    st.dataframe(data=data.head(), width=2000, height=1500)
    st.write('Размер набора данных: {}'.format(data.shape))
    st.subheader('Типы данных в колонках:')
    st.dataframe(data.dtypes)
    st.write('Количество пропущенных значений по колонкам:')
    st.dataframe(data.isnull().sum())

# Числовые колонки для масштабирования
scale_cols = ['MinTemp','MaxTemp','Rainfall','WindGustSpeed','WindSpeed9am','WindSpeed3pm',
              'Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Temp9am','Temp3pm']

data = preprocess_data(data)

x = data[['Rainfall', 'WindGustSpeed', 'Humidity9am', 'Humidity3pm', 'RainToday']]
y = data['RainTomorrow']
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.3, random_state=1)

if st.checkbox('Показать распределение признаков'):

    st.subheader('Отмасштабированные признаки:')
    for col in scale_cols:
        fig, ax = plt.subplots()
        ax.hist(data[col], 50)
        ax.title.set_text(col)
        st.pyplot(fig)

if st.checkbox('Показать корреляционную матрицу'):
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(data.corr(), annot=True, fmt='.2f')
    st.pyplot(fig)
    st.write('По построенной корреляционной матрице выбираем 5 признаков для построения моделей: Rainfall, '
             'WindGustSpeed, Humidity9am, Humidity3pm, RainToday. Эти признаки имеют большую корреляцию с '
             'целевым признаком (RainTomorrow) чем остальные.')

metric = MetricLogger()

if st.checkbox('Показать модели без подбора гиперпараметров'):
    train_model(models_select, clas_models, X_train, X_test, y_train, y_test, metric)


if st.checkbox('Показать модели с подбором гиперпараметров'):
    # ПОДБОР ГИПЕРПАРАМЕТРОВ

    # Логическая регрессия
    params = {'C': np.logspace(-2, 2, 40)}
    grid_log = GridSearchCV(LogisticRegression(max_iter=30000), params, cv=5, scoring='roc_auc')
    grid_log.fit(X_train.head(25000), y_train.head(25000))

    # Метод ближ соседей
    param_knn = range(100, 500, 25)
    n_range = np.array(param_knn)
    tuned_params = [{'n_neighbors': n_range}]
    grid_knn = GridSearchCV(KNeighborsClassifier(), tuned_params, cv=StratifiedKFold(n_splits=5),
                            scoring='roc_auc')
    grid_knn.fit(X_train.head(25000), y_train.head(25000))

    # Дерево решений
    param_dt = [{'max_depth': range(1, 10),
                 'min_samples_split': [2, 5, 7, 10],
                 'min_samples_leaf': range(1, 5),
                 'max_features': range(1, 5)
                 }]
    grid_dt = GridSearchCV(DecisionTreeClassifier(), param_dt, cv=StratifiedKFold(n_splits=5), scoring='roc_auc')
    grid_dt.fit(X_train.head(25000), y_train.head(25000))

    # Случайный лес
    param_rf = [{'max_depth': range(1, 10),
                 'min_samples_split': [2, 5, 10],
                 'max_features': range(1, 5)
                 }]
    grid_rf = RandomizedSearchCV(RandomForestClassifier(), param_rf, cv=5, scoring='roc_auc')
    grid_rf.fit(X_train.head(10000), y_train.head(10000))

    # Градиентный бустинг
    param_gb = [{'n_estimators': range(5, 200, 25),
                 'max_features': range(1, 5)}]
    grid_gb = GridSearchCV(GradientBoostingClassifier(), param_gb, cv=StratifiedKFold(n_splits=5), scoring='roc_auc')
    grid_gb.fit(X_train.head(20000), y_train.head(20000))

    d_train = lgb.LGBMClassifier()
    lgbm_params = {'learning_rate': [0.01, 0.05, 0.001],
                   'boosting_type': ['dart', 'gbdt'],
                   'objective': ['binary'],
                   'metric': ['auc'],
                   'num_leaves': [50, 70, 100, 150],
                   'max_depth': [5, 6, 7, 8]}
    grid_lgb = RandomizedSearchCV(d_train, lgbm_params, verbose=1, n_jobs=-1, cv=10, scoring='roc_auc')
    grid_lgb.fit(X_train.head(10000), y_train.head(10000))

    clas_models_grid = {'LogR_grid': grid_log.best_estimator_,
                        'KNN_grid': grid_knn.best_estimator_,
                        'Tree_grid': grid_dt.best_estimator_,
                        'RF_grid': grid_rf.best_estimator_,
                        'GB_grid': grid_lgb.best_estimator_}

    train_model(models_select2, clas_models_grid, X_train, X_test, y_train, y_test, metric)

if st.checkbox('Показать сравнение метрик'):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    clas_metrics = metric.df['metric'].unique()
    for i in clas_metrics:
        st.pyplot(metric.plot('Метрика: ' + i, i, figsize=(7, 6)))
        st.set_option('deprecation.showPyplotGlobalUse', False)

