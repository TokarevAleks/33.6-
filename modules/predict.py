
import dill
import json
import os
import glob
import pandas as pd

def load_test_data(test_data_dir):
    #Загрузка тестовых данных из указанной директории и формирование списка словарей с данными по каждому id.
    test_data = []
    for filename in os.listdir(test_data_dir):
        if filename.endswith('.json'):
            with open(os.path.join(test_data_dir, filename), 'r') as file:
                data = json.load(file)
                test_data.append(data)
    return test_data

def make_predictions(model, df):
    ## Создание нового датафрейма предсказаний с использованием загруженной модели для указанных тестовых данных.
    predictions_df = pd.DataFrame()

    X = df  # df - это тестовый датасет, который нужно использовать для предсказаний

    # Получаем предсказание для текущей строки
    prediction = model.predict(X)

    # Добавляем предсказание в новый датафрейм
    predictions_df['id'] = df['id']
    predictions_df['prediction'] = prediction
    predictions_df['price'] = df['price']

    return predictions_df

def save_predictions(predictions_df, output_dir):

    #Сохранение предсказаний в формате CSV в указанной директории.
    predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)


def predict():
    # Пути к директориям
    path = os.path.expanduser('~/airflow_hw')  # 'это путь до папки проекта
    model_dir = path + '/data/models'
    test_data_dir = path + '/data/test'
    output_dir = path + '/data/predictions'

    # Получаем список всех файлов с расширением pkl в указанной директории
    files = glob.glob(f"{model_dir}/cars_pipe_*.pkl")

    # Сортируем файлы по дате изменения (по убыванию), чтобы получить последний сохраненный файл
    latest_file = max(files, key=os.path.getctime)

    # Выводим на экран путь к последнему сохраненному файлу
    print("\nПоследний сохраненный файл:")
    print(latest_file)

    # Загружаем последней обученной модели
    with open(latest_file, 'rb') as file:
        model = dill.load(file)

    # Загрузка тестовых данных в датафрейм
    test_data = load_test_data(test_data_dir)
    df = pd.DataFrame(test_data)
    #print(df)

    # Создание предсказаний
    predictions_df = make_predictions(model, df)
    print(predictions_df)

    # Сохранение предсказаний
    save_predictions(predictions_df, output_dir)


if __name__ == '__main__':
    predict()
