import pandas as pd
import numpy as np
import argparse
import pickle
from sklearn.metrics import confusion_matrix


parser = argparse.ArgumentParser()
parser.add_argument("--file", default='data_test_feat.csv', help="This is the test file")
parser.add_argument("--to", default='answers_test.csv', help="Save to this filename")
parser.add_argument("--threshold", default='0', help="This is threshold")
parser.add_argument("--pproba", default='0', help="This is activate predict_proba")
args = parser.parse_args()


def get_predict(file, to_file, threshold, predict_proba):
    df_test = pd.read_csv(file)

    df_predict = pd.DataFrame()
    df_predict['id'] = df_test['id']
    df_predict['buy_time'] = df_test['buy_time']
    df_predict['vas_id'] = df_test['vas_id']

    pkl_filename = "./pickle_model.pkl"
    with open(pkl_filename, 'rb') as pkl_file:
        pickle_model = pickle.load(pkl_file)

    df_test.drop(['id', 'buy_time'], axis=1, inplace=True)
    Y_predict = pickle_model.predict_proba(df_test)[:, 1]
    if predict_proba != '1':
        Y_predict = [(x > float(threshold))*1 for x in Y_predict]
    df_predict['target'] = Y_predict
    df_predict.to_csv(to_file)
    print( f"Прогноз модели сохранен в файл {to_file}")
    print(df_predict.shape)


def compare_threshold(single_client_revenue, per_client_costs):
    Y_test = pd.read_csv("Y_test.csv")
    preds = pd.read_csv("preds_test.csv")
    compare_thr = pd.DataFrame(columns=['Threshold', 'Recall', 'Formula', 'Profit'])
    for x in range(10, 90, 1):
        cnf_matrix = confusion_matrix(Y_test, preds>x/100)
        TN = cnf_matrix[0][0]
        FN = cnf_matrix[1][0]
        TP = cnf_matrix[1][1]
        FP = cnf_matrix[0][1]

        TPR = round(TP/(TP+FN), 2)
        ALL = TP + FP

        formula = f'Profit = {TP} * SCR - {ALL} * PCC'

        profit = TP * single_client_revenue - ALL * per_client_costs

        compare_thr = compare_thr.append({'Threshold': x/100, 'Recall': TPR, 'Formula': formula, 'Profit': profit}, ignore_index=True)

    max_p = np.argmax(compare_thr['Profit'])

    max_profit = int(compare_thr['Profit'][max_p])
    best_threshold = compare_thr['Threshold'][max_p]
    recall = compare_thr['Recall'][max_p]

    return compare_thr, max_profit, best_threshold, recall


if __name__ == "__main__":
    print("Старт...")
    if args.threshold == '0' and args.pproba != '1':
        SCR = input(f'Введите предполагаемый ДОХОД ОТ ОДНОГО пользователя при подключении услуги: ')
        PCC = input(f'Введите предполагаемый РАСХОД на отправку ОДНОГО предложения: ')
        try:
            SCR = float(SCR)
            PCC = float(PCC)
        except ValueError:
            exit('Одно из введенных значений не является числом! Запустите команду заново.')
        print('Ожидайте...')
        print(f'Идет расчет оптимального значения threshold для максимизации прибыли при доходе от одного клиента равному {SCR} у.е. '
              f'и расходе на доставку предложения одному клиенту равному {PCC} у.е.')
        compare_thr, max_profit, best_threshold, recall = compare_threshold(SCR, PCC)

        if (max_profit < 0):
            print(f'(!) При доходе от одного клиента равному {SCR} у.е. '
                  f'и расходе на доставку предложения одному клиенту равному {PCC} у.е.'
                  f'данная активность убыточна для бизнеса!')
            print(f'Попробуйте изменить входные условия. Построение прогноза отменено!')
        else:
            print('best_threshold:', best_threshold)
            print('max_profit:', max_profit)
            print('recall:', recall)
            print(f'Применяем оптимальное значение threshold={best_threshold} для прогноза на новых данных')
            get_predict(args.file, args.to, best_threshold, args.pproba)
            print( f"Готово!")
    else:
        if args.pproba != '1':
            print(f'Внимание! Производится расчет с параметром threshold, указанным вручную. Его значение может быть не оптимальным.')
        else:
            print(f'Производится расчет вероятностей.')
        get_predict(args.file, args.to, args.threshold, args.pproba)
        print( f"Готово!")
