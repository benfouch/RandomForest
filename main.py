from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import datasets 



'''
This all was from before when I was using a data set that was non numeric
It all worked well, but the dataset had other limiting flaws so I switched to
the iris set
'''
# def get_data(data_path:str):
#     df = pd.read_csv(data_path)
#     return df

# def make_numeric(series):
#     dict = {}
#     output = series.apply(lambda element : process_element(element, dict))
#     return output

# def process_element(element, dict):
#     if not element in dict.keys():
#         dict[element] = len(dict.keys())
#     return dict[element]

# def clean(df):
#     data = df.apply(make_numeric, axis=1)
#     return data

def main():
    X, y = datasets.load_iris( return_X_y = True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

    model = RandomForestClassifier(n_estimators=500)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print(confusion_matrix(y_test, pred))
    print('acc: ' + str(accuracy_score(y_test, pred)))

if (__name__):
    main()
