import pandas as pd
import numpy as np

path = "./tic_tac_toe.csv"
import pandas as pd
# df = pd.read_csv("./tic-tac-toe.txt",delimiter=",")
# df.to_csv("tic_tac_toe.csv", encoding='utf-8', index=False)

df = pd.read_csv(path, header=0, names=['A','B','C','D','E','F','G','H','I','J'])
print(df)
# print(df.head())
# for i in df.columns:
#   print(df[i].value_counts())
#   print()

X = pd.get_dummies(df[['A','B','C','D','E','F','G','H','I','J']])
print(X)
df.replace('negative',0,inplace=True)   
df.replace('positive',1,inplace=True)   

from sklearn import preprocessing

# iloc 跟 loc 's different?
X = df.iloc[:,0:9].values
Y = df.iloc[:, [9]].values
# print(X, end="\n")
# print(Y)

from sklearn.model_selection import train_test_split
                                                        # 有 random_state和 shuffle參數可以使用
                                                        # stratify (?) 按照比例分配是什麼意思 , stratify=Y, random_state=1, shuffle=True, 
x_train, x_test, y_train, y_test = train_test_split( 
    X, Y, test_size= 0.1
)

from sklearn.preprocessing import StandardScaler
# le = preprocessing.LabelEncoder()
# X_train = le.fit_transform(x_train)
# X_test = le.fit_transform(x_test)
# print(X_train)

sc = StandardScaler()           # What is StandardScaler? Why use this?
sc.fit(x_train)
x_train_std = sc.fit_transform(x_train)    # fit_transform?
x_test_std = sc.fit_transform(x_test)
print(x_train_std)

# from sklearn.tree import DecisionTreeClassifier
# classifier = DecisionTreeClassifier(criterion="entropy", random_state=1 ) # What is those parameter mean?
# classifier.fit(x_train, y_train)

# y_pred = classifier.predict(x_test)
# print(y_pred)
