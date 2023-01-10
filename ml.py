import pandas as pd
import numpy as np

path = "./tic_tac_toe.csv"
import pandas as pd
# df = pd.read_csv("./tic-tac-toe.txt",delimiter=",")
# df.to_csv("tic_tac_toe.csv", encoding='utf-8', index=False)

df = pd.read_csv(path, header=0, names=['A','B','C','D','E','F','G','H','I','J'])

X = pd.get_dummies(df[['A','B','C','D','E','F','G','H','I','J']])       # 這行要看
# print(X)

print(df["J"].value_counts())

df.replace('negative',0,inplace=True)   
df.replace('positive',1,inplace=True)   



from sklearn import preprocessing

        # iloc 跟 loc 's different?

Y = df.iloc[:, [9]].values

from sklearn.model_selection import train_test_split
                                                        # 有 random_state和 shuffle參數可以使用
                                                        # stratify (?) 按照比例分配是什麼意思 , stratify=Y, random_state=1, shuffle=True, 
x_train, x_test, y_train, y_test = train_test_split( 
    X, Y, test_size= 0.1, stratify=Y
)

from sklearn.preprocessing import StandardScaler, MinMaxScaler


mms = MinMaxScaler()           
mms.fit(x_train)
x_train_std = mms.fit_transform(x_train)   
x_test_std = mms.fit_transform(x_test)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion="entropy", random_state=1 ) # What is those parameter mean?
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print(y_pred)
accuracy = classifier.score(x_test, y_test)
print(f"The accuracy is {accuracy} ")
