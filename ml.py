import pandas as pd

path = "./tic-tac-toe.txt"
df = pd.read_csv(path,
                 header=None,
                 encoding="utf-8"
)
# print(df.head())


df.replace('negative',0,inplace=True)   
df.replace('positive',1,inplace=True)   

from sklearn import preprocessing

# iloc 跟 loc 's different?
X = df.iloc[:,0:9].values

Y = df.iloc[:, [9]].values
# print(X, end="\n")
print(Y)

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
