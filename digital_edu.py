import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix 
df = pd.read_csv('train.csv')
print(df.tail())
df.drop(['id','has_photo','has_mobile','followers_count','graduation','relation','education_status','langs','life_main','people_main','city','last_seen','occupation_type','occupation_name','career_start','career_end'], axis = 1,inplace = True)
def data(year):
    if len(str(year))<8 or pd.isnull(year):
        return 1990
    return int(year[-4:])
print(df['education_form'].value_counts())
df['bdate']= df['bdate'].apply(data)
df[list(pd.get_dummies(df['education_form']).columns)] = pd.get_dummies(df['education_form'])
df['education_form'].fillna('Full-time', inplace = True)
df.drop('education_form', axis = 1, inplace = True)
X = df.drop('result',axis = 1)
Y = df['result']
sc = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(X , Y, test_size = 0.25)
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
classifier = KNeighborsClassifier(n_neighbors = 103)
classifier.fit(x_train, y_train)
pred = classifier.predict(x_test)
print(accuracy_score(y_test, pred) * 100)