#kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

#kodlar
#veri yükleme
veriler=pd.read_csv("AB_NYC_2019.csv",error_bad_lines=False)
#veri ön işleme
ide=veriler[["id"]]
print(ide)
#eksik veriler
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN", strategy="mean",axis=0)
deger=veriler.iloc[:,9:12].values#eksik veri yok
deger1=veriler.iloc[:,0:1].values#eksik veri yok
deger2=veriler.iloc[:,2:3].values#eksik veri yok
deger3=veriler.iloc[:,6:8].values#eksik veri yok
deger4=veriler.iloc[:,12:13].values#eksik veri var
deger5=veriler.iloc[:,13:14].values#eksik veri var
deger6=veriler.iloc[:,14:].values#eksik veri yok
#print(deger)

#print(deger1)

#print(deger2)

#print(deger3)

#print(deger4)
print(deger5)
#print(deger6)
imputer=imputer.fit(deger5[:,0:])
deger5[:,0:]=imputer.transform(deger5[:,0:])
print(deger5)

veriler =veriler.drop(columns ="last_review")
veri1=veriler._get_numeric_data()
print(veri1.isnull().sum().sort_values(ascending=False))
veri2 = veri1.dropna()
print(veri2)
print(veri2.isnull().sum().sort_values(ascending=False))
print(veri2.columns)

X=veri2.iloc[:,0:7].values
y=veri2.iloc[:,7].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.22, random_state = 0)
#ölçeklendirme
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler() 
X_train = sc.fit_transform(X_train) 
X_test = sc.fit_transform(X_test)

print(X_train)
print(X_test)

#PCA
from sklearn.decomposition import PCA,KernelPCA
pca=PCA(n_components=2)

X_train2=pca.fit_transform(X_train)
X_test2=pca.transform(X_test)

#integer
X_train=X_train.astype ("int")
X_test=X_test.astype("int")
y_train=y_train.astype("int")
y_test=y_test.astype("int")

#pca dönüşümünden önce gelen lojistik regresyon
from sklearn.linear_model import LogisticRegression

classifier=LogisticRegression(random_state=0,solver="lbfgs",multi_class="ovr")
classifier.fit(X_train,y_train)

#pca dönüşümünden sonra gelen lojistik regresyon
classifier2=LogisticRegression(random_state=0,solver="lbfgs",multi_class="ovr")
classifier2.fit(X_train2,y_train)

#tahminler
y_pred=classifier.predict(X_test)
y_pred = y_pred.astype ("int")
y_pred2=classifier2.predict(X_test2)
y_pred2 = y_pred2.astype ("int")

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#actual/pca olmadan çıkan sonuç
print("gerçek/pca olmadan")
cm1=confusion_matrix(y_test,y_pred)
print(cm1)
pl.matshow(cm1)
pl.title('gerçek/pca olmadan')
pl.colorbar()
pl.show()
print(accuracy_score(y_test,y_pred, normalize=False))
print(accuracy_score(y_test,y_pred))
#actual/pca sonrası çıkan sonuç
print("gerçek/pca sonrası")
cm2=confusion_matrix(y_test,y_pred2)
print(cm2)
pl.matshow(cm2)
pl.title('gerçek/pca sonrası')
pl.colorbar()
pl.show()
print(accuracy_score(y_test,y_pred2, normalize=False))
print(accuracy_score(y_test,y_pred2))
#pca sonrası/pca öncesi
print("pca öncesi/pca sonrası")
cm3=confusion_matrix(y_pred,y_pred2)
print(cm3)
pl.matshow(cm3)
pl.title('pca öncesi/pca sonrası')
pl.colorbar()
pl.show()
print(accuracy_score(y_pred,y_pred2, normalize=False))
print(accuracy_score(y_pred,y_pred2))

#normalizasyon
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_normed = mms.fit_transform(X_train)
X_test_normed= mms.transform(X_test)
print(X_train_normed)
print(X_test_normed)
#standardizasyon
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_std = stdsc.fit_transform(X_train)
X_std1 = stdsc.transform(X_test)
print(X_std)
print(X_std1)
#MDS
from sklearn.datasets import load_digits
from sklearn.manifold import MDS
mahmut=veri2.iloc[:,6:].values
mahmut.shape
embedding = MDS(n_components=3)
mahmut_transformed = embedding.fit_transform(mahmut[:150])
mahmut_transformed.shape
print(MDS)
print(mahmut_transformed.shape)

'''#SVM-linear
from sklearn.svm import SVC
svc=SVC(kernel='linear')
svc.fit(X_train,y_train)
y_pred =svc.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print('SVC-LİNEAR')
print(cm)
pl.matshow(cm)
pl.title('SVC-LİNEAR')
pl.colorbar()
pl.show()
print(accuracy_score(y_test,y_pred, normalize=False))
print(accuracy_score(y_test,y_pred))
#SVM-rbf
from sklearn.svm import SVC
svc=SVC(kernel='rbf',gamma='auto')
svc.fit(X_train,y_train)
y_pred =svc.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print('SVC-RBF')
print(cm)
pl.matshow(cm)
pl.title('SVC-RBF')
pl.colorbar()
pl.show()
print(accuracy_score(y_test,y_pred, normalize=False))
print(accuracy_score(y_test,y_pred))
#SVM-poly
from sklearn.svm import SVC
svc=SVC(kernel='poly',gamma='auto')
svc.fit(X_train,y_train)
y_pred =svc.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print('SVC-POLY')
print(cm)
pl.matshow(cm)
pl.title('SVC-POLY')
pl.colorbar()
pl.show()
print(accuracy_score(y_test,y_pred, normalize=False))
print(accuracy_score(y_test,y_pred))'''
'''#KARAR AĞAÇLARI
X=veri2.iloc[:,0:7].values
y=veri2.iloc[:,7].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.22, random_state = 0)
#ölçeklendirme
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler() 
X_train = sc.fit_transform(X_train) 
X_test = sc.fit_transform(X_test)
#integer
X_train=X_train.astype ("int")
X_test=X_test.astype("int")
y_train=y_train.astype("int")
y_test=y_test.astype("int")
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train,y_train)
y_pred=dtc.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print('DTC')
print(cm)'''
#KARAR AĞACI GÖRESEL ŞÖLEN
# Load libraries
import pandas as pd
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings('ignore')
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OutputCodeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
feature_cols = ['host_id','host_name', 'neighbourhood_group', 'neighbourhood']
#X = veriler[feature_cols] # Features
X=veri2.iloc[0:38843,2:6].values
veriler=veriler.replace(to_replace ="Shared room", 
                 value ="Private room") 
memoli=veriler.iloc[:,8:9].values
print(memoli)
le=LabelEncoder()
memoli[:,0]=le.fit_transform(memoli[:,0])
ohe=OneHotEncoder(categorical_features='all')
memoli=ohe.fit_transform(memoli).toarray()
y=memoli
y=pd.DataFrame(y)
y=y.iloc[0:38843,0:2].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("KARAR AĞACI")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
'''from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
import graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from IPython.display import Image  
from sklearn import tree
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('leylailemecnun.png')
Image(graph.create_png())'''
#KÜTÜPHANE
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
#RASSALORMAN
print("RASSAL ORMAN")
clf = RandomForestClassifier(criterion="entropy", max_depth=3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
cennet_mahallesi=accuracy_score(y_test, y_pred)
print("Accuracy:",cennet_mahallesi)
#BAGGİNG
#integer
X_train=X_train.astype ("int")
X_test=X_test.astype("int")
y_train=y_train.astype("int")
y_test=y_test.astype("int")
y_train=y_train[:,0].reshape(-1)
y_test=y_test[:,0].reshape(-1)
print("BAGGİNG")
from sklearn.ensemble import BaggingClassifier
bg = BaggingClassifier(clf,max_samples=0.5, max_features=1.0, n_estimators=20)
bg.fit(X_train, y_train)
y_pred =bg.predict(X_test)
polat=accuracy_score(y_test, y_pred)
print("Accuracy:",polat)
#ADABOOST
#integer
X_train=X_train.astype ("int")
X_test=X_test.astype("int")
y_train=y_train.astype("int")
y_test=y_test.astype("int")
print("ADABOOST")
clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=10, learning_rate=0.01)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
memati=accuracy_score(y_test, y_pred)
print("Accuracy:",memati)
'''# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('mahmut.png')
Image(graph.create_png())'''
'''#POLYREGRİ
# Load libraries
import pandas as pd
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings('ignore')
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OutputCodeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
feature_cols = ['host_id','host_name', 'neighbourhood_group', 'neighbourhood']
#X = veriler[feature_cols] # Features
#veri3=veri2.sort_index(by=['price', 'number_of_reviews'],ascending=True)
X=veri2.iloc[0:100,6:7].values
X1=X.sort_index()
veriler=veriler.replace(to_replace ="Shared room", 
                 value ="Private room") 
memoli=veriler.iloc[:,8:9].values
print(memoli)
le=LabelEncoder()
memoli[:,0]=le.fit_transform(memoli[:,0])
ohe=OneHotEncoder(categorical_features='all')
memoli=ohe.fit_transform(memoli).toarray()
y=memoli
y=pd.DataFrame(y)
y=y.iloc[0:3884,0:1].values
y=veri2.iloc[0:100,7:8].values
Y1=y.sort_index()
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X1,Y1)
plt.scatter(X1,Y1)
plt.plot(X1,lin_reg.predict(X1))
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2,interaction_only=False, include_bias=True)
x_poly = poly_reg.fit_transform(X1)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X1,Y1,color = 'red')
plt.plot(X1,lin_reg2.predict(poly_reg.fit_transform(X1)), color = 'blue')
plt.show()'''
#LİNEER REGRESYON
#leyla=veri2[['price']]
#mecnun=veri2[['number_of_reviews']]
leyla=veri2.iloc[0:35000,4:5]
mecnun=veri2.iloc[0:35000,6:7]
leyla=leyla.sort_index()
mecnun=mecnun.sort_index()
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(leyla)
poly_reg.fit(X_poly, mecnun)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, mecnun)
plt.scatter(leyla, mecnun, color = 'red')
plt.plot(leyla, lin_reg_2.predict(poly_reg.fit_transform(leyla)), color = 'blue')
plt.xlabel('price')
plt.ylabel('number_of_reviews')
plt.show()
'''from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(leyla, mecnun, test_size=0.3, random_state=1)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)
Y_train=sc.fit_transform(y_train)
Y_test=sc.fit_transform(y_test)
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x_train,y_train)
lin_reg.predict(x_test)
import matplotlib.pyplot as plt
x_train=x_train.sort_index()
y_train=y_train.sort_index()
plt.scatter(x_train,y_train)
plt.xlabel("price")
plt.ylabel("number_of_reviews")
plt.show()
plt.plot(x_train,y_train)
plt.plot(x_test,lin_reg.predict(x_test))
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2,interaction_only=False, include_bias=True)
x_poly = poly_reg.fit_transform(x_train)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y_train)
plt.scatter(x_train,y_train,color = 'red')
plt.xlabel("price")
plt.ylabel("number_of_reviews")
plt.plot(x_train,lin_reg2.predict(poly_reg.fit_transform(x_train)), color = 'blue')
plt.show()'''

'''#polynomial regresyon
x=veri2.iloc[0:3884,3:4]
y=veri2.iloc[0:3884,9:10]
X=x.values
Y=y.values
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)
plt.scatter(X,Y)
plt.plot(x,lin_reg.predict(X))
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()'''
'''#SpectralClustering
mt=veri2.iloc[:,6:].values
from sklearn.cluster import SpectralClustering
clustering = SpectralClustering(n_clusters=2,assign_labels="discretize",random_state=0)
clustering.fit(mt)
print(SpectralClustering)
print(clustering.labels_)
print(clustering)
sonuclar1=[]
for i in range(1,11):
    clustering = SpectralClustering(n_clusters=2,assign_labels="discretize",random_state=123)
    clustering.fit(mt)
    sonuclar1.append(clustering.inertia_)

plt.plot(range(1,11),sonuclar1)
plt.show()
clustering = SpectralClustering(n_clusters=2,assign_labels="discretize",random_state=123)
Y_tahmin1= clustering.fit_predict(mt)
print(Y_tahmin1)
plt.scatter(mt[Y_tahmin1==0,0],mt[Y_tahmin1==0,1],s=100, c="red")
plt.scatter(mt[Y_tahmin1==1,0],mt[Y_tahmin1==1,1],s=100, c="blue")
plt.scatter(mt[Y_tahmin1==2,0],mt[Y_tahmin1==2,1],s=100, c="green")
plt.scatter(mt[Y_tahmin1==3,0],mt[Y_tahmin1==3,1],s=100, c="yellow")
plt.title("SpectralClustering")
plt.show()
'''
'''#K-MEANS kümeleme
m=veri2.iloc[0:100,6:8].values

from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=6,init="k-means++")
kmeans.fit(m)
print(kmeans.cluster_centers_)
sonuclar=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=123)
    kmeans.fit(m)
    sonuclar.append(kmeans.inertia_)

plt.plot(range(1,11),sonuclar)
plt.show()
kmeans = KMeans (n_clusters = 6, init="k-means++", random_state= 123)
Y_tahmin= kmeans.fit_predict(m)
print(Y_tahmin)
plt.scatter(m[Y_tahmin==0,0],m[Y_tahmin==0,1],s=100, c="red")
plt.scatter(m[Y_tahmin==1,0],m[Y_tahmin==1,1],s=100, c="blue")
plt.scatter(m[Y_tahmin==2,0],m[Y_tahmin==2,1],s=100, c="green")
plt.scatter(m[Y_tahmin==3,0],m[Y_tahmin==3,1],s=100, c="yellow")
plt.title("KMeans")
plt.show()'''
'''#ROCK
from sklearn.metrics import roc_curve, auc
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X=veri2.iloc[:,0:3].values
y=veri2.iloc[:,len(veri2.iloc[0])-1].values
y = label_binarize(y, classes=[0,1,2])
n_classes = 3

X_train, X_test, y_train, y_test =\
    train_test_split(X, y, test_size=0.33, random_state=0)
#integer
X_train=X_train.astype ("int")
X_test=X_test.astype("int")
y_train=y_train.astype("int")
y_test=y_test.astype("int")

# classifier
clf = OneVsRestClassifier(LinearSVC(random_state=0))
y_score = clf.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for a specific class
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()'''

'''#hiyerarşik kümeleme
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="ward")
Y_tahmin=ac.fit_predict(m)
print(Y_tahmin)
plt.scatter(m[Y_tahmin==0,0],m[Y_tahmin==0,1],s=100,c="red")
plt.scatter(m[Y_tahmin==1,0],m[Y_tahmin==1,1],s=100,c="blue")
plt.scatter(m[Y_tahmin==2,0],m[Y_tahmin==2,1],s=100,c="green")
plt.scatter(m[Y_tahmin==3,0],m[Y_tahmin==3,1],s=100,c="yellow")
plt.title("HC")
plt.show()

import scipy.cluster.hierarchy as sch
dendogram=sch.dendrogram(sch.linkage(m, method="ward"))
plt.show()'''
#k-nn en yakın komşu
print("en yakın komşu")
import pandas as pd
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings('ignore')
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OutputCodeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
X=veri2.iloc[0:38843,1:9].values
veriler=veriler.replace(to_replace ="Shared room", 
                 value ="Private room") 
memoli=veriler.iloc[:,8:9].values
#print(memoli)
le=LabelEncoder()
memoli[:,0]=le.fit_transform(memoli[:,0])
ohe=OneHotEncoder(categorical_features='all')
memoli=ohe.fit_transform(memoli).toarray()
y=memoli
y=pd.DataFrame(y)
y=y.iloc[0:38843,0:2].values

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5,metric="minkowski")
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)
pl.matshow(cm)
pl.title('en yakın komşu')
pl.colorbar()
pl.show()
print(accuracy_score(y_test,y_pred, normalize=False))
print(accuracy_score(y_test,y_pred))
error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
    
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
#SVMMMM
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import pandas as pd
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings('ignore')
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OutputCodeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
X=veri2.iloc[0:100,1:3].values
veriler=veriler.replace(to_replace ="Shared room", 
                 value ="Private room") 
memoli=veriler.iloc[:,8:9].values
print(memoli)
le=LabelEncoder()
memoli[:,0]=le.fit_transform(memoli[:,0])
ohe=OneHotEncoder(categorical_features='all')
memoli=ohe.fit_transform(memoli).toarray()
y=memoli
y=pd.DataFrame(y)
y=y.iloc[0:100,0:1].values
import sys, os
import matplotlib.pyplot as plt
from sklearn import svm
from matplotlib import *

from sklearn.model_selection import train_test_split, GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

# Plot traning and test data
#plt.plot(X_train, y_train, X_test, y_test)
# Create a linear SVM classifier 
clf = svm.SVC(kernel='linear')
# Train classifier 
clf.fit(X_train, y_train)
 
# Plot decision function on training and test data
#plt.plot(X_train, y_train, X_test, y_test, clf)
# Make predictions on unseen test data
clf_predictions = clf.predict(X_test)
print("Accuracy: {}%".format(clf.score(X_test, y_test) * 100 ))
# Create a linear SVM classifier with C = 1
clf = svm.SVC(kernel='linear', C=1)
# Create SVM classifier based on RBF kernel. 
clf = svm.SVC(kernel='rbf', C = 10.0, gamma=0.1)
# Grid Search
# Parameter Grid
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}
 
# Make grid search classifier
clf_grid = GridSearchCV(svm.SVC(), param_grid, verbose=1)
 
# Train the classifier
clf_grid.fit(X_train, y_train)
 
# clf = grid.best_estimator_()
print("Best Parameters:\n", clf_grid.best_params_)
print("Best Estimators:\n", clf_grid.best_estimator_)


'''#Gaussian bayes
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)
y_pred=gnb.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print("GNB")
print(cm)
pl.matshow(cm)
pl.title('GNB')
pl.colorbar()
pl.show()
print(accuracy_score(y_test,y_pred, normalize=False))
print(accuracy_score(y_test,y_pred))
#hedef veri
import pandas as pd
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings('ignore')
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OutputCodeClassifier

#memoli=veriler.iloc[:,8:9].values
#print(memoli)
#yilan_hikayesi=MultiLabelBinarizer().fit_transform(memoli)
#print(yilan_hikayesi)'''
'''#ROCKKK
print(_doc_)

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

# Import some data to play with
X=veri2.iloc[:,0:9].values
y=veri2.iloc[:,len(veri2.iloc[0])-1].values

# Binarize the output
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples,n_features)]

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()'''
#SVMBEBEĞİM
# Load libraries
import pandas as pd
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings('ignore')
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OutputCodeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

feature_cols = ['host_id','host_name', 'neighbourhood_group', 'neighbourhood']
#X = veriler[feature_cols] # Features
veri3=veri2.sort_values(by=['price', 'number_of_reviews'],ascending=True)
X=veri3.iloc[0:100,6:7].values
'''veriler=veriler.replace(to_replace ="Shared room", 
                 value ="Private room") 
memoli=veriler.iloc[:,8:9].values
print(memoli)
le=LabelEncoder()
memoli[:,0]=le.fit_transform(memoli[:,0])
ohe=OneHotEncoder(categorical_features='all')
memoli=ohe.fit_transform(memoli).toarray()
y=memoli
y=pd.DataFrame(y)
y=y.iloc[0:3884,0:1].values'''
y=veri3.iloc[0:100,4:5].values

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
x_olcekli=sc1.fit_transform(X)
sc2=StandardScaler()
y_olcekli=sc1.fit_transform(y)
from sklearn.svm import SVR
svr_reg=SVR(kernel='poly')
svr_reg.fit(x_olcekli,y_olcekli)
print("SVMBEBEĞİM")
plt.scatter(x_olcekli,y_olcekli,color='red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue')
#LOGİSTİCBEBEĞİM
# Load libraries
import pandas as pd
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings('ignore')
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OutputCodeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

feature_cols = ['host_id','host_name', 'neighbourhood_group', 'neighbourhood']
#X = veriler[feature_cols] # Features
veri3=veri2.sort_values(by=['price', 'number_of_reviews'],ascending=True)
X=veri3.iloc[0:100,5:8].values
'''veriler=veriler.replace(to_replace ="Shared room", 
                 value ="Private room") 
memoli=veriler.iloc[:,8:9].values
print(memoli)
le=LabelEncoder()
memoli[:,0]=le.fit_transform(memoli[:,0])
ohe=OneHotEncoder(categorical_features='all')
memoli=ohe.fit_transform(memoli).toarray()
y=memoli
y=pd.DataFrame(y)
y=y.iloc[0:3884,0:1].values'''
y=veri3.iloc[0:100,4:5].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.transform(x_test)
from sklearn.linear_model import LogisticRegression
logr=LogisticRegression(random_state=0)
logr.fit(X_train,y_train)
cemal=logr.predict(X_test)
print(cemal)
print(y_test)
#ROCKKKKKKK
import pandas as pd
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings('ignore')
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OutputCodeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
X=veri2.iloc[0:38843,1:9].values
veriler=veriler.replace(to_replace ="Shared room", 
                 value ="Private room") 
memoli=veriler.iloc[:,8:9].values
print(memoli)
le=LabelEncoder()
memoli[:,0]=le.fit_transform(memoli[:,0])
ohe=OneHotEncoder(categorical_features='all')
memoli=ohe.fit_transform(memoli).toarray()
y=memoli
y=pd.DataFrame(y)
y=y.iloc[0:38843,0:2].values
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=.25,
                                                    random_state=1)
# Import the classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve, roc_auc_score

# Instantiate the classfiers and make a list
classifiers = [LogisticRegression(random_state=1), 
               GaussianNB(), 
               KNeighborsClassifier(), 
               DecisionTreeClassifier(random_state=1),
               RandomForestClassifier(random_state=1)]

# Define a result table as a DataFrame
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])
y_train=y_train[:,0].reshape(-1)
y_test=y_test[:,0].reshape(-1)
# Train the models and record the results
for cls in classifiers:
    model = cls.fit(X_train, y_train)
    yproba = model.predict_proba(X_test)[::,1]
    
    fpr, tpr, _ = roc_curve(y_test,  yproba)
    auc = roc_auc_score(y_test, yproba)
    
    result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)

# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)
fig = plt.figure(figsize=(8,6))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("Flase Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.show()
'''#ROCKKIYAS
import pandas as pd
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings('ignore')
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OutputCodeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
X=veri2.iloc[0:38843,1:9].values
#X=veri2.iloc[:,0:9].values
veriler=veriler.replace(to_replace ="Shared room", 
                 value ="Private room") 
memoli=veriler.iloc[:,8:9].values
#memoli=pd.DataFrame(memoli)
print(memoli)
le=LabelEncoder()
memoli[:,0]=le.fit_transform(memoli[:,0])
ohe=OneHotEncoder(categorical_features='all')
memoli=ohe.fit_transform(memoli).toarray()
#memoli=np.split(memoli,'.')
y=memoli
y=pd.DataFrame(y)
y=y.iloc[0:38843,0:2].values
print(y)
#y=veri2.iloc[:,len(veri2.iloc[0])-1].values
#y=veri2.iloc[:,-1:].values
#y=MultiLabelBinarizer().fit_transform(y)
def correct_round(x):
    try:
        y = [ round(z) for z in x ]
    except:
        y = round(x)    
    return y
#y=correct_round(y)
#y=MultiLabelBinarizer().fit_transform(y)
y = label_binarize(y, classes=[0,1])
n_classes =y.shape[1]
#y.argmax(axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=.25,
                                                    random_state=0)
# Import the classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


from sklearn.metrics import roc_curve, roc_auc_score
#sklearn.metrics.roc_auc_score(y_true, y_score, average='macro', sample_weight=None)

# Instantiate the classfiers and make a list
classifiers = [LogisticRegression(random_state=0,solver='warn',multi_class='ovr'), 
               GaussianNB(), 
               KNeighborsClassifier()]

# Define a result table as a DataFrame
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

def multiclass_roc_auc_score(y_train,y_test, yproba,y_score,y_true,X_train,X_test, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    lb.fit(y_train)
    lb.fit(X_train)
    lb.fit(X_test)
    y_test = lb.transform(y_test)
    y_train = lb.transform(y_train)
    yproba = lb.transform(yproba)
    y_true = lb.transform(y_true)
    y_score = lb.transform(y_score)
    X_train = lb.transform(X_train)
    X_test = lb.transform(X_test)
    
    return roc_auc_score(X_train,y_train,y_test, y_pred, average="average")
print(y_train)
y_train=pd.DataFrame(y_train)
y_train=y_train.iloc[:,0:2].values
y_train=y_train[:,0].reshape(-1)
y_test=y_test[:,0].reshape(-1)
#X_train=X_train.reshape(2,2)
#X_test=X_test.reshape(2,2)
#y_train=y_train.reshape(2,2)
#y_test=y_test.reshape(2,2)
# Train the models and record the results
for cls in classifiers:
    model = cls.fit(X_train, y_train)
    yproba = model.predict_proba(X_test)[::,1]
    fpr, tpr ,_= roc_curve(y_test,  yproba)
    
    auc =OutputCodeClassifier(roc_auc_score(y_test, yproba,average='average', sample_weight=None))
    
    result_table = result_table.append({'classifiers':cls._class.name_,
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)

# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)
fig = plt.figure(figsize=(8,6))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("Flase Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.show()'''
'''#kernel pca 
#from sklearn.datasets import make_circles
#from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 2, kernel = 'rbf')
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0,solver="liblinear",multi_class="auto")
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.legend()
plt.show()'''
