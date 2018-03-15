from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, \
                                          QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# operation parameters
select_model = False
select_feature = False
produce_results = True
visualize_error = True
draw_ROC = True

# data paths
train_path = '../data/zip.train'
test_path = '../data/zip.test'
output_file = 'results.txt'

# feature extraction parameters
plain = True
pool = {'take': False, 'class': 'max'}  # 'max' or 'mean'
hist = {'take': False, 'h': [4], 'w': [4]}
grad = {'take': True, 'class': 'hist'}  # 'hist', 'plain' or 'pool'
chain = {'take': True, 'class': 'hist'}

# model definition
lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()
lr1 = LogisticRegression(solver='newton-cg')
lr2 = LogisticRegression(multi_class='multinomial', solver='newton-cg')
lr3 = LogisticRegression()
lr4 = LogisticRegression(solver='lbfgs')
lr5 = LogisticRegression(multi_class='multinomial', solver='lbfgs')
kn1 = KNeighborsClassifier(n_neighbors=1)
kn2 = KNeighborsClassifier(n_neighbors=3)
kn3 = KNeighborsClassifier(n_neighbors=10)

# model selection
models = [lda, qda, lr1, lr2, kn1, kn2]
names = ['LDA', 'QDA', 'Logistic regression',
         'Multiclass logistic regression', '1-NN', '3-NN']
models_select1 = [lr1, lr3, lr4]  # LR
names1 = ['Logistic regression, Newton-CG',
          'Logistic regression, CD', 'Logistic regression, LBFGS']
models_select2 = [lr2, lr5]  # multi-class LR
names2 = ['Multinomial logistic regression, Newton-CG',
          'Multinomial logistic regression, LBFGS']
models_select3 = [kn1, kn2, kn3]  # knn
names3 = ['1-NN', '3-NN', '10-NN']
