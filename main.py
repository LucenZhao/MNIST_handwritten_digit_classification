import tensorflow as tf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import KFold, cross_val_score
from preprocess import extract_data, visualize
from feature_extraction import mnist_features
from feature_selection import select_feats
from models import LogisticReg
import numpy as np

def main():
   # load and preprocess data
   train_labels, train_imgs = extract_data('data/zip.train')
   test_labels, test_imgs = extract_data('data/zip.test')
   
   # feature extraction
   train_feats = []
   test_feats = []
   num_train = len(train_imgs)
   num_test = len(test_imgs)
   plain = True  # set parameters for feature extraction
   pool = {'take': False, 'class': 'max'}
   hist = {'take': False, 'h': [4], 'w': [4]}
   grad = {'take': True, 'class': 'hist'}
   chain = {'take': True, 'class': 'hist'}
   
   print("Extract feature from training data set...")
   for i, img in enumerate(train_imgs):
      img_feats = mnist_features(img=img, plain=plain, pool=pool, hist=hist, grad=grad, chain=chain)
      train_feats.append(img_feats.feats)
      if i % 500 == 0:
         print(str(i)+" / "+str(num_train)+" images processed.")
   train_feats = np.array(train_feats)

   print("Extract feature from testing data set...")
   for i, img in enumerate(test_imgs):
      img_feats = mnist_features(img=img, plain=plain, pool=pool, hist=hist, grad=grad, chain=chain)
      test_feats.append(img_feats.feats)
      if i % 500 == 0:
         print(str(i)+" / "+str(num_test)+" images processed.")
   test_feats = np.array(test_feats)

   print("All data processed. Number of features extracted is "+str(len(train_feats[0])))
   
   '''
   select_model = SelectPercentile(chi2, percentile=30)
   select_model.fit(train_feats, train_labels)
   train_selected = select_model.transform(train_feats)
   test_selected = select_model.transform(test_feats)
   '''


   lda = LinearDiscriminantAnalysis()
   qda = QuadraticDiscriminantAnalysis()
   #lr1 = LogisticRegression()
   lr1 = LogisticRegression(solver='newton-cg')
   #lr3 = LogisticRegression(solver='lbfgs')
   lr2 = LogisticRegression(multi_class='multinomial', solver='newton-cg')
   #lr5 = LogisticRegression(multi_class='multinomial', solver='lbfgs')
   kn1 = KNeighborsClassifier(n_neighbors=1)
   kn2 = KNeighborsClassifier(n_neighbors=3)
   #kn3 = KNeighborsClassifier(n_neighbors=10)
   k_fold = KFold(n_splits=3)
#   score1 = sum(cross_val_score(lr1, train_feats, train_labels, cv=k_fold, scoring='accuracy')) / 3
#   score2 = sum(cross_val_score(lr2, train_feats, train_labels, cv=k_fold, scoring='accuracy')) / 3
#   score3 = sum(cross_val_score(lr3, train_feats, train_labels, cv=k_fold, scoring='accuracy')) / 3
#   print(score1)
#   print(score2)
#   print(score3)
#   score1 = sum(cross_val_score(kn1, train_feats, train_labels, cv=k_fold, scoring='accuracy')) / 3
#   score2 = sum(cross_val_score(kn2, train_feats, train_labels, cv=k_fold, scoring='accuracy')) / 3
#   score3 = sum(cross_val_score(kn3, train_feats, train_labels, cv=k_fold, scoring='accuracy')) / 3
#   print(score1)
#   print(score2)
#   print(score3)
#   score1 = sum(cross_val_score(lr4, train_feats, train_labels, cv=k_fold, scoring='accuracy')) / 3
#   score2 = sum(cross_val_score(lr5, train_feats, train_labels, cv=k_fold, scoring='accuracy')) / 3
#   print(score1)
#   print(score2)

#   models = [lda]
#   for i, clf in enumerate(models):
#      clf.fit(train_feats, train_labels)
#      preds = clf.predict(test_feats)
#      preds_proba = clf.predict_proba(test_feats)
#      scores_3 = preds_proba[:,3]
#      scores_4 = preds_proba[:,4]
#      scores_8 = preds_proba[:,8]
#      scores_9 = preds_proba[:,9]
#      scores_8 = np.reshape(scores_8, (-1,))
#      scores_9 = np.reshape(scores_9, (-1,))
#      scores_3 = np.reshape(scores_3, (-1,))
#      scores_4 = np.reshape(scores_4, (-1,))   
#      print("Current model is "+str(i))
#      print(sum(preds == test_labels))
#      print(sum(preds != test_labels))
#      print(sum(preds != test_labels) / len(test_labels))
#      print(classification_report(test_labels, preds))
#      print(confusion_matrix(test_labels, preds))
#      fpr_3, tpr_3, _ = roc_curve(test_labels, scores_3, pos_label=3)
#      fpr_4, tpr_4, _ = roc_curve(test_labels, scores_4, pos_label=4)
#      fpr_8, tpr_8, _ = roc_curve(test_labels, scores_8, pos_label=8)
#      fpr_9, tpr_9, _ = roc_curve(test_labels, scores_9, pos_label=9)
#      print(fpr_3)
#      print(tpr_3)
#      import matplotlib.pyplot as plt      
#      plt.figure(1)
#      plt.plot([0, 1], [0, 1], 'k--')
#      plt.plot(fpr_3, tpr_3, label='Scores for 3')
#      plt.plot(fpr_4, tpr_4, label='Scores for 4')
#      plt.plot(fpr_8, tpr_8, label='Scores for 8')
#      plt.plot(fpr_9, tpr_9, label='Scores for 9')
#      plt.xlabel('False positive rate')
#      plt.ylabel('True positive rate')
#      plt.title('ROC curve')
#      plt.legend(loc='best')
#      plt.savefig("report/lda_roc.png", pad_inches = 0)
#      plt.show()


   models = [lr1]
   for i, clf in enumerate(models):
#      score = sum(cross_val_score(clf, train_feats, train_labels, cv=k_fold, scoring='accuracy')) / 3
#      print(score)
      clf.fit(train_feats, train_labels)
      preds = clf.predict(test_feats)
      print("Current model is "+str(i))
      print(sum(preds == test_labels))
      print(sum(preds != test_labels))
      print(sum(preds != test_labels) / len(test_labels))
      print(classification_report(test_labels, preds))
      print(confusion_matrix(test_labels, preds))
      err_imgs = test_imgs[preds != test_labels]
      err_labels = test_labels[preds != test_labels]
      visualize(err_labels, err_imgs)

if __name__ == '__main__':
   main()