from sklearn.feature_selection import chi2

def select_feats(feats, labels, thresh):
   scores, pvalues = chi2(feats, labels)
   print(scores)
   print(pvalues)
   print("counting p values")
   print(sum(pvalues > 0.0001))
   print(sum(pvalues > 0.001))
   print(sum(pvalues > 0.01))