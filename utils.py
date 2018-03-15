from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import KFold, cross_val_score
from feature_extraction import mnist_features
import numpy as np


def get_feats(all_imgs, plain, pool, hist, grad, chain):
    feats = []
    num = len(all_imgs)
    for i, img in enumerate(all_imgs):
        img_feats = mnist_features(img=img, plain=plain, pool=pool,
                                   hist=hist, grad=grad, chain=chain)
        feats.append(img_feats.feats)
        if i % 500 == 0:
            print(str(i) + " / " + str(num) + " images processed.")
    feats = np.array(feats)

    return feats


def cross_valid(models, feats, labels):
    ns = 3
    k_fold = KFold(n_splits=ns)
    scores = []
    for clf in models:
        score = sum(cross_val_score(clf, feats, labels, cv=k_fold,
                                    scoring='accuracy')) / ns
        scores.append(score)
    return scores


def final_result(models, names, train_feats, train_labels,
                 test_feats, test_labels, f):
    all_preds = []
    for i, clf in enumerate(models):
        clf.fit(train_feats, train_labels)
        preds = clf.predict(test_feats)
        print("Current model is "+names[i])
        f.write(names[i] + '\n')
        right = sum(preds == test_labels)
        wrong = sum(preds != test_labels)
        report = classification_report(test_labels, preds)
        matrix = confusion_matrix(test_labels, preds)
        print(right)
        print(wrong)
        print(report)
        print(matrix)
        f.write('correct: ' + str(right) + '; wrong:' + str(wrong) + '\n')
        f.write(str(report))
        f.write('\n')
        f.write(str(matrix))
        f.write('\n\n\n')
        all_preds.append(preds)

    return all_preds


def plot_ROC(clf, train_feats, train_labels, test_feats, test_labels):
    clf.fit(train_feats, train_labels)
    preds_proba = clf.predict_proba(test_feats)
    scores_3 = preds_proba[:, 3]
    scores_4 = preds_proba[:, 4]
    scores_8 = preds_proba[:, 8]
    scores_9 = preds_proba[:, 9]
    scores_3 = np.reshape(scores_3, (-1, ))
    scores_4 = np.reshape(scores_4, (-1, ))
    scores_8 = np.reshape(scores_8, (-1, ))
    scores_9 = np.reshape(scores_9, (-1, ))
    fpr_3, tpr_3, _ = roc_curve(test_labels, scores_3, pos_label=3)
    fpr_4, tpr_4, _ = roc_curve(test_labels, scores_4, pos_label=4)
    fpr_8, tpr_8, _ = roc_curve(test_labels, scores_8, pos_label=8)
    fpr_9, tpr_9, _ = roc_curve(test_labels, scores_9, pos_label=9)

    import matplotlib.pyplot as plt
    plt.figure(11)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_3, tpr_3, label='Scores for 3')
    plt.plot(fpr_4, tpr_4, label='Scores for 4')
    plt.plot(fpr_8, tpr_8, label='Scores for 8')
    plt.plot(fpr_9, tpr_9, label='Scores for 9')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig("lda_roc.png", pad_inches=0)
    plt.show()

    return preds_proba
