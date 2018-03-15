from preprocess import extract_data, visualize
from utils import get_feats, cross_valid, final_result, plot_ROC
import config


def main():
    # load and preprocess data
    train_labels, train_imgs = extract_data(config.train_path)
    test_labels, test_imgs = extract_data(config.test_path)
    f = open(config.output_file, 'w')

    # model selection
    if config.select_model is True:
        print("Selecting model...")
        f.write("Scores for model selection:\n")
        # original image
        plain = True  # set parameters for feature extraction
        pool = {'take': False, 'class': 'max'}
        hist = {'take': False, 'h': [4], 'w': [4]}
        grad = {'take': False, 'class': 'hist'}
        chain = {'take': False, 'class': 'hist'}
        select_feats1 = get_feats(train_imgs, plain,
                                  pool, hist, grad, chain)

        # feature vector
        pool = {'take': False, 'class': 'max'}
        hist = {'take': True, 'h': [4], 'w': [4]}
        grad = {'take': True, 'class': 'hist'}
        chain = {'take': True, 'class': 'hist'}
        select_feats2 = get_feats(train_imgs, plain,
                                  pool, hist, grad, chain)

        # get cross-validation scores
        f.write("Baseline (original image):" + '\n')
        print("logistic regression models:")
        scores = cross_valid(config.models_select1,
                             select_feats1, train_labels)
        print(scores)
        for i, s in enumerate(scores):
            f.write(config.names1[i] + ':' + str(scores[i]) + '\n')
        print("multi-class logistic regression models:")
        scores = cross_valid(config.models_select2,
                             select_feats1, train_labels)
        print(scores)
        for i, s in enumerate(scores):
            f.write(config.names2[i] + ':' + str(scores[i]) + '\n')
        print("k-nearest neighbour models:")
        scores = cross_valid(config.models_select3,
                             select_feats1, train_labels)
        print(scores)
        for i, s in enumerate(scores):
            f.write(config.names3[i] + ':' + str(scores[i]) + '\n')

        f.write("\nFeature vector:" + '\n')
        scores = cross_valid(config.models_select1,
                             select_feats2, train_labels)
        print(scores)
        for i, s in enumerate(scores):
            f.write(config.names1[i] + ':' + str(scores[i]) + '\n')
        print("multi-class logistic regression models:")
        scores = cross_valid(config.models_select2,
                             select_feats2, train_labels)
        print(scores)
        for i, s in enumerate(scores):
            f.write(config.names2[i] + ':' + str(scores[i]) + '\n')
        print("k-nearest neighbour models:")
        scores = cross_valid(config.models_select3,
                             select_feats2, train_labels)
        print(scores)
        for i, s in enumerate(scores):
            f.write(config.names3[i] + ':' + str(scores[i]) + '\n')

        f.write("\n######################\n\n")

    # feature selection
    if config.select_feature is True:
        print("Selecting features...")
        f.write("Scores for feature selection:\n")
        plain = True  # set parameters for feature extraction
        pool = {'take': False, 'class': 'max'}
        hist = {'take': False, 'h': [4], 'w': [4]}
        grad = {'take': False, 'class': 'hist'}
        chain = {'take': False, 'class': 'hist'}

        # histogram
        hist['take'] = True
        print("Extract histogram from training data set...")
        f.write("\nHistogram:\n")
        select_feats = get_feats(train_imgs, plain, pool, hist, grad, chain)
        scores = cross_valid(config.models, select_feats, train_labels)
        print(scores)
        for i, s in enumerate(scores):
            f.write(config.names[i] + ':' + str(scores[i]) + '\n')

        # gradient histogram
        hist['take'] = False
        grad['take'] = True
        print("Extract gradient histogram from training data set...")
        f.write("\nGradient histogram:\n")
        select_feats = get_feats(train_imgs, plain, pool, hist, grad, chain)
        scores = cross_valid(config.models, select_feats, train_labels)
        print(scores)
        for i, s in enumerate(scores):
            f.write(config.names[i] + ':' + str(scores[i]) + '\n')

        # gradient image
        grad['class'] = 'plain'
        print("Extract gradient image from training data set...")
        f.write("\nGradient image:\n")
        select_feats = get_feats(train_imgs, plain, pool, hist, grad, chain)
        scores = cross_valid(config.models, select_feats, train_labels)
        print(scores)
        for i, s in enumerate(scores):
            f.write(config.names[i] + ':' + str(scores[i]) + '\n')

        # chain code histogram
        grad['take'] = False
        chain['take'] = True
        print("Extract chain code histogram from training data set...")
        f.write("\nChain code histogram:\n")
        select_feats = get_feats(train_imgs, plain, pool, hist, grad, chain)
        scores = cross_valid(config.models, select_feats, train_labels)
        print(scores)
        for i, s in enumerate(scores):
            f.write(config.names[i] + ':' + str(scores[i]) + '\n')

        f.write("\n######################\n\n")

    if config.produce_results is True or config.draw_ROC is True:
        # feature extraction
        print("Extract feature from training data set...")
        train_feats = get_feats(train_imgs, config.plain, config.pool,
                                config.hist, config.grad, config.chain)
        print("Extract feature from testing data set...")
        test_feats = get_feats(test_imgs, config.plain, config.pool,
                               config.hist, config.grad, config.chain)
        print("All data processed. Number of features extracted is " +
              str(len(train_feats[0])))

        if config.produce_results is True:
            print("Producing prediction results...")
            f.write("Prediction results\n")
            f.write('original image: ' + str(config.plain))
            f.write('\n')
            f.write('pooled:' + str(config.pool))
            f.write('\n')
            f.write('histogram:' + str(config.hist))
            f.write('\n')
            f.write('gradient:' + str(config.grad))
            f.write('\n')
            f.write('chain code:' + str(config.chain))
            f.write('\n\n')
            all_preds = final_result(config.models, config.names, train_feats,
                                     train_labels, test_feats, test_labels, f)

            if config.visualize_error is True:
                preds = all_preds[2]
                err_imgs = test_imgs[preds != test_labels]
                err_labels = test_labels[preds != test_labels]
                visualize(err_labels, err_imgs)

        if config.draw_ROC is True:
            print("Drawing ROC for LDA model...")
            preds_proba = plot_ROC(config.lda, train_feats, train_labels,
                                   test_feats, test_labels)


if __name__ == '__main__':
    main()
