import argparse
import pandas as pd
import csv
import operator
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score


class NaiveBayesClassifier:
    def __init__(self):
        """
        Constructor: Initializes the P(C) and also the P(w|C) and the vocabulary
        Parameters:
            N/A
        Returns:
            N/A
        """
        # vocabulary list is just like a dictionary book!
        self.vocabulary_list = []
        # the class probability variable
        self.class_prob = {}
        # the word given class probability variable
        self.word_prob = {}

    def k_fold_cv(self, sentences, n=3):
        """
        Hyper-parameter tuning using K-fold cross validation using K=3
        Parameters:
            sentences (list): a list, containing all of the train sentences
            n (int): number of folds
        Returns:
            average_acc (float): the average accuracy of the K-folded cross validation procedure
        """
        # defining the KFold object
        kf = KFold(n_splits=n)
        # setting the average accuracy to zero
        average_acc = 0
        # looping through the splitted kf object
        for train_idx, dev_idx in kf.split(sentences):
            gold_label_prediction_list = []
            train_data = [sentences[i] for i in train_idx]
            dev_data = [sentences[i] for i in dev_idx]

            # calling the train method to train the data
            self.train(train_data)

            for i in range(len(dev_data)):
                # taking the first element as the sentence
                x_dev = dev_data[i][0]
                # taking the second element as the gold label
                gold_label = dev_data[i][1]
                # calling the predict method to predict the output label of the sentence
                predicted_label = self.predict(x_dev)
                gold_label_prediction_list.append((gold_label, predicted_label))

            # calculating the average accuracy
            average_acc += self.cal_accuracy(gold_label_prediction_list)/n
        return average_acc

    def cal_accuracy(self, gold_label_prediction_list):
        """
        Calculates the accuracy, given the gold standard label and the predicted list
        Parameters:
            gold_label_prediction_list (list): a list, containing ground truth and the predictions
        Returns:
            accuracy (float): the accuracy w.r.t the input prediction and true label
        """
        correct = 0
        for gold_label, predicted_label in gold_label_prediction_list:
            if gold_label == predicted_label:
                correct += 1
        accuracy = correct / len(gold_label_prediction_list)
        return accuracy

    def train(self, sentences, smoothing=1):
        """
        Trains the model given the input sentences
        Parameters:
            sentences (list): a list, containing the input sentences
            smoothing (int): the laplace smoothing value
        Returns:
            N/A
        """

        # number of sentences in each class
        sentence_count_per_class = {}
        # number each word occurrence in each class
        word_count_per_class = {}
        # number of words in each class
        class_word_count = {}

        for sentence, _class in sentences:
            try:
                # try to increment the count by one
                sentence_count_per_class[_class] += 1
            # if not found
            except KeyError:
                # set the count to 1 (since it's the first time we're seeing it)
                sentence_count_per_class[_class] = 1
                class_word_count[_class] = 0
                word_count_per_class[_class] = {}

            self.vocabulary_list = list(set.union(set(self.vocabulary_list), set(sentence)))

            # counting the number of words for the specific class
            for word in sentence:
                class_word_count[_class] += 1
                try:
                    # if the word already has been seen in the specific class
                    word_count_per_class[_class][word] += 1
                except KeyError:
                    # if the word has not seen before
                    word_count_per_class[_class][word] = 1

        for _class, num in sentence_count_per_class.items():
            # looping through the sentence count per each class to calculate the P(C)
            self.class_prob[_class] = num/len(sentences)

        for _class in word_count_per_class.keys():
            # looping through the word count per class to calculate the P(w|C)
            self.word_prob[_class] = {}
            # for the case where we have an unknown word
            self.word_prob[_class]['<U>'] = 1 / (class_word_count[_class] + len(self.vocabulary_list))

            for word, num in word_count_per_class[_class].items():
                # applying smoothing to the word probability given class
                self.word_prob[_class][word] = (num+smoothing) / (class_word_count[_class]+len(self.vocabulary_list))

    def predict(self, sentence):
        """
        Predicts the output label, given the input sentence
        Parameters:
            sentence (list): input sentence
        Returns:
            the classes (relations) with maximum accuracy
        """
        classes = [c for c in self.class_prob.keys()]
        probs = {}

        # for each class (relation), let's calculate the probability
        for _class in classes:
            prob = 1
            # looping through the sentence to calculate the prob of each word and finally the sentence
            for word in sentence:
                # only consider the words that are listed in the vocabulary
                if word in self.vocabulary_list:
                    # if the word does not exist in the word probability of the class
                    if word not in self.word_prob[_class].keys():
                        # consider it's probability as the probability of the unknown word
                        prob *= self.word_prob[_class]['<U>']
                    # if the word has a probability
                    else:
                        prob *= self.word_prob[_class][word]
            # the final probability would be equal to P(C) * P(w|C)
            probs[_class] = self.class_prob[_class] * prob
        # return the key of the maximum value
        return max(probs, key=probs.get)

    def load_file(self, file_path):
        """
        Loads the input train and test file, given the file path
        Parameters:
            file_path (str): path to the input file
        Returns:
            processed_sentences (list): a list, containing the sentences and relations
            ids (list): a list, containing each row id

                """
        df = pd.read_csv(file_path)
        # print(df['relation'].value_counts())
        ids = df['row_id'].tolist()
        processed_sentences = []

        # looping through the dataframe
        for index, row in df.iterrows():
            sentence = []
            for word in row['tokens'].split(' '):
                if word.isalnum():
                    sentence.append(word)
            processed_sentences.append((sentence, row['relation']))
        # assertions
        # print(len(ids))
        # print(len(processed_sentences))
        return processed_sentences, ids


def main():
    # taking the input arguments from the terminal
    arg_parser = argparse.ArgumentParser(description="naive Bayes classifier")
    arg_parser.add_argument('--train', required=True, help="Relative path to the train csv file")
    arg_parser.add_argument('--test', required=True, help="Relative path to the test csv file")
    arg_parser.add_argument('--output', required=True, help="Relative path to the output csv file")

    # parsing and loading the args
    args = arg_parser.parse_args()
    train_path = args.train
    test_path = args.test
    output_path = args.output

    # initializing an object from our NaiveBayesClassifier class
    classifier = NaiveBayesClassifier()
    # loading the sentences and ids
    train_sentences, _ = classifier.load_file(train_path)

    acc = classifier.k_fold_cv(train_sentences)
    print('NB training accuracy using K=3 fold accuracy: {:.3f}'.format(acc))

    classifier.train(train_sentences)
    test_sentences, test_ids = classifier.load_file(test_path)

    with open(output_path, 'w', encoding='utf8', newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(['original_label', 'output_label', 'row_id'])
        gold_label_prediction_list = []
        # iterate through all sentences
        for i in range(len(test_sentences)):
            # taking the sentence from the list and assigning it to the sentence variable
            sentence = test_sentences[i][0]
            # using the classifier to predict the label of the corresponding sentence
            predicted_label = classifier.predict(sentence)
            # storing the ground truth in golden_label variable
            golden_label = test_sentences[i][1]
            gold_label_prediction_list.append((golden_label, predicted_label))
            row_id = test_ids[i]
            # writing the row (as requested in the assignment description) to a row of the writer object
            writer.writerow([golden_label, predicted_label, row_id])

    accuracy = classifier.cal_accuracy(gold_label_prediction_list)
    print('=' * 25)
    print('NB test accuracy on the test set is equal to: {}'.format(accuracy))
    y_true = [pair[0] for pair in gold_label_prediction_list]
    y_pred = [pair[1] for pair in gold_label_prediction_list]
    tmp_list = confusion_matrix(y_true, y_pred).tolist()
    print('=' * 25)
    print('Confusion matrix of the NB classifier on the test set: ')
    print(confusion_matrix(y_true, y_pred).transpose())
    conf_matrix_list = confusion_matrix(y_true, y_pred).transpose().tolist()
    first_class_p = conf_matrix_list[0][0]/sum(conf_matrix_list[0])
    second_class_p = conf_matrix_list[1][1]/sum(conf_matrix_list[1])
    third_class_p = conf_matrix_list[2][2]/sum(conf_matrix_list[2])
    fourth_class_p = conf_matrix_list[3][3]/sum(conf_matrix_list[3])
    print('=' * 25)
    print('Precision of the character class is equal to: {:.3f}'.format(first_class_p))
    print('Precision of the director class is equal to: {:.3f}'.format(second_class_p))
    print('Precision of the performer class is equal to: {:.3f}'.format(third_class_p))
    print('Precision of the publisher class is equal to: {:.3f}'.format(fourth_class_p))
    print('=' * 25)
    print('Recall of the character is equal to: {:.3f}'.format(tmp_list[0][0]/sum(tmp_list[0])))
    print('Recall of the director is equal to: {:.3f}'.format(tmp_list[1][1]/sum(tmp_list[1])))
    print('Recall of the performer is equal to: {:.3f}'.format(tmp_list[2][2]/sum(tmp_list[2])))
    print('Recall of the publisher is equal to: {:.3f}'.format(tmp_list[3][3]/sum(tmp_list[3])))
    print('=' * 25)
    # calculating the macro average precision
    macro_avg_p = (first_class_p + second_class_p + third_class_p + fourth_class_p) / 4
    # calculating the micro average precision
    first_class_tp = conf_matrix_list[0][0]
    second_class_tp = conf_matrix_list[1][1]
    third_class_tp = conf_matrix_list[2][2]
    fourth_class_tp = conf_matrix_list[3][3]
    # summing up all of the true positives (pooled tp)
    tp_sum = first_class_tp + second_class_tp + third_class_tp + fourth_class_tp
    # calculating the pooled false positive sum
    fp_sum = conf_matrix_list[1][0] + conf_matrix_list[2][0] + conf_matrix_list[3][0] + \
        conf_matrix_list[0][1] + conf_matrix_list[2][1] + conf_matrix_list[3][1] + \
        conf_matrix_list[0][2] + conf_matrix_list[1][2] + conf_matrix_list[3][2] + \
        conf_matrix_list[0][3] + conf_matrix_list[1][3] + conf_matrix_list[2][3]
    micro_avg_p = tp_sum / (tp_sum + fp_sum)

    print('Micro-average precision is equal to: {:.3f}'.format(micro_avg_p))
    print('Macro-average precision is equal to: {:.3f}'.format(macro_avg_p))
    print('=' * 25)


if __name__ == "__main__":
    main()
