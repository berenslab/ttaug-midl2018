import numpy as np
import csv
from PIL import Image


class AdvancedReader(object):
    # Extended from StratifiedReader.
    # Generates minibatches that are STRATIFIED w.r.t. class probabilities or OVERSAMPLED (balanced)

    def build_inverted_index(self):
        """ An inverted index is created and used as a joint QUEUE to keep track of examples to sample from each class.
        Once the queues are empty sampling can be reset via rebuilding the inverted index.
        """
        list_count = len(np.unique(list(self.data_dict.values())))
        self.inv_idx = []  # inverted index is a list of lists
        for i in range(list_count):  # create the index structure
            self.inv_idx.append([])

        for k, v in self.data_dict.items():  # k: filename, v: label
            v = int(v)
            k = str(k)
            self.inv_idx[v].append(k)

        for i in range(len(self.inv_idx)):  # shuffle the indexes for randomness in multiple epochs
            np.random.shuffle(self.inv_idx[i])
            self.inv_idx[i].append('monkey')  # monkey is a sentinel value that marks the end of a list

    ###########################################################################

    def estimate_class_probs(self):
        unique_labels = np.unique(list(self.data_dict.values()))
        self.class_probs = np.zeros([len(unique_labels), 1])

        for k, v in self.data_dict.items():  # k: filename, v: label
            v = int(v)
            self.class_probs[v] = self.class_probs[v] + 1
        self.class_probs = np.divide(self.class_probs, len(self.data_dict))

    ###########################################################################

    def retrieve_image(self, file_name):
        with Image.open(self.source + file_name + self.file_type) as img:
            # img_out = np.reshape(np.asarray(img, dtype=np.float32), (1,-1)) # 1x(512*512*3) row vector
            img_out = np.asarray(img, dtype=np.float32)  # 512x512x3 image is read

        return img_out

    ###########################################################################

    def normalize(self, xb):
        epsilon = 1e-14
        xb_centered = np.subtract(xb, self.norm_data['mean'])
        xb_norm = np.divide(xb_centered, np.add(self.norm_data['std_dev'], epsilon))
        return np.asarray(xb_norm, dtype=np.float32)

    ###########################################################################

    def next_batch(self, batch_size, normalize=True, shuffle=True, sampling='stratified'):
        """ Returns a new batch of instance from the source. Bookkeeping is achieved via the inverted index.
        Minimum batchsize should be 8.
        """

        files_to_read = []
        unique_labels = np.unique(list(self.data_dict.values()))  # globally unique labels

        if self.mode == 'train':
            # Set up the batch composure, i.e., number of examples from each class
            if sampling == 'stratified':
                assert batch_size >= 8, "Minimum batch size should be 8 for STRATIFIED minibatches for TRAINING."
                # ceil is to avoid zero examples when batchsize is too small
                examples_per_class = np.ceil(np.multiply(self.class_probs, batch_size))
                total = np.sum(examples_per_class)
                if total > batch_size:  # discount from the largest class: more than 73% of instances come from Class 0
                    diff = total - batch_size
                    examples_per_class[0] = examples_per_class[0] - diff
            elif sampling == 'balanced':
                examples_per_class = np.floor(np.divide(batch_size, len(unique_labels))) * \
                                     np.ones(shape=[len(unique_labels), 1])

        # if mode is test and batch_size is larger than remaining_size, retrieve all
        remaining_size = len(self.test_list)  # not used for training since sampling continues in cycles for training.

        if self.mode == 'train':
            # follow the examples per class information and dequeue accordingly
            for i in range(len(self.inv_idx)):
                count = examples_per_class[i]
                j = 0
                # while the sublist is non-empty and the number of examples to sample is not reached
                while self.inv_idx[i] and j < count:
                    file_name = self.inv_idx[i].pop(0)  # pop from the beginning
                    if file_name == 'monkey':  # monkey is a sentinel value, not a file name
                        # print('Monkey reached in inv.idx %d' % i)
                        np.random.shuffle(self.inv_idx[i])  # shuffle and mark the end of list with monkey
                        self.inv_idx[i].append('monkey')
                        continue
                    files_to_read.append(file_name)
                    self.inv_idx[i].append(file_name)
                    j = j + 1
        elif batch_size < remaining_size:  # have enough remaining to fill the current batch
            for i in range(batch_size):
                files_to_read.append(self.test_list.pop(0))  # pop from the beginning
        else:  # batch_size > remaining_size: not enough test cases left. consume all remaining
            for _ in range(remaining_size):  # consume all remaining test items
                files_to_read.append(self.test_list.pop(0))  # pop from the beginning
            self.exhausted_test_cases = True

        # for all files dequeued, read the images from disk and their labels from dictionary
        images = []
        labels = []

        for file_name in files_to_read:
            images.append(self.retrieve_image(file_name))
            labels.append(self.data_dict[file_name])

        x_batch = np.reshape(np.asarray(images, dtype=np.float32), [-1, 512, 512, 3])
        if normalize:
            x_batch = np.divide(x_batch, 255.)  # normalize into [0,1]
            # x_batch = self.normalize(x_batch)
        y_batch = np.reshape(np.asarray(labels, dtype=np.float32), [-1, 1])

        # data packed as np.arrays. Now, I can shuffle them in concert.
        if self.mode == 'train' and shuffle:
            shuffle_idx = np.random.permutation(len(files_to_read))  # number of images to read in
            x_batch = x_batch[shuffle_idx, :, :, :]
            y_batch = y_batch[shuffle_idx]

        if self.onset_level > 0 and self.onset_level <= 4:
            y_batch = np.greater_equal(y_batch, self.onset_level)
            y_batch = np.asarray(y_batch, dtype=np.int32)
            unique_labels = list([0, 1])  # binary labels for onset thresholding

        # estimate the class probs in the minibatch, which will be used for tackling class imbalance.
        batch_class_probs = np.zeros([len(unique_labels), 1])
        for v in y_batch:
            v = int(v)
            batch_class_probs[v] = batch_class_probs[v] + 1
        batch_class_probs = np.divide(batch_class_probs, len(y_batch))

        K = len(unique_labels)
        # add epsilon to avoid DivByZero
        pos_weights = np.transpose(np.divide(1, np.add(np.multiply(batch_class_probs, K), 1e-14)))

        return x_batch, y_batch, pos_weights

    ###########################################################################

    def __init__(self, source='/gpfs01/berens/user/mayhan/kaggle_dr_data/train_JF_BG_512/',
                 file_type='.jpeg', csv_file='/gpfs01/berens/user/mayhan/kaggle_dr_data/trainLabels.csv',
                 onset_level=-1, mode='train',
                 #norm_data_file='/gpfs01/berens/user/mayhan/Documents/MyPy/KaggleDRproject/KaggleDR/normData_JF_BG.npz'
                 ):
        """ Returns a DataReader object that reads files of type fileType from a given source. Bookkeeping of source files are
            achieved via a dictionary loaded from a .csv file that contains the labels.
        """
        self.source = source
        self.file_type = file_type
        self.csv_file = csv_file
        self.onset_level = onset_level
        self.mode = mode
        # self.norm_data_file = norm_data_file

        self.exhausted_test_cases = False  # used only for test case
        # below are to be populated by the respective methods
        self.data_dict = {}
        self.class_probs = []
        self.inv_idx = []
        self.test_list = []  # same order as in the source .csv file.
        # self.norm_data = np.load(self.norm_data_file)

        with open(csv_file, newline='') as f:
            reader = csv.reader(f)
            next(reader, None)  # skip the header: image, level
            for row in reader:
                self.data_dict[str(row[0])] = row[1]
                if self.mode == 'valtest':  # both val and test instances
                    self.test_list.append(str(row[0]))
                elif self.mode == 'val' and str(row[2]) == 'Public':  # only validation instances: 10906 in total
                    self.test_list.append(str(row[0]))
                elif self.mode == 'test' and str(row[2]) == 'Private':  # only test instances: 42670 in total
                    self.test_list.append(str(row[0]))
        # Dictionary is ready. Now, estimate the class probabilities and build an inverted index to help sampling
        if self.mode == 'train':
            self.estimate_class_probs()
            self.build_inverted_index()

    ###########################################################################
