import datetime

from keras.models import model_from_json
from keras.preprocessing.image import DirectoryIterator
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
import os, os.path
import logging
import time


def files_count(path_fo_folder):
    return sum([len(files) for r, d, files in os.walk(path_fo_folder)])


cur_path = os.path.dirname(os.path.abspath(__file__))

log_frmt = '%(asctime)s | %(name)-15s | %(levelname)-9s | %(message)s'
datefmt = '%Y/%m/%d %H:%M:%S'

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# create console handler and set level to info
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter(log_frmt, datefmt=datefmt)
handler.setFormatter(formatter)
logger.addHandler(handler)

# create error file handler and set level to error
handler = logging.FileHandler(os.path.join("./", "error.log"), "w", encoding=None, delay="true")
handler.setLevel(logging.ERROR)
formatter = logging.Formatter(log_frmt, datefmt=datefmt)
handler.setFormatter(formatter)
logger.addHandler(handler)

# create debug file handler and set level to debug
handler = logging.FileHandler(os.path.join("./", "all.log"), "w")
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(log_frmt, datefmt=datefmt)
handler.setFormatter(formatter)
logger.addHandler(handler)


MAX_BATCH_SIZE = 160


class MyNet(object):
    def __init__(self):
        logging.debug("MyNet::init")
        self.start_time = time.time()
        self.epochs = 1
        self.epoch_counter = 0
        self.bs = 0  # Batch size
        img_size = 50
        self.img_size = img_size
        self.img_width, self.img_height = img_size, img_size  # Image size
        # Target Size
        self.t_s = (self.img_width, self.img_height)
        # Размерность тензора на основе изображения для входных данных в нейронную сеть
        # backend Tensorflow, channels_last
        self.input_shape = (self.img_width, self.img_height, 3)
        self.train_dir = os.path.join(cur_path, 'train_ss')  # Directory with data for training
        self.val_dir = os.path.join(cur_path, 'val_ss')  # Directory with data for validation
        self.test_dir = os.path.join(cur_path, 'test_ss')  # Directory with data for test
        self.test2_dir = os.path.join(cur_path, 'test2_ss')  # Directory with data for test

        self.nb_train_samples = files_count(self.train_dir)  # Number of images for training # 17500
        self.nb_validation_samples = files_count(self.val_dir)  # Number of images for validation # 3750
        self.nb_test_samples = files_count(self.test_dir)  # Number of images for test # 3750
        self.nb_test2_samples = files_count(self.test2_dir)  # Number of images for test # 40
        self.spe = 0
        self.vs = 0
        self.set_batch_size(MAX_BATCH_SIZE)
        logging.info("==================================================")
        self.mul_log_info("Folder sizes:\n "
                          "# Train samples:\t\t{}\n "
                          "# Validation samples:\t{}\n "
                          "# Test samples:\t\t{}\n "
                          "# Test 2 samples:\t\t{}".
                          format(self.nb_train_samples,
                                 self.nb_validation_samples,
                                 self.nb_test_samples,
                                 self.nb_test2_samples))
        logging.info("Train  data dir:\t{}".format(self.train_dir))
        logging.info("Val    data dir:\t{}".format(self.val_dir))
        logging.info("Test   data dir:\t{}".format(self.test_dir))
        logging.info("Test 2 data dir:\t{}".format(self.test2_dir))
        logging.info("--------------------------------------------------")
        logging.info("Epoch :{}".format(self.epochs))
        # logging.info("Batch Size :{}".format(self.bs))
        # logging.info("Steps peer epoch :{} ".format(self.spe))
        # logging.info("Validation steps :{} ".format(self.vs))
        logging.info("Image size :{} ".format(self.t_s))
        logging.info("Input Shape :{} ".format(self.input_shape))
        logging.info("==================================================")
        self.model = None
        self.datagen = None
        self.train_g = None  # type: DirectoryIterator
        self.val_g = None  # type: DirectoryIterator
        self.test_g = None  # type: DirectoryIterator
        self.test2_g = None  # type: DirectoryIterator
        self.__score = 0
        self.__date = datetime.datetime.now().strftime("%y%m%d_%H%M")
        self.create_network()
        self.create_image_generator()

    def mul_log_info(self, msg):
        tmp_arr = msg.split("\n")
        for s in tmp_arr:
            logging.info(s)

    def create_network(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=self.input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        self.model = model

    def compile_network(self, model=None):
        logging.info("Start Compiling neural network")
        if model is not None:
            logging.warning("Provided model, replace and compile network by provided")
            self.model = model
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        logging.info("End Compiling neural network")
        return self.model

    def create_image_generator(self):
        logging.info("Start Creating image generator")
        self.datagen = ImageDataGenerator(rescale=1. / 255)
        logging.info("End Creating image generator")

    def load_images(self):
        fnc = self.datagen.flow_from_directory  # type: DirectoryIterator.__init__()
        logging.info("Data generator for TRAIN based on images in directory")
        self.train_g = fnc(self.train_dir, target_size=self.t_s, batch_size=self.bs, class_mode='binary')

        logging.info("Data generator for VAL based on images in directory")
        self.val_g = fnc(self.val_dir, target_size=self.t_s, batch_size=self.bs, class_mode='binary')

        logging.info("Data generator for TEST based on images in directory")
        self.test_g = fnc(self.test_dir, target_size=self.t_s, batch_size=self.bs, class_mode='binary')

        logging.info("Data generator for TEST 2 based on images in directory")
        self.test2_g = fnc(self.test2_dir, target_size=self.t_s, batch_size=self.bs, class_mode='binary')

    def train_model(self):
        # Train MODEL with generators usage
        # train_generator - data generator for training
        # validation_data - generators for validation
        st = time.time()
        logging.info(
            "Starting to train the model with generators usage. "
            "Steps peer epoch:{} Validations steps:{}".format(self.spe, self.vs))
        while self.get_network_score(True) <= 99:
            # self.get_network_score()
            lst = time.time()
            if self.epoch_counter == 0:
                self.__print_batch_size_info()
            logging.info("---------------------------------------------------------------------------")
            logging.info(" ------------------- Starting fit_generator for epoch # {} -----------------".
                         format(self.epoch_counter+1))
            self.model.fit_generator(self.train_g, steps_per_epoch=self.spe, epochs=self.epochs,
                                     validation_data=self.val_g, validation_steps=self.vs, workers=8)
            self.epoch_counter += self.epochs
            self.save_network()
            logging.info(" --> Training finished in {} s , total {} s - Total epochs pass: {}.".
                         format(round(time.time() - lst, 2), round(time.time() - st, 2), self.epoch_counter))
            self.set_batch_size(self.bs / 1.2)
            time.sleep(2)
        logging.info("Total train finished in {} s".format(round(time.time() - st, 2)))

    def get_network_score(self, from_memory=False):
        if from_memory:
            return self.__score
        scores1 = self.model.evaluate_generator(self.test_g, max(1, self.nb_test_samples // self.bs))
        __score1 = round(scores1[1] * 100, 6)
        scores2 = self.model.evaluate_generator(self.test2_g, max(1, self.nb_test2_samples // self.bs))
        __score2 = round(scores2[1] * 100, 6)
        total_score = round((__score1 + __score2) / 2.0, 2)
        logging.info(
            "Accuracy on TEST 1/2 data is: {}% / {}%. [ Final score {}% ]. ".format(__score1, __score2, total_score))
        self.__score = total_score
        return self.__score

    def set_batch_size(self, new_value):
        old_bs = self.bs
        if new_value < 1:
            logging.error("Bad batch size, setting default as {} to be batch size".format(MAX_BATCH_SIZE))
            new_value = MAX_BATCH_SIZE
        self.bs = round(new_value, 0)
        if self.bs == old_bs:
            logging.error("Old BS = New BS  <-> {}={}. Set MAX".format(old_bs, self.bs))
            self.set_batch_size(MAX_BATCH_SIZE)
            # logging.error("Old BS = New BS  <-> {}={}. Dividing by 2".format(old_bs, self.bs))
            # self.set_batch_size(old_bs/2)
        self.spe = self.nb_train_samples // self.bs  # Steps peer epoch
        self.vs = self.nb_validation_samples // self.bs  # Validation steps
        self.__print_batch_size_info()

    def __print_batch_size_info(self):
        logging.info(" ---> Batch Size :{}. Steps peer epoch :{}. Validation steps :{}".
                     format(self.bs, self.spe, self.vs))

    def finish(self):
        logging.info("Checking the accuracy of network job with Test generator")

        end_time = time.time()
        logging.info("Accuracy on TEST data is: {}%. Finished in {} s".format(self.get_network_score(),
                                                                              round(end_time - self.start_time, 2)))

    def __get_file_postfix(self):
        return "mnmodel_{}_ep_{}__scr_{}_img_{}".format(self.__date, self.epoch_counter,
                                                        int(round(self.get_network_score(), 0)), self.img_size)

    def save_network(self, save_json=True, save_yml=False):
        f_str = self.__get_file_postfix()
        if save_json:
            model_json = self.model.to_json()
            json_file = open("{}.json".format(f_str), "w")
            json_file.write(model_json)
            json_file.close()
        if save_yml:
            model_yaml = self.model.to_yaml()
            yaml_file = open("{}.yml".format(f_str), "w")
            yaml_file.write(model_yaml)
            yaml_file.close()

        self.model.save_weights("{}.h5".format(f_str))

    @staticmethod
    def __check_file_exist(file_path):
        return os.path.exists(file_path) and os.path.isfile(file_path)

    @staticmethod
    def get_file_full_path(file_path):
        return os.path.abspath(file_path)

    def load_network(self, file_path_json, file_path_h5=None):
        if file_path_h5 is None and not self.__check_file_exist(file_path_json):
            no_ext_name = file_path_json
            logging.warning(" ---> Try simple load [{}]...".format(no_ext_name))
            file_path_json = "{}.json".format(no_ext_name)
            file_path_h5 = "{}.h5".format(no_ext_name)

        if not self.__check_file_exist(file_path_json):
            logging.error("Provided file not exist {}".format(file_path_json))
            return False
        if not self.__check_file_exist(file_path_h5):
            logging.error("Provided file not exist {}".format(file_path_h5))
            return False
        file_path_json = self.get_file_full_path(file_path_json)
        file_path_h5 = self.get_file_full_path(file_path_h5)
        logging.info("Loading network:")
        logging.info("{}".format(file_path_json))
        logging.info("{}".format(file_path_h5))
        # Load data about arch from JSON file
        json_file = open(file_path_json, "r")
        loaded_model_json = json_file.read()
        json_file.close()
        # Create model from loaded data
        loaded_model = model_from_json(loaded_model_json)  # type: Sequential
        # Load weights
        loaded_model.load_weights(file_path_h5)
        # Compiling the model
        self.compile_network(loaded_model)
        # loaded_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        # Checks the model with test data
        # scores = loaded_model.evaluate(X_test, Y_test, verbose=0)
        self.get_network_score()
        # scores = self.model.evaluate_generator(self.train_g, max(1, self.nb_train_samples // self.bs))
        # __score1 = round(scores[1] * 100, 6)
        # logging.info("Accuracy of model based on test data is: {}%".format(__score1))
        # self.model = loaded_model


nt = MyNet()

nt.load_images()

# mnist_model_66__81.0_img_150.h5
# mnist_model_66__81.0_img_150.json
# mnist_model_66__81.0_img_150.yml
nt.load_network("mnmodel_180518_1046_ep_7__scr_77_img_50")

# nt.compile_network()
nt.train_model()
nt.finish()
nt.save_network()
