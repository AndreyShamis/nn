import os
import logging
from datetime import datetime
from net.my_net import MyNet


run_prefix = datetime.now().strftime("%y%m%d_%H%M")
log_frmt = '%(asctime)s | %(name)-10s | %(levelname)-9s | %(message)s'
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
handler = logging.FileHandler(os.path.join("./", "error_{}.log".format(run_prefix)), "w", encoding=None, delay="true")
handler.setLevel(logging.ERROR)
formatter = logging.Formatter(log_frmt, datefmt=datefmt)
handler.setFormatter(formatter)
logger.addHandler(handler)

# create debug file handler and set level to debug
handler = logging.FileHandler(os.path.join("./", "all_{}.log".format(run_prefix)), "w")
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(log_frmt, datefmt=datefmt)
handler.setFormatter(formatter)
logger.addHandler(handler)

loaded = False
logging.info("Starting execution : ID {}".format(run_prefix))
nt = MyNet(epochs=50, date_time=run_prefix)

nt.load_images()

nt.load_network("mnmodel_180520_1156_ep_15__scr_85_img_75")
#
# #direc = "./data/test2_ss/dogs"
# direc = "./dogs"
# cats_found = 0
# dogs_found = 0
# unknown_found = 0
# counter = 0
# for filename in os.listdir(direc):
#     counter += 1
#     if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
#         # print(os.path.join(directory, filename))
#         filik = os.path.join(direc, filename)
#         res = nt.get_predict(filik)
#         if res == 0:
#             title, ext = os.path.splitext(os.path.basename(filename))
#             os.rename(filik, os.path.join("./dogs_bad", "dog.bad.{}{}".format(counter, ext)))
#         if res == 0:
#             cats_found +=1
#         if res == 1:
#             dogs_found +=1
#         if res is None:
#             unknown_found +=1
#         continue
#     else:
#         continue
# logging.info("Cats found {}".format(cats_found))
# logging.info("Dogs found {}".format(dogs_found))
# logging.info("Unknown found {}".format(unknown_found))

# nt.get_predict("cat.30001.jpg")
# nt.get_predict("cat.30002.jpg")
# nt.get_predict("cat.30003.jpg")
# nt.get_predict("cat.30004.jpg")
# nt.get_predict("cat.30005.jpg")
# nt.get_predict("dog.30001.jpg")
# nt.get_predict("dog.30002.jpg")
# nt.get_predict("dog.30003.jpg")
# nt.get_predict("dog.30004.jpg")
# nt.get_predict("dog.30005.jpg")

nt.train_model()
nt.finish()
nt.save_network()

#   Allocation of X exceeds 10% of system memory
# export TF_CPP_MIN_LOG_LEVEL=2
