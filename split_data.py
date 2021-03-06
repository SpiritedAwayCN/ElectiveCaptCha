import os
import random
from shutil import copyfile

dataset_dir = 'dataset'

training_dir = 'train'
test_dir = 'test'
split_rate = 0.2

if not os.path.exists(training_dir):
    os.mkdir(training_dir)
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

for cate_name in os.listdir(dataset_dir):
    training_dir_c = os.path.join(training_dir, cate_name)
    test_dir_c = os.path.join(test_dir, cate_name)
    dataset_dir_c = os.path.join(dataset_dir, cate_name)

    if not os.path.exists(training_dir_c):
        os.mkdir(training_dir_c)
    if not os.path.exists(test_dir_c):
        os.mkdir(test_dir_c)
    
    shuffled = os.listdir(dataset_dir_c)
    data_size = len(shuffled)
    random.shuffle(shuffled)

    test_num = int(data_size * split_rate)
    for i in range(test_num):
        copyfile(os.path.join(dataset_dir_c, shuffled[i]), os.path.join(test_dir_c, shuffled[i]))
    for i in range(test_num + 1, data_size):
        copyfile(os.path.join(dataset_dir_c, shuffled[i]), os.path.join(training_dir_c, shuffled[i]))
