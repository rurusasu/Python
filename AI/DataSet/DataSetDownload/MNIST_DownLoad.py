import urllib.request
import gzip
import numpy as np
import pickle

dataset_dir = 'D:/My_programing/python/AI/DataSet/MNIST'  # データを保存する場所
# ディレクトリ構成
key_file = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz'
}


# データをダウンロードする
def MNIST_DownLoad(SaveDir, key_file):
    url_base = 'http://yann.lecun.com/exdb/mnist/'

    for v in key_file.values():
        file_path = SaveDir + '/' + v
        urllib.request.urlretrieve(url_base + v, file_path)


def LoadFile(LoadDir, file_tag, offset=None):
    file_path = LoadDir + '/' + file_tag
    with gzip.open(file_path, 'rb') as f:
        if offset is None:
            data = np.frombuffer(f.read(), np.uint8)
        elif type(offset) is int:
            data = np.frombuffer(f.read(), np.uint8, offset=offset)
    
    return data

# Pickle形式で保存する
def SavedPickle(SaveDir, Name, dataset):
    save_file = SaveDir + '/' + Name + '.pkl'
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1) # -1は最も高いプロトコルバージョンで保存することを指定している．

def LoadPickle(LoadDir, FileName):
    load_path = LoadDir + '/' + FileName + '.pkl'
    with open(load_path, 'wb') as f:
        dataset = pickle.load(f)

        return dataset



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    #MNIST_DownLoad(dataset_dir)
    dataset = {}
    #data = LoadFile(dataset_dir, key_file['train_img'], 16).reshpe(-1, 784)
    dataset['train_img'] = LoadFile(dataset_dir, key_file['train_img'], 16).reshape(-1, 784)
    dataset['train_label'] = LoadFile(dataset_dir, key_file['train_label'], 8)
    dataset['test_img'] = LoadFile(dataset_dir, key_file['test_img'], 16).reshape(-1, 784)
    dataset['test_label'] = LoadFile(dataset_dir, key_file['test_label'], 8)

    #SavedPickle(dataset_dir, 'mnist', dataset)
    example = dataset['train_img'][0].reshape((28, 28))
    example = plt.imshow(example)
    plt.show()