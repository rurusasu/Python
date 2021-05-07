import sys

sys.path.append(".")
sys.path.append("..")

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class LineModDataset(Dataset):
    """
    DataLoader for the Linemod dataset

    """
    def __init__(self,
                             base_dir: str,
                             object_name: str='all'):
        """
        Initializes a Linemod DataLoader
        Args:
            base_dir (str): path to the Linemod dataset
            object_name (str): Integer object id of the Linemod object on which to generate data

        """
        linemod_objects = [
            "ape",
            "benchviseblue",
            "bowl",
            "cam",
            "can",
            "cat",
            "cup",
            "driller",
            "duck",
            "eggbox",
            "glue",
            "holepuncher",
            "iron",
            "lamp",
            "phone"
        ]
        self.base_dir = base_dir
        self.image_shape=(480, 640), # (h, w)
        if object_name = 'all':
            self.object_names = linemod_objects
        elif object_name in linemod_objects:
            self.object_names = [object_name]
        else:
            raise ValueError('Invalid object name: {}'.format(object_name))
        # compute length
        self.lengths = {}
        self.total_length = 0
        for object_name in object_names:
            length = len(list(filter(lambda x:x.endswith('jpg'), os.listdir(os.path.join(base_dir, object_name, 'data')))))
            self.lengths[object_name] = length
            self.total_length += length

    def read_3d_points(self, filename):
        with open(filename) as f:
            in_vertex_list = False
            vertices = []
            in_mm = False
            for line in f:
                if in_vertex_list:
                    vertex = line.split()[:3]
                    vertex = np.array([[float(vertex[0])],
                                                          [float(vertex[1])],
                                                          [float(vertex[2])]],
                                                          dtype=np.float32)
                    if in_mm:
                        vertex = vertex / np.float32(10) # mm -> cm
                    vertex = vertex / np.float32(100) # cm -> m
                    vertices.append(vertex)
                    if len(vertices) >= vertex_count:
                        break
                elif line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif line.startswith('end_header'):
                    in_vertex_list = True
                elif line.startswith('element face'):
                    in_mm = True
        return vertices

    def __len__(self):
        return self.total_length

    def read_rotation(self, filename: str) -> np.matrix:
        """LineMod3Dモデルの回転行列を計算する関数

        Arg:
            filename (str): 単一のLineMod3Dモデルの回転情報が保存された '.rot' ファイルへの絶対パス

        Return:
            R [np.matrix]: 単一のLineMod3Dモデルの(3, 3)回転行列
        """
        with open(filename) as f:
            f.readline()
            R = []
            for line in f:
                R.append(line.split())
            R = np.array(R, dtype=np.float32)
        return R

    def read_translation(self, filename: str) -> np.matrix:
        """LineMod3Dモデルの並進行列を計算する関数

        Arg:
            filename (str): 単一のLineMod3Dモデルの並進情報が保存された '.tra' ファイルへの絶対パス

        Return:
            T [np.matrix]: 単一のLineMod3Dモデルの(3, 3)並進行列
        """
        with open(filename) as f:
            f.readline()
            T = []
            for line in f:
                T.append([line.split()[0]])
            T = np.array(T, dtype=np.float32)
            T = T / np.float32(100) # cm -> m
        return T

    def __getitem__(self, idx):
        local_idx = idx
        for object_name in self.object_names:
            if local_idx < self.lengths[object_name]:
                # image
                image_name = os.path.join(self.base_dir,
                                                                          object_name,
                                                                          'data',
                                                                          'color{}.jpg'.format(local_idx))
                image = transforms.ToTensor()(Image.open(image_name).convert('RGB'))
                # pose
                R_name = os.path.join(self.base_dir,
                                                                object_name,
                                                                'data',
                                                                'rot{}.rot'.format(local_idx))
                R = self.read_rotation(R_name)
                T_name = os.path.join(self.base_dir,
                                                                object_name,
                                                                'data',
                                                                'tra{}.tra'.format(local_idx))
                T = self.read_translation(T_name)