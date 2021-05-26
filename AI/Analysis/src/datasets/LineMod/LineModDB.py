import os
import sys
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')

import numpy as np
from PIL import Image
from torchvision import transforms

from src.config.config import cfg
from src.utils.base_utils import read_ply_model

class LineModDB(object):
    """
    DataLoader for the Linemod dataset

    """
    def __init__(self,
                              linemod_dir: str,
                              obj_name: str = 'all'):
        """
        Initializes a Linemod DataLoader

        Args:
            linemod_dir(str): LineMod データセットのディレクトリパス
            obj_name(str): LineMod データセットに含まれるオブジェクト名
        """

        self.base_dir = linemod_dir
        self.image_shape=(480, 640), # (h, w)
        if obj_name == 'all':
            self.obj_names = cfg.linemod_obj_names
        elif obj_name in cfg.linemod_obj_names:
            self.obj_names = [obj_name]
        else:
            raise ValueError('Invalid object name: {}'.format(obj_name))
        # compute length
        self.lengths = {}
        self.total_length = 0
        for obj_name in self.obj_names:
            length = len(list(filter(lambda x:x.endswith('jpg'), os.listdir(os.path.join(self.base_dir, obj_name, 'data')))))
            self.lengths[obj_name] = length
            self.total_length += length
        self.model_path = os.path.join(linemod_dir,
                                                                        '{}/mesh.ply'.format(obj_name))
        self.old_model_path = os.path.join(linemod_dir,
                                                                                 '{}/OLDmesh.ply'.format(obj_name))
        self.transform_dat_path = os.path.join(self.base_dir,
                                                                                        '{}/transform.dat'.format(obj_name))


    def load_ply_model(self) -> np.ndarray:
        """
        LineMod データセットから `.ply` ファイルに保存された3dモデルデータをスケール変換を行いつつ読み出すための関数

        Return:
            (np.ndarray): numpy配列に変換した 3D モデル
        """
        if os.path.exists(self.model_path):
            return read_ply_model(self.model_path) / 1000.
        else:
            transform = read_transform_dat(self.transform_dat_path)
            old_model = read_ply_model(self.old_model_path) / 1000.
            old_model = np.dot(old_model, transform[:, :3].T) + transform[:, 3]
            return old_model


    def __len__(self):
        return self.total_length


    def __getitem__(self, idx):
        local_idx = idx
        for obj_name in self.obj_names:
            if local_idx < self.lengths[obj_name]:
                obj_pth = os.path.join(self.base_dir, obj_name)
                dat_dir = os.path.join(obj_pth, 'data')
                # image
                img_pth = os.path.join(dat_dir,
                                                                'color{}.jpg'.format(local_idx))
                img = transforms.ToTensor()(Image.open(img_pth).convert('RGB'))
                pose = read_pose(os.path.join(dat_dir, 'rot{}.rot'.format(local_idx)),
                                                      os.path.join(dat_dir, 'tra{}.tra'.format(local_idx)))


def read_pose(rot_path: str, tra_path: str) -> np.matrix:
        """オブジェクトの '.rot' ファイル と '.tra' ファイルの情報を読み出し，スケール変換を加えて返す関数

        Args:
            rot_path str: `.rot` ファイルのパス
            tra_path str: `.tra` ファイルのパス

        Returns:
            T (np.matrix): 姿勢の同次変換行列
        """
        rot = np.loadtxt(rot_path, skiprows=1)
        tra = np.loadtxt(tra_path, skiprows=1) / 100.
        T = np.concatenate([rot, np.reshape(tra, newshape=[3, 1])], axis=-1)
        return T


def read_rotation(filename: str) -> np.matrix:
        """LineMod3Dモデルの回転行列を計算する関数

        Arg:
            filename (str): 単一のLineMod3Dモデルの回転情報が保存された '.rot' ファイルへの絶対パス

        Return:
            R [np.matrix]: 単一のLineMod3Dモデルの回転行列(3, 3)
        """
        with open(filename) as f:
            f.readline()
            R = []
            for line in f:
                R.append(line.split())
            R = np.array(R, dtype=np.float32)
        return R


def read_transform_dat(dat_path: str) -> np.matrix:
        """
        オリジナルの LineMod データセットから `.dat` ファイルに保存された同次変換行列を読み出す関数．このデータは，`OLDmesh.ply` に保存されているオブジェクトの3Dモデルを 'mash.ply' ファイルに保存されている3Dモデルに変換する際に使用される．

        Arg:
            dat_path(str): `.dat` ファイルパス
        Return:
            T (np.matrix): 姿勢の同次変換行列
        """
        T= np.loadtxt(dat_path, skiprows=1)[:, 1]
        T = np.reshape(T, newshape=[3, 4])
        return T


def read_translation(filename: str) -> np.matrix:
        """LineMod3Dモデルの並進行列を計算する関数

        Arg:
            filename (str): 単一のLineMod3Dモデルの並進情報が保存された '.tra' ファイルへの絶対パス

        Return:
            tra [np.matrix]: 単一のLineMod3Dモデルの並進行列(3, 1)
        """
        with open(filename) as f:
            f.readline()
            tra = []
            for line in f:
                tra.append([line.split()[0]])
            tra = np.array(tra, dtype=np.float32)
            tra = tra / np.float32(100) # cm -> m
        return tra


if __name__ == '__main__':
    from glob import glob
    base_dir = cfg.LINEMOD_DIR
    obj_name = 'ape'

    # Setup
    filenames = glob(os.path.join(base_dir, obj_name))
    filename  = filenames[0]

    # ----------------------- #
    # インスタンス生成 #
    # ----------------------- #
    db = LineModDB(linemod_dir = base_dir,
                                       obj_name = obj_name)

    # read_rotation Test
    rot_file = glob(os.path.join(filename, 'data', '*.rot'))[0]
    # rot = read_rotation(rot_file)
    # print('R =', rot)

    # read_translation Test
    tra_file = glob(os.path.join(filename, 'data', '*.tra'))[0]
    # tra = read_translation(tra_file)
    # print('T =', tra)

    # read pose Test
    pose = read_pose(rot_file, tra_file)
    print(pose)

    # read transform data Test
    dat_file = glob(os.path.join(filename, '*.dat'))[0]
    transform_dat = read_transform_dat(dat_file)
    print(transform_dat)