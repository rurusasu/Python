import os
import sys
sys.path.append('..')
sys.path.append('../../')

import numpy as np

from src.config.config import cfg
from src.datasets.LineMod.LineModDB import LineModDB
from src.utils.base_utils import read_ply_model

class ModelAligner(object):
    """
    オブジェクトの姿勢をオリジナルの LineMod のものと Blender 上のものに変換するクラス
    """
    rotation_transform = np.array([[1., 0., 0.],
                                   [0., -1., 0.],
                                   [0., 0., -1.]])
    translation_transforms = {
        # 'cat': np.array([-0.00577495, -0.01259045, -0.04062323])
    }
    intrinsic_matrix = {
        'linemod': np.array([[572.4114, 0., 325.2611],
                              [0., 573.57043, 242.04899],
                              [0., 0., 1.]]),
        # 'blender': np.array([[280.0, 0.0, 128.0],
        #                      [0.0, 280.0, 128.0],
        #                      [0.0, 0.0, 1.0]]),
        'blender': np.array([[700.,    0.,  320.],
                             [0.,  700.,  240.],
                             [0.,    0.,    1.]])
    }

    def __init__(self,
                              obj_name: str,
                              pvnet_linemod_dir: str,
                              linemod_dir: str):
        """
        ModelAligner の初期化関数

        Args:
            obj_name(str): LineMod データセットに含まれるオブジェクト名
            pvnet_linemod_dir(str): PVNet_LineMod データセットのディレクトリパス
            linemod_dir(str): LineMod データセットのディレクトリパス
        """
        self.obj_name = obj_name
        self.linemod_dir = linemod_dir
        self.LineModDB = LineModDB(linemod_dir = self.linemod_dir,
                                                                       obj_name=self.obj_name)
        self.blender_model_path = os.path.join(pvnet_linemod_dir,
                                                                                           '{}/{}.ply'.format(obj_name, obj_name))
        self.R_p2w, self.t_p2w, self.s_p2w = self.setup_p2w_transform()

    @staticmethod
    def setup_p2w_transform():
        transform1 = np.array([[0.161513626575, -0.827108919621, 0.538334608078, -0.245206743479],
                                                        [-0.986692547798, -0.124983474612, 0.104004733264, -0.050683632493],
                                                        [-0.018740313128, -0.547968924046, -0.836288750172, 0.387638419867]])
        transform2 = np.array([[0.976471602917, 0.201606079936, -0.076541729271, -0.000718327821],
                                                        [-0.196746662259, 0.978194475174, 0.066531419754, 0.000077120210],
                                                        [0.088285841048, -0.049906700850, 0.994844079018, -0.001409600372]])

        R1 = transform1[:, :3]
        t1 = transform1[:, 3]
        R2 = transform2[:, :3]
        t2 = transform2[:, 3]

        # printer system to world system
        t_p2w = np.dot(R2, t1) + t2
        R_p2w = np.dot(R2, R1)
        s_p2w = 0.85
        return R_p2w, t_p2w, s_p2w

    def pose_p2w(self, RT):
        """
        ある姿勢行列を別の姿勢行列に変換する関数

        Arg:
            RT(np.matrix): 変換前の姿勢行列

        Return:
            (np.matrix): 変換後の姿勢行列
        """
        t,R = RT[:,3],RT[:,:3]
        R_w2c = np.dot(R, self.R_p2w.T)
        t_w2c = -np.dot(R_w2c,self.t_p2w)+self.s_p2w*t
        return np.concatenate([R_w2c, t_w2c[:,None]],1)

    def get_translation_transform(self) -> np.ndarray:
        """
        Blender 上で作成したオブジェクトの3Dモデルとオリジナルの LineMod オブジェクトの3Dモデルを読み出し，この2つのオブジェクトの相対的並進配列を返す関数

        Return:
            (np.ndarray): 2つのオブジェクトの相対的並進配列
        """
        if self.obj_name in self.translation_transforms:
            return self.translation_transforms[self.obj_name]

        blender_model = read_ply_model(self.blender_model_path)
        orig_model = self.LineModDB.load_ply_model()
        blender_model = np.dot(blender_model, self.rotation_transform.T)
        translation_transform = np.mean(orig_model, axis=0) - np.mean(blender_model, axis=0)
        self.translation_transforms[self.obj_name] = translation_transform

        return translation_transform