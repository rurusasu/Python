import os
import sys
sys.path.append('..')
from concurrent.futures import ProcessPoolExecutor

from src.config.config import cfg


class PoseTransformer(object):
    rotation_transform = np.array([[1., 0., 0.],
                                                                     [0., -1., 0.],
                                                                     [0., 0., -1.]])
    translation_transforms = {}
    obj_name_to_number = {
        'ape': '001',
        'can': '004',
        'cat': '005',
        'driller': '006',
        'duck': '007',
        'eggbox': '008',
        'glue': '009',
        'holepuncher': '010'
    }
    blender_models={}

    def __init__(self,
                              obj_name,
                              pvnet_linemod_dir,
                              linemod_dir):
        self.obj_name = obj_name
        self.blender_model_path = os.path.join(pvnet_linemod_dir,'{}/{}.ply'.format(obj_name, obj_name))
        self.orig_model_path = os.path.join(linemod_dir,'{}/mesh.ply'.format(obj_name))
        self.model_aligner = ModelAligner(obj_name,pvnet_linemod_dir,linemod_dir)

    def orig_pose_to_blender_pose(self, pose):
        rot, tra = pose[:, :3], pose[:, 3]
        tra = tra + np.dot(rot, self.model_aligner.get_translation_transform())
        rot = np.dot(rot, self.rotation_transform)
        return np.concatenate([rot, np.reshape(tra, newshape=[3, 1])], axis=-1)


def _collect_linemod_set_info(pvnet_linemod_dir: str,
                                                                 obj_name: str,
                                                                 linemod_dir: str,
                                                                 cache_dir: str = './') -> list:
        """
        PVNet LineMod データセット と LineMod データセットの各オブジェクトについて以下の情報を読み出し，'_info.pkl' として保存する関数

        * rgb_pth: JPEG 画像のパス
        * dpt_pth: JPEG 画像に対応する Mask 画像のパス
        * RT: 対象オブジェクトの姿勢情報

        Args:
            pvnet_linemod_dir (str): PVNet_LineMod データセットのディレクトリパス
            obj_name (str): LineMod データセットに含まれるオブジェクト名
            linemod_dir (str): LineMod データセットのディレクトリパス
            cache_dir (str, optional): オブジェクトごとに作成される _info.pkl データの保存先のパス. Defaults to './'.

        Returns:
            database(list): カテゴリごとに上記の情報が保存された辞書のリスト
        """
        database=[]
        if os.path.exists(os.path.join(cache_dir,'{}_info.pkl').format(obj_name)):
            return read_pickle(os.path.join(cache_dir,'{}_info.pkl').format(obj_name))

        _, train_fns = __collect_train_val_test_info(pvnet_linemod_dir, obj_name)
        print('begin generate database {}'.format(obj_name))
        # PVNet LineMod データセットから情報を取得
        # 画像
        rgb_dir = os.path.join(pvnet_linemod_dir,
                                                      obj_name,
                                                      'JPEGImages')
        # マスク画像
        msk_dir = os.path.join(pvnet_linemod_dir,
                                                        obj_name,
                                                        'mask')
        # 姿勢
        rt_dir = os.path.join(linemod_dir,
                                                  obj_name,
                                                  'data')
        img_num = len(os.listdir(rgb_dir))
        for k in range(img_num):
            data = {}
            data['rgb_pth'] = os.path.join(rgb_dir, '{:06}.jpg'.format(k))
            data['dpt_pth'] = os.path.join(msk_dir, '{:04}.png'.format(k))
            if data['rgb_pth'].split('/')[-1] not in train_fns: continue # 訓練で使用するファイル名に含まれていなければ、次のファイルへ

            pose = read_pose(os.path.join(rt_dir, 'rot{}.rot'.format(k)),
                                                  os.path.join(rt_dir, 'tra{}.tra'.format(k)))
            pose_transformer = PoseTransformer(obj_name,
                                                                                           pvnet_linemod_dir,
                                                                                           linemod_dir)
            data['RT'] = pose_transformer.orig_pose_to_blender_pose(pose).astype(np.float32)
            database.append(data)

        print('success generate database {} len {}'.format(obj_name, len(database)))
        save_pickle(database, os.path.join(cache_dir,'{}_info.pkl').format(obj_name))
        return database


def __collect_train_val_test_info(pvnet_linemod_dir: str,
                                                                      obj_name: str) -> list:
    """
    PVNet LineMod データセットのカテゴリごとに保存されている training.txt と test.txt ファイルから訓練とテストとして使用するファイル名のリストを返す関数

    Args:
        pvnet_linemod_dir(str): PVNet_LineMod データセットのディレクトリパス
        obj_name(str): LineMod データセットに含まれるオブジェクト名

    Return:
        test_fns(list): テストデータとして使用される画像のファイル名
        train_fns(list): 訓練データとして使用される画像のフィアル名
    """
    with open(os.path.join(pvnet_linemod_dir, obj_name, 'test.txt'), 'r') as f:
        test_fns = [line.strip().split('/')[-1] for line in f.readlines()]

    with open(os.path.join(pvnet_linemod_dir, obj_name, 'train.txt'), 'r') as f:
        train_fns=[line.strip().split('/')[-1] for line in f.readlines()]

    return test_fns, train_fns


if __name__=="__main__":
    from src.utils.utils import MakeDir

    tmp_dir = MakeDir(cfg.TEMP_DIR, newly=True)

    output_dir = tmp_dir
    pvnet_linemod_dir = cfg.PVNET_LINEMOD_DIR
    linemod_dir = cfg.LINEMOD_DIR
    background_dir='/home/liuyuan/data/SUN2012pascalformat/JPEGImages'
    cache_dir='./'
    fuse_num=10000
    worker_num=2
    prepare_dataset_parallel(output_dir, pvnet_linemod_dir, linemod_dir, fuse_num, background_dir, cache_dir, worker_num)