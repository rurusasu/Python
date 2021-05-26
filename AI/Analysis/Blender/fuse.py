import os
import sys

from PIL import Image
sys.path.append('..')
import time
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from logging import getLogger

import cv2
import numpy as np

from Blender.render_base_utils import ModelAligner, PoseTransformer
from src.config.config import cfg
from src.datasets.LineMod.LineModDB import read_pose
from src.utils.base_utils import read_mask_np, read_rgb_np, read_pickle, save_pickle


def _prepare_dataset_single(output_dir: str,
                                                             idx: int,
                                                             pvnet_linemod_dir: str,
                                                             linemod_dir: str,
                                                             bg_imgs_dir: str,
                                                             obj_names: list = cfg.linemod_obj_names,
                                                             cache_dir: str = cfg.TEMP_DIR,
                                                             seed: int = 0):
    """
    オブジェクトの画像と別の背景画像を合成する関数

    Args:
        output_dir(str): 合成された画像を保存するディレクトリへのパス
        idx(int): 保存されるデータに振られる番号
        pvnet_linemod_dir (str): PVNet で作成された LINEMODデータセットが保存されているディレクトリパス
        linemod_dir (str): オリジナルのLINEMODデータセットが保存されているディレクトリパス
        bg_imgs_dir (str): 背景画像として使用する画像の読み出し先のパス
        obj_names(list optional): 合成で使用するデータセットのクラス名．Defaults to cfg.linemod_obj_names.
        cache_dir (str, optional): オブジェクトごとに作成される `obj_name_info.pkl` データや合成画像の背景として使用される画像のパスを保存した `background_inf.pkl`の保存先のパス. Defaults to 'cfg.TEMP_DIR'.
        seed (int, optional): オブジェクトを配置する際に使用する乱数のseed値. Defaults to 0.
    """
    time_begin = time.time()
    np.random.seed(seed)
    rgbs,masks,begins,poses = [], [], [], []
    image_dbs = {}
    for obj_id,obj_name in enumerate(obj_names):
        image_dbs[obj_id] = _collect_linemod_set_info(pvnet_linemod_dir,
                                                                                                             obj_name,
                                                                                                             linemod_dir,
                                                                                                             cache_dir)

    for obj_id, _ in enumerate(obj_names):
        rgb, mask, begin, pose = __randomly_sample_foreground(image_dbs[obj_id], pvnet_linemod_dir)
        mask *= obj_id+1
        rgbs.append(rgb)
        masks.append(mask)
        begins.append(begin)
        poses.append(pose)

    background = __randomly_read_background(bg_imgs_dir, cache_dir)

    fuse_img, fuse_mask, fuse_begins = _fuse_regions(rgbs, masks, begins, background, 480, 640)

    _save_fuse_data(output_dir, idx, fuse_img, fuse_mask, fuse_begins, poses)
    print('{} cost {} s'.format(idx, time.time()-time_begin))


def prepare_dataset_parallel(output_dir: str,
                                                              bg_imgs_dir: str,
                                                              linemod_dir: str,
                                                              pvnet_linemod_dir: str,
                                                              obj_names: list = cfg.linemod_obj_names,
                                                              fuse_num: int=50,
                                                              cache_dir: str = cfg.TEMP_DIR,
                                                              worker_num=8):
    """
    データセットの準備を並列して行う関数

    Args:
        output_dir (str):
        bg_imgs_dir (str): 背景画像として使用する画像が保存されているディレクトリのパス
        linemod_dir (str): オリジナルのLINEMODデータセットが保存されているディレクトリパス
        pvnet_linemod_dir (str): PVNet で作成された LINEMODデータセットが保存されているディレクトリパス
        obj_names(list optional): 合成で使用するデータセットのクラス名．Defaults to cfg.linemod_obj_names.
        fuse_num (int optional): 作成される合成画像枚数．Defaults to 50.
        cache_dir (str, optional): オブジェクトごとに作成される `obj_name_info.pkl` データや合成画像の背景として使用される画像のパスを保存した `background_inf.pkl`の保存先のパス. Defaults to 'cfg.TEMP_DIR'.
        worker_num (int, optional): 並列処理で使用するワーカー数. Defaults to 8.
    """
    futures=[]
    with ProcessPoolExecutor(max_workers = worker_num) as executor:
        for idx in np.arange(fuse_num):
            seed=np.random.randint(5000)
            future = executor.submit(_prepare_dataset_single,
                                                                output_dir,
                                                                idx,
                                                                pvnet_linemod_dir,
                                                                linemod_dir,
                                                                bg_imgs_dir,
                                                                obj_names,
                                                                cache_dir,
                                                                seed)
            futures.append(future)

        for f in futures:
            f.result()


def _collect_linemod_set_info(linemod_dir: str,
                                                                 pvnet_linemod_dir: str,
                                                                 obj_name: str,
                                                                 cache_dir: str = cfg.TEMP_DIR) -> list:
        """
        PVNet LineMod データセット と LineMod データセットの各オブジェクトについて以下の情報を読み出し，'_info.pkl' として保存する関数

        * rgb_pth: JPEG 画像のパス
        * dpt_pth: JPEG 画像に対応する Mask 画像のパス
        * RT: 対象オブジェクトの姿勢情報

        Args:
            linemod_dir (str): オリジナルのLINEMODデータセットが保存されているディレクトリパス
            pvnet_linemod_dir (str): PVNet で作成された LINEMODデータセットが保存されているディレクトリパス
            obj_name (str): LineMod データセットに含まれるオブジェクト名
            cache_dir (str, optional): オブジェクトごとに作成される _info.pkl データの保存先のパス. Defaults to 'cfg.TEMP_DIR'.

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
            pose_transformer = PoseTransformer(linemod_dir = linemod_dir,
                                                                                           pvnet_linemod_dir = pvnet_linemod_dir,
                                                                                           obj_name = obj_name)
            data['RT'] = pose_transformer.orig_pose_to_blender_pose(pose).astype(np.float32)
            database.append(data)

        print('success generate database {} len {}'.format(obj_name, len(database)))
        save_pickle(database, os.path.join(cache_dir, '{}_info.pkl').format(obj_name))
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
        train_fns = [line.strip().split('/')[-1] for line in f.readlines()]

    return test_fns, train_fns


def _fuse_regions(rgbs: list,
                                      masks: list,
                                      begins: list,
                                      background: np.ndarray,
                                      th: int,
                                      tw: int):
    """
    RGB背景画像とそのマスク画像を (tw, th) にリサイズし，その上にオブジェクトのRGB画像とそのマスク画像を合成する関数．返り値は，合成されたRGB画像とそのマスク画像と合成されたオブジェクトの位置

    Args:
        rgbs(list): 合成するオブジェクトの ndarray が保存されたリスト
        masks(list): 合成するオブジェクトのマスク画像の ndarray が保存されたリスト
        begins(list):
        background(np.ndarray): 背景の ndarray
        th(int): 背景画像のリサイズ後の高さ
        tw(int): 背景画像のリサイズ後の幅

    Returns:
        fuse_img(np.ndarray): 合成後の画像 [max=255, min=0]
        fuse_mask(np.ndarray): 合成後のマスク画像 [max=1 min=0]
        begins(list)
    """
    fuse_order = np.arange(len(rgbs))
    np.random.shuffle(fuse_order)
    fuse_img = background
    fuse_img = cv2.resize(fuse_img, (tw,th), interpolation = cv2.INTER_LINEAR) # Bilinearによる内挿法を用いてリサイズ
    fuse_mask = np.zeros([fuse_img.shape[0], fuse_img.shape[1]], np.int32)

    for idx in fuse_order:
        rh,rw = masks[idx].shape
        # 背景画像より外にオブジェクトが配置されないように調整
        bh = np.random.randint(0, fuse_img.shape[0]-rh)
        bw = np.random.randint(0, fuse_img.shape[1]-rw)

        silhouette = masks[idx] > 0
        out_silhouette = np.logical_not(silhouette)
        fuse_mask[bh:bh+rh,bw:bw+rw] *= out_silhouette.astype(fuse_mask.dtype)
        fuse_mask[bh:bh+rh,bw:bw+rw] += masks[idx]

        fuse_img[bh:bh+rh,bw:bw+rw] *= out_silhouette.astype(fuse_img.dtype)[:, :, None]
        fuse_img[bh:bh+rh,bw:bw+rw] += rgbs[idx]

        begins[idx][0] =- begins[idx][0] + bh
        begins[idx][1] =- begins[idx][1] + bw

    return fuse_img, fuse_mask, begins


def __randomly_sample_foreground(image_db: dict, pvnet_linemod_dir: str):
    """
    画像合成の際に背景上に配置するオブジェクトを `_collect_linemod_set_info` で保存した下記の情報をもとに読み出し，rgb画像，mask画像，対応するオブジェクトの姿勢を返す関数

    image_db:
        * rgb_pth: JPEG 画像のパス
        * dpt_pth: JPEG 画像に対応する Mask 画像のパス
        * RT: 対象オブジェクトの姿勢情報

    Aegs:
        image_db(dict): pvnet_linemod の rgb_pth, dpt_pth, と orig_pose_to_blender_pose によって変換された 同次変換行列が格納された辞書．
        例: {'rgb_pth': './AI/Analysis/data/PVNet_linemod/ape/JPEGImages/000914.jpg',
                'dpt_pth': './AI/Analysis/data/PVNet_linemod/ape/mask/0914.png',
                'RT': array([[-0.0536252 ,  0.975377  , -0.213925  ,  0.00464225],
                                        [ 0.962547  , -0.00652554, -0.271037  , -0.07789672],
                                        [-0.265759  , -0.220448  , -0.938496  ,  0.6899649 ]],dtype=float32)},
        pvnet_linemod_dir (str): PVNet_LineMod データセットのディレクトリパス

    Return:
        rgb(np.ndarray): rgb画像の ndarray 配列 [max = 255, min = 0]
        mask(np.ndarray): mask画像の ndarray 配列 [max = 1, min = 0]
        begin(list):
        pose(np.ndarray): オブジェクトの姿勢 [RT| 3x4行列]
    """
    idx = np.random.randint(0, len(image_db))
    rgb_pth = os.path.join(pvnet_linemod_dir,image_db[idx]['rgb_pth'])
    dpt_pth = os.path.join(pvnet_linemod_dir,image_db[idx]['dpt_pth'])
    rgb = read_rgb_np(rgb_pth)
    mask = read_mask_np(dpt_pth)
    mask = np.sum(mask, 2) > 0
    mask = np.asarray(mask, np.int32)

    hs, ws = np.nonzero(mask)
    hmin, hmax = np.min(hs), np.max(hs)
    wmin, wmax  = np.min(ws), np.max(ws)

    mask = mask[hmin:hmax, wmin:wmax]
    rgb = rgb[hmin:hmax, wmin:wmax]

    rgb *= mask.astype(np.uint8)[:, :, None]
    begin = [hmin, wmin]
    pose = image_db[idx]['RT']

    return rgb, mask, begin, pose


def _save_fuse_data(output_dir: str,
                                          idx: int,
                                          fuse_img: np.ndarray,
                                          fuse_mask: np.ndarray,
                                          fuse_begins,
                                          fuse_poses: list):
    """
    合成したデータを保存する関数

    Args:
        output_dir(str): データを保存するディレクトリのパス
        idx(int): 作成したデータの番号
        fuse_img(np.ndarray): 合成された画像の ndarray
        fuse_mask(np.ndarray): 合成された画像のマスク画像の ndarray
        fuse_begins()
        fuse_poses(list): 合成された画像に使用されているオブジェクトの姿勢 [RT| 3x4行列] のリスト
    """
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir,
                                                        '{}_rgb.jpg'.format(idx)),
                                                        fuse_img)
    fuse_mask=fuse_mask.astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir,
                                                        '{}_mask.png'.format(idx)),
                                                        fuse_mask)
    save_pickle([np.asarray(fuse_begins,np.int32),
                                np.asarray(fuse_poses,np.float32)],
                              os.path.join(output_dir,'{}_info.pkl'.format(idx)))


if __name__=="__main__":
    from src.utils.utils import MakeDir

    tmp_dir = MakeDir(cfg.TEMP_DIR, newly=True)
    cache_dir= tmp_dir
    output_dir = tmp_dir
    idx = 1
    pvnet_linemod_dir = cfg.PVNET_LINEMOD_DIR
    linemod_dir = cfg.LINEMOD_DIR
    obj_name = 'ape'
    bg_imgs_dir = cfg.TEST_IMG_ORG_DIR
    fuse_num = 5
    worker_num=2

    data_base = _collect_linemod_set_info(pvnet_linemod_dir = pvnet_linemod_dir,
                                                                                       obj_name = obj_name,
                                                                                       linemod_dir = linemod_dir)


    prepare_dataset_parallel(output_dir = output_dir,
                                                          pvnet_linemod_dir = pvnet_linemod_dir,
                                                          linemod_dir = linemod_dir,
                                                          bg_imgs_dir = bg_imgs_dir,
                                                          fuse_num = fuse_num,
                                                          cache_dir = cache_dir,
                                                          worker_num = worker_num)
