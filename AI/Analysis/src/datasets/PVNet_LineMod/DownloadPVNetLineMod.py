import os
import shutil
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')

from config.config import cfg
from src.utils.download_utils import DownloadZip, UnpackZip
from src.utils.utils import MakeDir


"""
LINEMOD Dataset 製作者の Stefan Hinterstoißer さんのサイトから
LINEMOD Dataset の元データをダウンロードする。
"""

# Stefan Hinterstoißer さんのサイトのURL
base_url = 'https://zjueducn-my.sharepoint.com/personal/pengsida_zju_edu_cn/_layouts/15/onedrive.aspx?originalPath=aHR0cHM6Ly96anVlZHVjbi1teS5zaGFyZXBvaW50LmNvbS86dTovZy9wZXJzb25hbC9wZW5nc2lkYV96anVfZWR1X2NuL0VYSzJLMEItUXJOUGk4TVlMREZIZEI4QlFtOWNXVHhSR1Y5ZFFnYXVjemtWWVE%5FcnRpbWU9SHVZV1hEUVIyVWc&id=%2Fpersonal%2Fpengsida%5Fzju%5Fedu%5Fcn%2FDocuments%2Fpvnet%2FLINEMOD%2Etar%2Egz&parent=%2Fpersonal%2Fpengsida%5Fzju%5Fedu%5Fcn%2FDocuments%2Fpvnet&p=14'
target_dir = cfg.LINEMOD_DIR
temp_dir = cfg.TEMP_DIR


class DownloadPVNetLineMod(object):
    """
    PVNetの訓練に使用された LineMod データセットをダウンロードする関数

    """
    def __init__(self,
                              target_dir: str = cfg.LINEMOD_DIR,
                              temp_dir: str = cfg.TEMP_DIR):
        """
        Downloadオブジェクトを初期化する. ダウンロードした zip ファイルは一度 temp ディレクトリへ保存され, その後 target ディレクトリへ解凍される. 初期化では temp ディレクトリを新しく作成する. もし既に作成されていた場合は削除して新しく作成する.

        Args:
            target_dir (str, optional): ダウンロードした zip データの解凍先を絶対パスで指定する. Defaults to cfg.LINEMOD.
            temp_dir (str, optional): ダウンロードした zip ファイルの保存先を絶対パスで指定する. Defaults to cfg.TEMP_DIR.
        """

        self.linemod_objects = [
            'ape',
            'benchviseblue',
            'bowl',
            'cam',
            'can',
            'cat',
            'cup',
            'driller',
            'duck',
            'eggbox',
            'glue',
            'holepuncher',
            'iron',
            'lamp',
            'phone'
        ]

        self.target_dir = target_dir
        self.temp_dir = temp_dir


    def __del__(self):
        """作成した temp ディレクトリを削除する
        """
        shutil.rmtree(self.temp_dir)


    def main(self, object_name: str = 'all'):
        """zip ファイルを temp ディレクトリへダウンロードし, target ディレクトリに解凍する

        Arg:
            object_name(str, optional): ダウンロードするオブジェクトのカテゴリ名. Defaults to 'all'.
        """

        # ディレクトリを作成する
        MakeDir(self.target_dir, newly=True)
        MakeDir(self.temp_dir, newly=True)

        if object_name == 'all':
            object_names = self.linemod_objects
        elif object_name in self.linemod_objects:
            object_names = [object_name]
        else:
            raise ValueError('Invalid object name: {}'.format(object_name))

        for object_name in object_names:
            url = base_url.format(object_name)
            f_path = DownloadZip(url, dir_path= self.temp_dir, file_name = 'PVNet_linemod')

            unpach_path = os.path.join(self.target_dir, object_name)
            UnpackZip(f_path, unpach_path)


if __name__ == "__main__":
    downloader = DownloadPVNetLineMod()
    downloader.main()
    del downloader