import os
import shutil


def MakeDir(dir_path: str, newly: bool = False) -> str:
    """dir_path で指定された位置にディレクトリを作成する関数．既に同名のディレクトリが存在する場合は，現在の「年, 月, 日_時, 分, 秒」を作成したい付ディレクトリ名の後ろに連結する．

    Args:
        dir_path(str): 作成するディレクトリまでのパス
        newly (bool): 新規ファイルを作成するか. Defaults to False.
    Return:
        dir_path (str): 作成したディレクトリの絶対パス
    """

    if os.path.exists(dir_path): # 存在する場合
        if newly:  # 新規ファイルを作成する
            __DeleteDir(dir_path)  # ディレクトリ内のフォルダごと削除
        else:
            now = datetime.datetime.now()
            dir_path = dir_path + '_' + now.strftime('%Y%m%d_%H%M%S') # 現在時刻をファイル名に付ける
    os.mkdir(dir_path)
    return dir_path


def  __DeleteDir(dir_path: str):
    """ディレクトリ内のフォルダごと削除

    Args:
        dir_path (str): 削除するディレクトリへの絶対パス
    """
    shutil.rmtree(dir_path)



if __name__ == "__main__":
    import sys
    sys.path.append(".")
    sys.path.append("..")

    from config.config import cfg

    path = MakeDir(cfg.TEMP_DIR, newly=True)
    __DeleteDir(path)
