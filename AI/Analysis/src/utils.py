import os

import datetime
import requests
import zipfile

class URLAccessError(Exception):
    """指定した URL でアクセスに失敗したことを知らせる例外クラス"""
    pass

class IMAGENotFound(Exception):
    """HTML 内に画像情報が含まれていなかったことを知らせる例外クラス"""
    pass


def _DownloadFile(url: str):
    """URL を指定してダウンロードする
    参考: https://www.lifewithpython.com/2015/10/python-how-to-download-extact-zip-file.html

    Args:
        url (str): ダウンロードしたい画像が掲載されているサイトの URL
        timeout (int, optional): タイムアウト. Defaults to 10[s].
    """
    res = requests.get(url, stream=True)
    if res.status_code != 200: # HTTPステータスコードの200番台以外はエラーコード
        raise URLAccessError("HTTP status: " + res.status_code)

    return res

def _ProgressPrint(block_count: int, block_size: int, total_size: int):
    """作業の進捗を表示するための関数
        イメージ：[=====>    ] 50.54% ( 1050KB )
        参考: https://qiita.com/jesus_isao/items/ffa63778e7d3952537db

    Args:
        block_count (int): 1回の処理で処理されるファイルサイズ
        block_size (int): チャンクサイズ
        total_size (int): トータルのファイルサイズ
    """
    percentage = 100 * block_count * block_size / total_size
    # 100 より大きいと見た目が悪いので...
    if percentage > 100:
        percentage = 100
    # バーは max_bar 個で 100% とする
    max_bar = 10
    bar_num = int(percentage / (100 / max_bar))
    progress_element = '=' * bar_num
    if bar_num != max_bar:
        progress_element += '>'
    bar_fill = ' ' # これで空のところを埋める
    bar = progress_element.ljust(max_bar, bar_fill)
    total_size_kb = total_size / 1024
    print(
        f'[{bar}] {percentage: .2f}% ( {total_size_kb: .0f}KB )\r', end=''
    )

    # メモリリーク防止のため，使用した変数は削除
    del percentage, max_bar, bar_num, progress_element, bar_fill, bar , total_size_kb


def DownloadImage(url: str) -> bytes:
    """URL 先の HP で動作している HTML テキスト内に含まれる画像情報を検出し，ダウンロードするための関数

    参考: https://qiita.com/donksite/items/21852b2baa94c94ffcbe

    Arg:
        url (str): ダウンロードしたい画像が掲載されているサイトの URL

    Return:
        response.content(str): 画像情報
    """
    res = _DownloadFile(url)
    content_type = res.headers["Content-Type"]
    if "image" not in content_type: # HTML 内の画像情報を検索
        raise IMAGENotFound("Image not found: " + url)

    # 読み込んだURLを画面上に表示
    print(url)
    return res.content


def DownloadZip(url: str, f_name: str = None, dir_name: str = None):
    """URL 先から ZIP ファイルをダウンロードするための関数

    Arg:
        url(str): ダウンロードしたい zip ファイルが存在するサイトの URL
        f_name(str): zip データを書き込むファイル名
        dir_name(str): zip ファイルをダウンロードするディレクトリパス Defaults to os.path.basename('temp').

    Return:
        f_path(str): ダウンロードしたzipファイルまでのパス
    """
    chunk_size = 1024
    if f_name is None:
        name = url.split('/')[-1]
    elif (f_name is not None) and (type(f_name) is str):
        name = f_name
    else:
        raise Exception("f_name には文字列を入力してください．")
    res = _DownloadFile(url)
    total = int(res.headers.get('content-length'))

    # ダウンロード時に使用するディレクトリを作成
    if dir_name is None:
        dir_name = os.path.basename('temp')
        dir_name = os.path.join(dir_name, name)
        if os.path.exists(dir_name):
            now = datetime.datetime.now()
            dir_name = dir_name + '_' + now.strftime('%Y%m%d_%H%M%S')
        os.makedirs(dir_name)
    elif not os.path.exists(dir_name):
        os.makedirs(dir_name)
    # データを書き込むファイルを作成
    f_path = os.path.join(dir_name, name)

    print("\n File Downloading: {}".format(name))
    try:
        with open(f_path, 'wb') as f:
            n = 0
            for chunk in res.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    f.flush()
                n += 1
                _ProgressPrint(block_count=n, block_size=chunk_size, total_size=total)
    except Exception as e:
        print("ZIP download file writing Error！: ", e)
        return False
    finally:
        del chunk_size, dir_name
        return f_path


def makedir(src: str, dir_name: str) -> str:
    """dst で指定された位置に dir_name で指定された名前のディレクトリを作成する関数．既に同名のディレクトリが存在する場合は，現在の「年, 月, 日_時, 分, 秒」を作成したい付ディレクトリ名の後ろに連結する．

    Args:
        src(str): ディレクトリを作成する場所の親ディレクトリまでのパス
        dir_name(str): 作成するディレクトリ名
    Return:
        dst(str): 作成したディレクトリの絶対パス
    """
    dst = os.path.join(src, dir_name)
    if os.path.exists(dst): # 存在する場合
        now = datetime.datetime.now()
        dst = dst + '_' + now.strftime('%Y%m%d_%H%M%S')
    os.mkdir(dst)

    return dst


# 画像の保存
def _SaveImage(filename: str, image):
    """画像を保存するための関数

    Args:
        filename (str): 画像データを書き込む拡張子付きファイルまでパス
        image ([type]): 画像情報
    """
    with open(filename, "wb") as f:
        f.write(image)


def UnpackZip(zip_path: str, unpack_path: str, create_dir: bool=True):
    """originalディレクトリ内のzipファイルを展開するための関数
    Arg:
        zip_path(str): zip ファイルのパス,
        unpack_path(str): zip ファイルの解凍先のパス
        create_dir(bool optional): 解凍するときに、解凍前のファイル名と同じ名前のディレクトリを作成する。default to True.
    """
    if create_dir:
        try:
            os.makedirs(unpack_path, exist_ok=True) # create directry
        except FileExistsError as e:
            print("ディレクトリ作成時にエラーが発生しました: ", e)
            return False

    with zipfile.ZipFile(zip_path) as existing_zip:
        existing_zip.extractall(unpack_path)


if __name__ == "__main__":
    # _DowinloadFile Test
    #url = "http://nagata.rs.socu.ac.jp/"
    #r = _DownloadFile(url)
    #print(r)

    # DownloadImage Test
    #url = "https://raw.githubusercontent.com/rurusasu/Diary/master/%E7%94%BB%E5%83%8F/AI%E3%81%AE%E9%96%A2%E9%80%A3%E7%A0%94%E7%A9%B6%E5%88%86%E9%87%8E.png"
    #img = DownloadImage(url) # b' \x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\...
    #print(type(img))

    # DownloadZip Test
    #url = 'https://github.com/rurusasu/Python/blob/master/AI/Analysis/data/2020-10-21_%E5%8B%95%E4%BD%9C%E5%BE%8C.zip'
    #zip = DownloadZip(url)

    object_names = ['ape', 'benchviseblue', 'bowl', 'cam', 'can', 'cat',
                'cup', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher',
                'iron', 'lamp', 'phone']

    # Stefan Hinterstoißer さんのサイトのURL
    base_url = 'http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/{}.zip'

    for object_name in object_names:
        target_file = base_url.format(object_name)
        DownloadZip(target_file)