import os
import sys
sys.path.append(".")
sys.path.append("..")

from config.config import cfg
from src.utils import DownloadZip, UnpackZip

url = 'https://cloudstore.zih.tu-dresden.de/index.php/s/a65ec05fedd4890ae8ced82dfcf92ad8/download'
target_dir = 'temp'


def download_and_unzip():
    os.makedirs(target_dir, exist_ok=True)
    target_file = os.path.join(target_dir, 'Occlusion.zip')
    urllib.request.urlretrieve(url, target_file)
    os.system('unzip {} -d data'.format(target_file))
    os.system('mv data/OcclusionChallengeICCV2015 data/occlusion_linemod')
    # to lower case
    os.system(
        'for i in $( ls data/occlusion_linemod/models | grep [A-Z] ); do mv -i data/occlusion_linemod/models/"$i" data/occlusion_linemod/models/"`echo $i | tr \'A-Z\' \'a-z\'`"; done')
    os.system(
        'for i in $( ls data/occlusion_linemod/poses | grep [A-Z] ); do mv -i data/occlusion_linemod/poses/"$i" data/occlusion_linemod/poses/"`echo $i | tr \'A-Z\' \'a-z\'`"; done')


#download_and_unzip()


url = 'https://cloudstore.zih.tu-dresden.de/index.php/s/a65ec05fedd4890ae8ced82dfcf92ad8/download'
target_dir = cfg.LINEMOD_DIR
temp_dir = cfg.TEMP_DIR

def main():
    f_path = DownloadZip(url, "OcclusionChallengeICCV2015.zip", temp_dir)


if __name__ == "__main__":
    main()