from pathlib import Path
import pprint


p = Path(r'E:\Data') # 内部にディレクトリを持つパス
#p = Path(r'E:\Data\H31_Miki\CAD') # 内部にディレクトリを持たないパス
glob = p.glob('*')
print(p)
print(glob)

# ディレクトリ名だけ取得
#pprint.pprint([x for x in p.iterdir() if x.is_dir()])
#l = [x for x in p.iterdir() if x.is_dir()] # ディレクトリへのパスを取得する
l = [x.name for x in p.iterdir() if x.is_dir()] # ディレクトリ名のみを取得する

pprint.pprint(l)
