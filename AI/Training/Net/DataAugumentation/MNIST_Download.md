# MNISTをダウンロードし，簡単な前処理を行う
手書き数字の認識データセットとして有名なMNISTのダウンロードと簡単な前処理の方法を残しておく．

ダウンロードしたデータは，pickle形式で保存して，すぐにロードできるようにしておく．\
しかし，現在保存している訓練データがすべて画像データ（拡張子が.pngや.jpg）のため，こちらの形式での保存方法も残しておく予定である．

# Pickleとは
Pickleとは，プログラム実行中のオブジェクトをファイルとして保存するものである．そのため，この形式で保存した後に再びロードすると，保存した時点でのオブジェクト状態を再現することができる．

ちなみに，「Pickle」とは漬物の意味で，データを漬物のように長期保存に適した形で保存するということを表している...のかな?

# データをダウンロードする
データをHPからダウンロードする．MNISTのアドレスは\
http://yann.lecun.com/exdb/mnist/\
である．ここからファイルをダウンロードするわけだが，ここからPythonを使って行う．

`import urllib.request`

`url_base = 'http://yann.lecun.com/exdb/mnist/'`\
`key_file = {`\
`'train_img':'train-images-idx3-ubyte.gz',`\
`'train_label':'train-labels-idx1-ubyte.gz',`\
`'test_img':'t10k-images-idx3-ubyte.gz',`\
`'test_label':'t10k-labels-idx1-ubyte.gz'`\
`}`

方法としては，`urllib`パッケージを使ってHPからデータをとってきて保存する．上のページにアクセスするとわかるが，4種類のフォルダがダウンロードできるようになっている．それぞれを`'train_img', 'train_label', 'test_img', 'test_label'`というキーを付けて区別する．

実際にダウンロードする場合，\
`dataset_dir = 'C:/Users/usr/Documents' #データを保存する場所`

`for v in key_file.values():`\
`fike_path = dataset_dir + '/' + v`\
`urllib.request.urlretrieve(url_base + v, file_path)`

少し待てばダウンロードが終了する．この時，フォルダは\
`Documents`\
`|_t10k-images-idx3-ubyte.gz`\
`|_t10k-labels-idx1-ubyte.gz`\
`|_train-images-idx3-ubyte.gz`\
`|_train-labels-ubyte.gz`

のようになっているはず．

# Pickle形式で保存する
必要なフォルダをダウンロードしたが，これは見ればわかる通り，圧縮ファイルである．これを扱うには，`gzip`ライブラリを使う．使い方は，ふつうのファイルを開くときと非常によく似ていて\
`gzip.open('ファイル名', '読み込み形式(r, wなど)')`

では，ここからPickle形式での保存を行っていく．\
`import gzip`\
`import numpy as np`\

`file_path = dataset_dir + key_file['train_img'] # 試しにtrain_imgを見てみる`\
`with gzip.open(file_path, 'rb') as f:`\
`data = np.frombuffer(f.read(),np.uint8)`\

`len(data) #->47040016`

ここで`np.frombuffer()`というのは，バッファーからnumpyの配列を生成するメソッドである．\
最終行では，`data`の長さを調べている．MNISTのページによると、数字の画像データは28x28のサイズをしているので，データ数は28x28の倍数であるはず．\
`len(data) % (28**2) #-> 16`

しかし，ダウンロードしたデータには16個ほど余分なデータが含まれていることがわかる．\
これをカットするためには\
`with gzip.open(file_path, ''rb) as f:`\
`data = np.frombuffer(f.read(), np.uint8, offset=16)`

とする同様にlabelの方も調べると，`offset=8`とすれば良いことがわかる．

最後にPickle形式で保存する．これにはPickleライブラリを使用する．このライブラリはいろいろなことができるが，今回は次のメソッドを使用する．

| メソッド | 動作 |
| ---- | ---- |
| dump |  オブジェクトをpickle化する  |
| load |  pickleオブジェクト表現を再構成 | 

## データを保存する場合
`import pickle`

`sabe_file = dataset_dir + '/mnist.pkl'` # 拡張子は.pkl\
`with open(save_file, 'wb') as f:`\
`pickle.dump.load(f)` #-1は最も高いプロトコルバージョンで保存することを指定している

保存したデータを読み込むときには，\
`with open(sabe_data, 'rb') as f:`\
`dataset = pickle.load(f)`

とする．