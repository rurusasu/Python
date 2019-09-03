#リストのサンプルコード
pc=["ノートパソコン","デスクトップパソコン","タブレット"]
print(pc)

print("\n")

#list()関数
empty_list=list() #空っぽのlistを作成
print(empty_list)

print("\n")

#文字列をリスト化
py=list("Python3")
print(py)

print("\n")

#split()関数の応用
s="人間万事塞翁が馬とは、どんな嫌な出来事、どんな幸いの、原因になるのか、分からないという意味である。"
print(s.split("、"))

print("\n")

#リストからの要素の取り出し
fruit_list = ["りんご","ばなな","すいか"]
print(fruit_list[0])
print(fruit_list[1])
print(fruit_list[2])

print("\n")

#listの要素としてlistを用いる
list_of_list = [["ステーキ","生姜焼き"],["ニンジン","キャベツ"]]
print(list_of_list[0])    #["ステーキ","生姜焼き"]
print(list_of_list[0][0]) #"ステーキ"

print("\n")

#リストの要素の変更・追加・削除
list = ["車","タイヤ","エンジン"]
list[0] = "自動車"
print(list) #["自動車","タイヤ","エンジン"]

print("\n")

#文字列のスライス
s = "あいうえお"
print(s[0]) #'あ'

print("\n")

#スタートからエンド-1文字目なので0文字目から0文字目を指定している。
print(s[0:1]) #'あ'
print(s[0:2]) #'あい'

print("\n")

#ステップが2なので1文字飛ばし
print(s[0:5:2])

print("\n")

#スライスとリスト
list = ['ケーキ','チョコ','クッキー','アメ','クラッカー']
print(list[0:2]) #['ケーキ','チョコ']
print(list[1:3]) #['チョコ','クッキー']
print(list[0:5:2]) #['ケーキ','クッキー','クラッカー']

print("\n")

#要素の順序を反転させる技
print(list[::-1]) #['クラッカー','アメ','クッキー','チョコ','ケーキ']

print("\n")

#append()関数「リストの末尾に新たな値を追記」
#書式は「list変数.append(追加したい要素)」

list = ['味噌汁','おしんこ','納豆']
list.append('白米')

print(list) #['味噌汁','おしんこ','納豆','白米']

print("\n")

#extend()関数「2つのリストを合体」
#書式「list変数.extend(合体させたいlist変数)」
list1 = ['鉛筆','シャーペン','ボールペン']
list2 = ['消しゴム','サインペン']
list1.extend(list2)
print(list1) #['鉛筆','シャーペン','ボールペン','消しゴム','サインペン']

print("\n")

#リストの要素を1個だけ削除したい
#書式「del list[削除したい要素の番号]」

list = ['数学','国語','英語']

#リストの最初の要素は「0」なことに注意！
del list[1] #2番目の要素を削除
print(list) #['数学','英語']
