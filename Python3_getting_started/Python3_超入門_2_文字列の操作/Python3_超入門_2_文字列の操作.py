#改行
s = "こんにちは！\n元気ですか？\n"
print(s)

#"+"を用いることで文字列を連結できる。
s1 = "abc"
s2 = "def \n"  #"\n"は改行
print(s1 + s2)

#"*"演算子を利用することで文字列を繰り返すことができる。
s3 = "abc"
print(s3 * 5)

print("\n")

#文字列の中から1つの文字列を取り出す
s4 = "abcde"
print(s4[0])  #a
print(s4[-1])  #e
print(s4[2])  #c

print("\n")

#文字列をスライスする。
test = "abcdefghijklmn"
print(test[:])      #全文の選択
print(test[2:])     #開始位置の選択（今は２）
print(test[2:5])    #開始位置と終了位置の選択
print(test[0:5:2])  #ステップを用いてスライス（ステップは２）
print(test[-1::-1]) #全文の順番を入れ替える

print("\n")

#文字列の長さを取得する
s = "大盛担々麺"
print(len(s))

print("\n")

#文字列の分割
s = "ようこそ！夢の国！Python3の世界へ!一緒にPython3を楽しみましょう。"
print(s.split("！")) #"!"マークが全角な点に注意
                     #特定の記号などにより文章を区切る
                     #セパレータに何も指定しないと「改行、スペース、タブ」をセパレータにする。

print("\n")

#resplace()関数による文字列の操作
#書式「文字列変数(例s).resplace(置き換えたい文字列,置換後の文字列)」
s="Python3はクールだね！"
print(s.replace("クール","エキサイティング"))

print("\n")
