#タプル
#書式「変数 = (値1,値2,値3,...)」
tuple1 = (1, 2, 3, 4, 5)
print(tuple1) #(1, 2, 3, 4, 5)

tuple2 = ('うさぎ','カメ','鶴')
print(tuple2) #('うさぎ','カメ','鶴')

print("\n")

#複数の変数に値を代入
tpl = (1, 2, 3)
a, b, c = tpl
print(a) #1
print(b) #2
print(c) #3

print("\n")

#辞書
#辞書内での順番は決まらない。
number = {"one":1,"two":2,"three":3}
print(number["one"]) #1

fruits = {"apple":"りんご","orange":"みかん","banana":"バナナ"}
print(fruits["orange"]) #みかん

print("\n")

#dict()関数「リストを辞書に変換」
#元の値が2つの値の連続になっていることが必要。
kakaku_list = [["アジの切り身","400円"],["豚ひき肉","250円"],["キャベツ","150円"]]
kakaku_dict = dict(kakaku_list)
print(kakaku_dict) #{'アジの切り身':'400円','豚ひき肉':'250円',"キャベツ":'150円'}
print(kakaku_dict["キャベツ"]) #150円
print(kakaku_list[2][1]) #150円

print("\n")