#for文により繰り返し処理
#書式「for 変数 in range(繰り返したい数):　繰り返したい処理」
for i in range(3):
    print("3回繰り返します")
    print(i)

print("\n")

#書式「for 変数 in データ:　変数を用いた処理」
str = "python3"
for c in str:
    print(c)

print("\n")

#リスト型とfor文
fruits = ['apple','orange','banana']
for e in fruits:
    print(e + 'が食べたい！')

print("\n")

#辞書型とfor文
dic = {'りんご':'apple','みかん':'orange','ばなな':'banana'}
for fruits in dic:
    print(fruits + 'を英訳すると' + dic[fruits])

