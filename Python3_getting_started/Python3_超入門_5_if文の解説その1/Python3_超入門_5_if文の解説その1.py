#条件分岐その1
#書式「if(条件式): 条件がTrue(真)のとき実行される処理」
if(10>=1):
    print("10は1より大きい数") #printの前にTABが必要！！

print("\n")

#条件分岐その2
#書式「if(条件式): 条件式がTrueのとき処理　else: 条件式がFalseのとき処理」
budget = 500
orange_price = 600

if(budget >= orange_price):
    print("みかんを購入しました。")
else:
    print("お金が足りなくてみかんが買えませんでした。")

print("\n")

#条件分岐その3
#書式「if(条件式): 条件式がTrueの場合の処理　elif(条件式2): 条件式2がTrueの場合の処理　else: 全ての条件式がFalseの場合の処理」
test_score = 65
if(test_score == 100):
    print("満足です！素晴らしい！")
elif(test_score >= 60):
    print("合格点です。")
else:
    print("60点未満なので不合格です。")
