#while文の文法
#書式「while(条件式):　繰り返したい処理」
cnt = 0
while(cnt < 10):
    print("繰り返しております！")
    print(cnt+1)
    cnt = cnt+1

#while文を使ってユーザの好きな回数だけ繰り返すサンプルコード
end = 0
while(end == 0):
    print("ユーザから入力があるまでグルグル回ります")
    end = input("ループを終了するには0以外の好きな値を入力してください。0と入力するともう1周しますよ：")
    end = int(end) #文字列の戻り値を数値型に変換

print("\n")

#input()
#input()関数を使ってユーザから何らかの入力を受けた場合、戻り値は「文字列」になる
num1 = input("何か好きな数値を半角で入力してください。")
num2 = input("何か好きな数値を半角で入力してください。")
print(num1+num2) #文字列どうしの値を合体
print("\n")

num11 = input("何か好きな数値を半角で入力してください：")
num22 = input("何か好きな数値を半角で入力してください：")
num11 = int(num11) #文字列のnum11を数値型に変換
num22 = int(num22) #文字列のnum22を数値型に変換
print(num11+num22)
print("\n")

#breakを使ってwhileから抜け出す
while(True): #Trueは永遠に処理を繰り返す
    print("１回しか繰り返さない。")
    break

print("\n")

cnt2 = 0
while(True):
    print(cnt2)
    cnt2 = cnt2 + 1
    if((cnt2 == 3)):
        print("cnt2の値が3になったのでループを抜けます。")
        break

print("\n")

#continue文
cnt3 = 0
while(cnt3 < 3):
    cnt3 = cnt3 + 1
    if(cnt3 == 2):
        print("cnt3が２のときは最後のメッセージを飛ばしますよ！")
        continue
    print("繰り返しております！")

print("\n")

#for文を使ったループでのcontinueの使い方
list = ['Apple','Microsoft','IBM']
for company in list:
    if(company == 'IBM'):
        print(company + "は戦前から日本にある老舗ですね！")
        continue #continueすることでprint(company + "はIT企業の花形ですね！")の行をスキップ
    print(company + "はIT企業の花形ですね！")

print("\n")

#for文を辞書型に対して使った場合（復習）
dic = {'Apple':'iPhone','Microsoft':'Windows','Amazon':'book'}
for company2 in dic:
    print(company2 + "と言えば" + dic[company2])
