#if文の中にif文を入れる場合(if文を「入れ子」にして使う場合)
gakusei_syou = True
test_score = 65
if(gakusei_syou == True):
    if(test_score > 40):
        print("学生証が確認できました。あなたは赤点ではありません。")

print("\n")

gakusei_syou2 = True
test_score2 = 85
if(gakusei_syou2 == True):
    if(test_score2 > 40):
        if(test_score2 >=80):
            print("学生証は確認できました。80点以上なのであなたの成績はAです。")

print("\n")

#複雑な条件式のif文
#書式（andを用いて複数の条件式を記述）
#「if((条件式1)and(条件式2)):　条件式1・条件式2の両方が満たされた場合（両方ともTrueの場合）実行される処理」
gakusei_syou3 = True
test_score3 = 80
if((gakusei_syou3 == True) and (test_score3 >= 75)):
    print("学生証を確認しました。テストの成績は75以上なのでAです！")

print("\n")

#書式（orを用いて複数の条件式を記述）
#「if((条件式1)or(条件式2)):　条件式1・条件式2のどちらか一方がTrueの場合に実行される処理」
test_score_english = 65
test_score_math = 10
if((test_score_english >= 60) or (test_score_math >= 60)):
    print("英語か数学のどちらか一方で60点以上とっているので合格です！")

print("\n")

#andとorを併用して作るより複雑な条件式
test_score_english2 = 65
test_score_math2 = 10
test_score_japanese = 80

if((test_score_japanese >= 70) and ((test_score_english2 >= 60) or (test_score_math2 >=60))):
    print("合格です！")

print("\n")

