import random

janken_hand = ["グーです。","チョキです。","パーです。"]

print("じゃんけんAIと「じゃんけん」をしよう！")
print("\n0:グー\n1:チョキ\n2:パー")
my_janken = int(input("じゃんけんの手を決めてください。数値を入力後「enter」キーを押してね。："))
print("じゃんけん、ぽん")

janken_ai = random.randint(0,2)

print("出した手：",janken_hand[my_janken])
print("じゃんけんAI",janken_hand[janken_ai])

hantei = (my_janken - janken_ai + 3) % 3
if hantei == 0:
    print("あいこです。")
elif hantei == 1:
    print("あなたの負けです。")
else:
    print("あなたの勝ちです！！")
