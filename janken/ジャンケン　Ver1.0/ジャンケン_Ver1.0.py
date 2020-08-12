import random
from random import randint
janken_hand = ["グーです。","チョキです。","パーです。"]


def Opening():
    print("\n")
    print("*********************************************\n")
    print("   じゃんけん大会　Ver1.0　by k.Miki 2017　**\n")
    print("*********************************************\n")
    print("\n\n")
    print("じゃんけんAIと「じゃんけん」をしよう！")

def func_inport():
    print("\n0:グー\n1:チョキ\n2:パー")
    my_janken = int(input("じゃんけんの手を決めてください。数値を入力後「enter」キーを押してね。："))

def func_janken_ai():
    janken_ai = random.randint(0,2)

def func_jadge():
    
Opening = Opening *
func_inport()
func_janken_ai()