#関数
#　def 関数名():　処理

#double関数
#　与えられた引数を2倍する

#引数を利用する関数を定義する場合
#　def 関数名(引数名):　処理

#isdigit()関数
#　文字列型の変数が数値かどうかを判定
#　文字列型の変数.isdigit()


#最も原始的な関数の例
def print_hello():
    print("こんにちは！ようこそPython3超入門講座へ！")

print_hello()

print("\n")

#引数
#引数numberの値を2倍にする関数
def double(number):
    print(number * 2)

#引数を指定して関数を呼び出し
double(199) #398

print("\n")

#2つの数を加減乗除する関数、num1,num2には数値を、
#modeには演算の種類を文字型で指定する。
#modeの演算の種類は、加算（plus）,減算(minus) ,乗算(multiply) ,除算(division)。
def calc(num1, num2, mode):
    if(mode == 'plus'):
        print(num1 + num2)
    elif(mode == 'minus'):
        print(num1 - num2)
    elif(mode == 'multiply'):
        print(num1 * num2)
    elif((mode == 'division') and (num2 != 0)):
        print(num1 / num2)
    else:
        print("ERROR:演算モードの指定が間違っています。又は0除算している可能性があります。")

#関数の呼び出しと、実行結果
calc(3, 4, 'plus')       #7
calc(1, 9, 'minus')      #-8
calc(12, 12, 'multiply') #144
calc(7, 11, 'division')  #0.636363634
calc(8, 7, 'iPhone5s')   #ERROR:演算モードの指定が間違っています。又は0除算している可能性があります。
calc(5, 0, 'division')   #ERROR:演算モードの指定が間違っています。又は0除算している可能性があります。

print("\n")

#戻り値
#ユーザの入力が戻り値として文字列型で変数strにセットされる
str = input('好きな数字を入力してください：')
print("あなたの好きな数字は" + str +"なんですね。素敵な数字ですね！")

print("\n")

#while文の条件式にTrueを指定して無限ループを意図的に作成、後でbreakで抜ける
while(True):
    #ユーザの入力が戻り値として文字列型でnum3にセットされる
    num3 = input('好きな数字を入力してください：')
    #ちゃんと文字列ではなく文字列型の数値が入力されているか判定
    if(num3.isdigit()):
        #num3の値を数値に変換
        num3 = int(num3)
        #num3を２乗して画面に表示
        print(num3 ** 2)
        #プログラムの目的は果たしたのでループを抜けて終了
        break
    else:
        print("ERROR:文字ではなく数字を入力してください")

print("\n")

#isdigit()関数の仕様
#書式「文字列型の変数.isdigit()」
#機能「文字型の変数が数値かどうかを判定」
#戻り値「TrueまたはFalse」
print('2'.isdigit())
print("45".isdigit())
print('iPhone'.isdigit())

print("\n")

#必ず整数型でユーザからの入力を受け取れる関数
#input_messageはユーザに入力を求めるときに表示するメッセージ
#error_messageはユーザが数値以外の値を入力した場合に表示するエラーメッセージ
def num_input(input_message,error_message):
    #変数numberにユーザからの入力を受け取る
    number = input(input_message)
    #ユーザからの入力が数値かどうか判定
    if(number.isdigit()):
        return int(number)
    else:
        return error_message

num = num_input('好きな数値を入力してください。','ERROR:数値を入力してください。')

#ユーザからの入力をそのまま表示
print(num)
#ユーザが入力した値のデータ型を表示
print(type(num))
