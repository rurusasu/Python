#関数の書式
#　def 関数名([引数1],[引数2],...,[引数n]):
#　　　処理
#　　return 戻り値


# サンプルコード1「引数も戻り値もとらない例」
def func():
    print("Python3最高！")

func()


# サンプルコード2「引数はとるが戻り値は返さない例」
def func(message):
    print(message)
    
func("PHPよりPython3の方が楽しいかも！？")


# サンプルコード3「引数もとるし、戻り値も返す例」
def func(message):
    message = message + 'は楽しい！'
    return message

print(func('Python3'))


# ちょっぴり応用的な関数の定義方法
def fruits_price(apple_price, lemon_price, orange_price):
    fruits_price_dictionary = {'apple':apple_price, 'lemon':lemon_price, 'orange':orange_price}
    return fruits_price_dictionary

#キーワード引数を用いて用いて関数を呼び出し
fruits_price_dictionary = fruits_price(300, 200, 150)

print(fruits_price_dictionary) #{'lemon':200, 'orange':150, 'spple':300}


# 辞書型を返す関数
def fruits_price2(apple_price, lemon_price, orange_price):
    fruits_price_dictionary2 = {'apple':apple_price, 'lemon':lemon_price, 'orange':orange_price}
    return fruits_price_dictionary2

# キーワード引数を用いて関数を呼び出し
fruits_price_dictionary2 = fruits_price2(orange_price = 150, lemon_price = 200, apple_price = 300)

print(fruits_price_dictionary2) # {'lemon': 200, 'orange': 150, 'apple': 300}


# デフォルト引数
def print_help_message(message1 = '困ったらPython3最高！と叫ぶ！', message2 = 'もっと困ったら寝る！！！'):
    print(message1)
    print(message2)

print_help_message()


# 立方体の体積を求める関数、デフォルト引数値を指定
def cube(base = 10, height = 10, depth = 10):
    return base * height * depth

print(cube(10, 15, 7)) #1050
print(cube()) #1000