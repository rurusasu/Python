import time

NUM = 10000000

list_ = [str(x) for x in range(NUM)]

start_time = time.time()
list1 = [x+'\n' for x in list_]
with open("text_writelines.txt", 'wt') as f:
    f.writelines(list1)

elapsed_time = time.time() - start_time
print("elapsed time:", elapsed_time, "[sec]")
