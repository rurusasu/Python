# cording: utf-8

from common.csvIO import csvIO

io = csvIO()
v = io.open_twoD_array('./data/data.csv')
v = io.twoD_FroatToStr(v, digit=0.01)
learn_array = io.Get_AnytwoD_array(v, col_range_end=3)
test_array  = io.Get_AnytwoD_array(v,col_range_first=4, col_range_end=7)

io.csv_write('./data/learn.csv', learn_array)
io.csv_write('./data/test.csv', test_array)