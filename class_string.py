import os

string = ''
for fn in os.listdir('dataset'):
    string += fn
print(string, len(string))
# only works on Windows
# 2345678ABCDEFGHKLMNPQRSTUVWXY 29
