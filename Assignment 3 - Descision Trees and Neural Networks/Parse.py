import re

train_file = 'Data/covType/train.dat'
valid_file = 'Data/covType/valid.dat'
test_file = 'Data/covType/test.dat'


def parse(filename):
    with open(filename, 'r+') as f:
        temp = f.read()
        temp = re.sub(',', ' ', temp)
        temp = re.sub('\w+_\w+:', '', temp)
        temp = re.sub('\w+:', '', temp)
        temp = re.sub('Discrete', '0', temp)
        temp = re.sub('Continuous', '1', temp)
        f.seek(0)
        f.write(temp)
        f.truncate()


parse(train_file)
parse(valid_file)
parse(test_file)
