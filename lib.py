from time import time

def pf(name, value):
    f = open(name + '.txt', 'w')
    f.write(str(value))
    f.close()
    print(name + '="%s"\n' % value)

def pp(l): return ' '.join(map(str, l))

def finish(start):
    print("--- %s seconds ---" % str(time() - start))

def lib_test():
    print('lib_test!')