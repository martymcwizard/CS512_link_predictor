import sys
from itertools import combinations

F_dict = dict()
M_dict = dict()
counter = 0

fileIn1 = sys.argv[1] #feb_less_april_clear
fileIn2 = sys.argv[2] #mar_clear.txt 
fileOut1 = sys.argv[3] #t_pairs_clear.feb
fileOut2 = sys.argv[4] #t_uniq_pairs_clear.mar

fi = open(fileIn1, 'r')
num_lines = sum(1 for line in fi)
fi.close()
print 'Number of lines in training data', num_lines

foF = open(fileOut1, 'w+')
fi = open(fileIn1, 'r')
for line in range(0, num_lines): #num_lines):
    inline = [i for i in fi.readline().rstrip('\n').split('|')]
    if len(inline) > 3:
        aidline = inline[2:]
        aidline = sorted(set(aidline), reverse = False)
        while 'NoMATCH' in aidline:
            aidline.remove('NoMATCH')
        #print aidline
        for c in combinations(aidline,2):
            c = sorted(c, reverse = False)
            cstr = c[0] + c[1]
            if cstr in F_dict:
                F_dict[cstr] += 1
            else:
                F_dict[cstr] = 1

print 'Number of lines packed into a dictionary from first file', len(F_dict)
for key, value in F_dict.iteritems():
    foF.write(key[:6] + '|' + key[-6:] + '|' + str(value) + '\n')

foF.close()
fi.close()

fi = open(fileIn2, 'r')
num_lines = sum(1 for line in fi)
fi.close()
print 'Number of rows in test data', num_lines

foR = open(fileOut2, 'w+')
fi = open(fileIn2, 'r')
for line in range(0, num_lines): #num_lines):
    inline = [i for i in fi.readline().rstrip('\n').split('|')]
    if len(inline) > 3:
        aidline = inline[2:]
        aidline = sorted(set(aidline), reverse = False)
        while 'NoMATCH' in aidline:
            aidline.remove('NoMATCH')
        for c in combinations(aidline,2):
            #print c
            c = sorted(c, reverse = False)
            #print 'sorted', c
            cstr = c[0] + c[1]
            if cstr not in F_dict:
                if cstr in M_dict:
                    M_dict[cstr] += 1
                else:
                    M_dict[cstr] = 1


print 'Number of lines packed into a dictionary from first and second file', len(F_dict)
for key, value in M_dict.iteritems():
    foR.write(key[:6] + '|' + key[-6:] + '|' + str(value) + '\n')
print 'Number of new, unique pairs in test data: ', len(M_dict)

foR.close()
fi.close()

