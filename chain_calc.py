import sys
import datetime
import hashlib
import re

fsalt = 'salt.txt'
fs = open(fsalt, 'r')
salt = fs.readline().rstrip('\n')
fs.close

def anon(anything):
    return hashlib.sha256(anything+salt).hexdigest()[-12:]

#this is working with cleartext ids
with open('cogs.txt') as f:
    chains = f.read().splitlines()

coc = {}
for i in xrange(len(chains)):
#for i in range(1, 100):
    #if i > 10:
    #    print chains[i]
    chain_list = []
    while len(chains[i]) > 0:
        chain_list.append(chains[i][-6:])
        chains[i] = chains[i][:-6]
    #if i > 10:
    #    print chain_list
    #print chain_list
    try:
        coc[chain_list[0]] = chain_list #some lines are blank?
    except:
        #print chain_list
        continue

print 'length of coc dictionary', len(coc)

fo = open('t_pairs_coc_dist_mar.txt', 'w+')
fa = open('t_pairs_coc_dist_anon_mar.txt', 'w+')

with open('t_pairs_clear.mar') as f:
    pairs = f.read().splitlines()

for i in xrange(len(pairs)):
#for i in range(0,100):
    peep1, peep2, count = pairs[i].split('|')
    chain1 = coc[peep1]
    chain2 = coc[peep2]
    try:
        match_peeps = [val for val in chain1 if val in chain2][0]
        distance = chain1.index(match_peeps) + chain2.index(match_peeps)
    except:
        #print chain1, ' ', chain2
        distance = len(chain1) + len(chain2)
    
    fo.write(peep1 + '|' + peep2 + '|' + str(distance) + '|' + str(count) + '\n')
    fa.write(anon(peep1) + '|' + anon(peep2) + '|' + str(distance) + '|' + str(count) + '\n')
    
fo.close()
fa.close()
