with open('all_cogs_anon_20160315.txt') as f:
    chains = f.read().splitlines()

coc = {}
for i in xrange(len(chains)):
    pass
    chain = chains[i].split('|')
    coc[chain[0]] = chain 
    #if i < 10:
    #    print chain[0], chain

fo = open('pairs_with_distance.txt', 'w+')

with open('t_pairs.feb') as f:
    pairs = f.read().splitlines()

for i in xrange(len(pairs)):
    peep1, peep2 = pairs[i].split()
    chain1 = coc[peep1]
    chain2 = coc[peep2]
    match_peeps = [val for val in chain1 if cal in chain2][0]
    if match_peeps == []:
        distance = len(chain1) + len(chain2)
    else:
        distance = chain1.index(match_people) + chain2.index(match_people)
    fo.write(peep1 + '|' + peep2 + '|' + distance)
    
fo.close()
