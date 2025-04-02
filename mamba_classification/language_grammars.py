import re
import itertools
import random

def tomita_1(word):
    # tomita 1: string containing no 0s
    return not "0" in word

def tomita_2(word):
    # tomita 2: string containing 10s only
    return word=="10"*(int(len(word)/2))

_not_tomita_3 = re.compile("((0|1)*0)*1(11)*(0(0|1)*1)*0(00)*(1(0|1)*)*$") 
# *not* tomita 3: words containing an odd series of consecutive ones and then later an odd series of consecutive zeros
def tomita_3(w):
    # tomita 3: opposite of above 
    return None is _not_tomita_3.match(w) #complement of _not_tomita_3

def tomita_4(word):
    # tomita 4: string containing no 000s
    return not "000" in word

def tomita_5(word):
    # tomita 5: string containing an even number of 0s and an even number of 1s
    return (word.count("0")%2 == 0) and (word.count("1")%2 == 0)

def tomita_6(word):
    # tomita 6: string such that the number of 0s minus the number of 1s is divisible by 3
    return ((word.count("0")-word.count("1"))%3) == 0

def tomita_7(word):
    # tomita 7: 0*1*0*1*0*
    return word.count("10") <= 1

def mod_n(n):
    def moder(word):
        if word == '':
            return 1
        return int(word, 2) % n == 0
    return moder

def make_train_set_for_target(target,alphabet,lengths=None,max_train_samples_per_length=300,search_size_per_length=1000,provided_examples=None):
    train_set = {}
    if None is provided_examples:
        provided_examples = []
    if None is lengths:
        lengths = list(range(15))+[15,20,25,30] 
    for l in lengths:
        samples = [w for w in provided_examples if len(w)==l]
        samples += n_words_of_length(search_size_per_length,l,alphabet)
        pos = [w for w in samples if target(w)]
        neg = [w for w in samples if not target(w)]
        pos = pos[:int(max_train_samples_per_length/2)]
        neg = neg[:int(max_train_samples_per_length/2)]
        minority = min(len(pos),len(neg))
        pos = pos[:minority+20]
        neg = neg[:minority+20]
        train_set.update({w:True for w in pos})
        train_set.update({w:False for w in neg})

    print("made train set of size:",len(train_set),", of which positive examples:",
        len([w for w in train_set if train_set[w]==True]))
    return train_set

def all_words_of_length(length,alphabet):
    return [''.join(list(b)) for b in itertools.product(alphabet, repeat=length)]

def n_words_of_length(n,length,alphabet):
    if 50*n >= pow(len(alphabet),length):
        res = all_words_of_length(length, alphabet)
        random.shuffle(res)
        return res[:n]
    #else if 50*n < total words to be found, i.e. looking for 1/50th of the words or less
    res = set()
    while len(res)<n:
        word = ""
        for _ in range(length):
            word += random.choice(alphabet)
        res.add(word)
    return list(res)