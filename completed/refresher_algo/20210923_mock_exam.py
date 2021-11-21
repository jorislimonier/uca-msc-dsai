# Mock exam
# Question 1
", ".join([str(n) for n in range(2000, 3201) if n%7==0 and n%5!=0])

# Question 2
def fact(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n*fact(n-1)
fact(8)

# Question 3
def alpha_sorter(s):
    s = s.split(",")
    s.sort()
    return ",".join(s)
s = "without,hello,bag,world"
alpha_sorter(s)

# Question 4
def squarer(n):
    print({i:i**2 for i in range(n+1)})
squarer(8)

# Question 5
s = "New to Python or choosing between Python 2 and Python 3? Read Python 2 or Python 3."
s = s.split(" ")
count_dict = {k: s.count(k) for k in s}
keys = list(count_dict.keys())
keys.sort()
for k in keys:
    print(f"{k}: {count_dict[k]}")

# Question 6
import numpy as np
def find_closest(v, scalar):
    abs_diff = np.absolute(v-scalar)
    index = np.argmin(abs_diff)
    return v[index]

# Question 7
def find_longest_word(low):
    return max(map(len, low))
low = ["a", "list", "of", "words"]
find_longest_word(low)

# Question 8
import sys
print(sys.version)

# Additional exercises
VOWELS = ["a","e","i","o","u"]
# Question 1
def truth_values(word):
    return [l in VOWELS for l in word]

# Question 2
def find_first_vowel(word):
    for letter in word:
        if letter in VOWELS:
            return letter

# Question 3
def initials(full_name):
    splitted_name = full_name.split(" ")
    if splitted_name[1] == "NA":
        del splitted_name[1]
    for name in splitted_name:
        print(name[0])

# Question 4
def is_float_int(s):
    try:
        s = int(s)
        return True
    except Exception as e:
        try:
            s = float(s)
            return True
        except Exception as e:
            return False
is_float_int("3")

# Question 5
def join_dicts(d1, d2):
    joined_dict = d1.copy()
    for k, v in d2.items():
        while k in joined_dict.keys():
            k += 1
        joined_dict[k] = v
    return joined_dict

d1 = {1: "a", 2: "b", 3: "c"}
d2 = {1: "e", 2: "f"}
join_dicts(d1, d2)

# Question 6
s1 = "a string"
s2 = "another string"
s3 = "a third string"
los = [s1, s2, s3]
print("\n".join(los))

# Question 7
# Question 8
# Question 9
# Question 10
# Question 11
# Question 12