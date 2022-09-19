#!/usr/bin/python3

counter = 0
while True:
    try:
        input()
        counter += 1
    except EOFError:
        break
print(counter)
