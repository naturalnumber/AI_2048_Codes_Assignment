ideals = [[2]]
last_ = ideals[0]

for i in range(1024+9):
    next_ = []

    prev_ = None
    for n in last_:
        if prev_ is None:
            prev_ = n
        elif prev_ == n:
            next_.append(n*2)
            prev_ = None
        else:
            next_.append(prev_)
            prev_  = n
    if prev_ is not None:
        next_.append(prev_)
    next_.append(2)

    ideals.append(next_)
    last_ = next_

for i in ideals:
    print(i)