def edit_distance(string1, string2):
    if len(string2) > len(string1):
        string1, string2 = string2, string1
    if not string2:
        return len(string1)
    before = [0] * len(string2)
    now = [0] * len(string2)
    for i, char1 in enumerate(string1):
        for j, char2 in enumerate(string2):
            if not j:
                now[j] = before[0] + int(char1 != char2)
            else:
                now[j] = min(now[j - 1] + 1, before[j] + 1, before[j - 1] + int(char1 != char2))
        before = now
        now = [0] * len(string2)
    return before[-1]
