# LCS algorithm is used to compare how similar two sentences are.
def lcs(maked_sentence: str, origin_sentence: str) -> float:
    prev = [0] * len(maked_sentence)
    for i, r in enumerate(maked_sentence):
        current = []
        for j, c in enumerate(origin_sentence):
            if r == c:
                e = prev[j - 1] + 1 if i * j > 0 else 1
            else:
                e = max(prev[j] if i > 0 else 0, current[-1] if j > 0 else 0)
            current.append(e)
        prev = current

    return round(current[-1] / len(maked_sentence), 3)


# output order is (longer, shorter)
def compare_text_lengths(text1, text2):
    if len(text1) > len(text2):
        return set([text1, text2])
    elif len(text1) < len(text2):
        return set([text2, text1])
    else:
        return set([text1, text2])
