def isin(o, it):
    """Function to test if an object is in an iterable."""
    res = False
    for i in it:
        if o is i:
            res = True
            break
    return res
