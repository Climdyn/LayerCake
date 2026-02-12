
"""

    Variables utility module
    ========================

    Defines useful functions.

"""


def combine_units(units1, units2, operation):
    """Combine units strings together with a given operation on the exponents.

    Parameters
    ----------
    units1: str
       First units string to combine.
    units2: str
       Second units string to combine.
    operation: str
       Operation to perform on the units exponents. Presently can be `'-'` or `'+'`.

    Returns
    -------
    str
        The resulting units string.

    """

    if not units1 and not units2:
        return ''
    elif not units2:
        return units1
    elif not units1:
        if operation == '+':
            return units2
        else:
            return power_units(units2, -1)

    ul = units1.split('][')
    ul[0] = ul[0][1:]
    ul[-1] = ul[-1][:-1]
    ol = units2.split('][')
    ol[0] = ol[0][1:]
    ol[-1] = ol[-1][:-1]

    usl = list()
    for us in ul:
        up = us.split('^')
        if len(up) == 1:
            up.append("1")

        if up[0]:
            usl.append(tuple(up))

    osl = list()
    for os in ol:
        op = os.split('^')
        if len(op) == 1:
            op.append("1")

        if op[0]:
            osl.append(tuple(op))

    units_elements = list()
    for us in usl:
        new_us = [us[0]]
        i = 0
        for os in osl:
            if os[0] == us[0]:
                if operation == '-':
                    power = int(os[1]) - int(us[1])
                else:
                    power = int(os[1]) + int(us[1])
                del osl[i]
                break
            i += 1
        else:
            power = int(us[1])

        if power != 0:
            new_us.append(str(power))
            units_elements.append(new_us)

    if len(osl) != 0:
        units_elements += osl

    units = list()
    for us in units_elements:
        if us is not None:
            if int(us[1]) != 1:
                units.append("[" + us[0] + "^" + us[1] + "]")
            else:
                units.append("[" + us[0] + "]")
    return "".join(units)


def power_units(units, power):
    """Apply power to a units strings.

    Parameters
    ----------
    units: str
       Units to take power of.
    power: int
       Power.

    Returns
    -------
    str
        The resulting units string.

    """

    if units:
        ul = units.split('][')
        ul[0] = ul[0][1:]
        ul[-1] = ul[-1][:-1]

        usl = list()
        for us in ul:
            up = us.split('^')
            if len(up) == 1:
                up.append("1")

            usl.append(tuple(up))

        units_elements = list()
        for us in usl:
            units_elements.append(list((us[0], str(int(us[1]) * power))))

        out_units = list()
        for us in units_elements:
            if us is not None:
                if int(us[1]) != 1:
                    out_units.append("[" + us[0] + "^" + us[1] + "]")
                else:
                    out_units.append("[" + us[0] + "]")
        return "".join(out_units)
    else:
        return ""
