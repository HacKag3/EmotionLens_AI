def f(x: int):
    ''' restituisce x-esimo numero della sequenza di fibonacci '''
    if x<=0 : return 0
    if x==1 : return 1
    return f(x-1) + f(x-2)