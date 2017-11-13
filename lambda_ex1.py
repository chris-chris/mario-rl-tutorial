def t(a,b,c):
  print(a,b,c)

def t2(n):
  return lambda *args, **kwargs: t(a=n, *args, **kwargs)

lambda_fun = t2(1)
lambda_fun(b=2, c=3)
# 1 2 3