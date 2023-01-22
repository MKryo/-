import numpy as np

def trans_to_categorical(x):
  x = np.array(x)
  x = x*100
  x = np.floor(x).astype(int)
  return x

def mutual_info(X, Y):

  def Pxy(x, y):
    ans = np.count_nonzero((X == x) & (Y == y)) / len(X)
    if ans==0:
      ans = 1e-15
    return ans

  def Px(x):
      ans = np.count_nonzero(X == x) / len(X)
      if ans==0:
        ans=1e-15
      return ans

  def Py(y):
      ans = np.count_nonzero(Y == y) / len(Y)
      if ans ==0:
        ans=1e-15
      return ans

  mutual_info=0
  for i in range(1,101):
    for j in range(1,101):
      mutual_info += Pxy(x=i, y=j) * np.log( Pxy(x=i, y=j) / (Px(x=i)*Py(y=j)) )

  return mutual_info