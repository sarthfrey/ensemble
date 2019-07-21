from ensemble import child, Ensemble


def h():
  pass

@child('e1', 'e2')
def f(x, y=3, z=4):
  return x + y + z

@child('e1')
def g(y):
  return y**3

@child('e1', 'e3')
def i(x):
  return x

@child('e3')
def i2(x):
  return x**2

@child('e3')
def i3(x, y):
  return x**3 + y


if __name__ == '__main__':

  # create our first ensemble and give it a name
  e1 = Ensemble('e1', mode='multiplex')
  # create a second ensemble
  e2 = Ensemble('e2', mode='multiplex')

  # you may use the ensembles as long as you specify which model you use
  print(e1(child='f', x=2))
  print(e1(child='g', y=3))
  print(e2(child='f', x=2))

  # try to use model `g` but it's not in ensemble `e2`
  try:
    print(e2(child='g', y=3))
  except ValueError:
    pass

  # try to use model `h` but it's not decorated with @model
  try:
    print(e1(child='h', y=3))
  except ValueError:
    pass

  # you may call your functions normally
  print(f(1))
  print(g(1))
  print(f(1))

  #print(e1)

  # send the same arguments to all the models in the ensemble and get all results
  e3 = Ensemble('e3')
  print(e3.all(x=2, y=3))

  def a(x):
    return x + 1

  def b(y):
    return y + 2

  def c(z):
    return z + 2

  # you may directly specify model functions to the ensemble
  e4 = Ensemble('e4', children=[a, b], mode='multiplex')
  print(e4(child='a', x=4))
  print(e4(child='b', y=4))

  print(e4.mean(x=2, y=3))

  e5 = Ensemble('e5', children=[a, b], weights=[3.0, 1.0]) 
  print(e5.weighted_mean(x=2, y=3))
  print(e5.weighted_sum(x=2, y=3))
  #print(e5)

  e6 = Ensemble('e6', [a, b, c])
  print(e6.vote(x=1, y=1, z=1))

  e7 = Ensemble('e7', children=[c])

  e8 = Ensemble('e8', children=[e4, e7], mode='all')
  e4.set_mode('all')
  e7.set_mode('all')
  print(e8(x=1, y=1, z=1))
  e4.set_mode('sum')
  e7.set_mode('sum')
  e8.set_mode('mean')
  print(e8(x=1, y=1, z=1))
  e8.wrap(lambda ret: ret + 1)
  print(e8(x=1, y=1, z=1))
  e8.wrap_children(lambda ret: ret + 1)
  print(e8(x=1, y=1, z=1))

  e = Ensemble('ab', [a, b])
  print(e(x=1, y=1))
  print(e(dict(a=dict(x=2), b=dict(y=2))))
  print(e8)










