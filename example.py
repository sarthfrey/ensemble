from ensemble import model, Ensemble


def h():
  pass

@model('e1', 'e2')
def f(x, y=3, z=4):
  return x + y + z

@model('e1')
def g(y):
  return y**3


if __name__ == '__main__':

  # create our first ensemble and give it a name
  e1 = Ensemble('e1')
  # create a second ensemble
  e2 = Ensemble('e2')

  # you may use the ensembles as long as you specify which model you use
  print(e1(model='f', x=2))
  print(e1(model='g', y=3))
  print(e2(model='f', x=2))

  # try to use model `g` but it's not in ensemble `e2`
  try:
    print(e2(model='g', y=3))
  except ValueError:
    pass

  # try to use model `h` but it's not decorated with @Model
  try:
    print(e1(model='h', y=3))
  except ValueError:
    pass

  # you may specify your arguments positionally as usual
  print(e1(3, model='f'))

  # you may call your functions normally
  print(f(1))
  print(g(1))
  print(f(1))