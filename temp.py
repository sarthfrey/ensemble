class A:
  x = 1
  def get(self):
    return self.x

class B(A):
  pass

print(A().get())
print(B().get())
A.x = 2
print(A().get())
print(B().get())
