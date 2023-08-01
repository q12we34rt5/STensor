import stensor

a = stensor.create([1, 2, 3], requires_grad=True)
b = stensor.create([[4], [5]], requires_grad=True)

print(f'a:\n{a}\n')
print(f'b:\n{b}\n')


print('======== a + b ========')
c = a + b
c.backward()

print(f'c:\n{c}\n')
print(f'grad a:\n{a.grad}\n')
print(f'grad b:\n{b.grad}\n')

a.grad.fill(0)
b.grad.fill(0)


print('======== a - b ========')
c = a - b
c.backward()

print(f'c:\n{c}\n')
print(f'grad a:\n{a.grad}\n')
print(f'grad b:\n{b.grad}\n')

a.grad.fill(0)
b.grad.fill(0)


print('======== a * b ========')
c = a * b
c.backward()

print(f'c:\n{c}\n')
print(f'grad a:\n{a.grad}\n')
print(f'grad b:\n{b.grad}\n')

a.grad.fill(0)
b.grad.fill(0)


print('======== a / b ========')
c = a / b
c.backward()

print(f'c:\n{c}\n')
print(f'grad a:\n{a.grad}\n')
print(f'grad b:\n{b.grad}\n')

a.grad.fill(0)
b.grad.fill(0)
