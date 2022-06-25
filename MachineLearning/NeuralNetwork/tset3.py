import numpy as np


price = np.array([53.80, 65.20, 77.20, 92.10 * 2, 250.40 * 2])
sum = price.sum()
print(price)
print(sum)
price_2 = []

for p in price:
	rate = (p / sum)
	price_2.append(rate * 561.2)

print(price_2)
print(price_2[-2]/2)
a = 168 + 85
b = 168 / a
c = 85 / a
print((price_2[-1]/2) * b)
print((price_2[-1]/2) * c)