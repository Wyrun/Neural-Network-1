import numpy as np

b = np.array([1, 2, 3, 4])

my_arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

print(my_arr)

print(my_arr[1, 2])

print(my_arr.ndim) # the rank

print(my_arr.size) # number of elements

print(type(my_arr)) # element type

####
print("#"*20)

a = np.ones((3, 2)) # array of 1s
print(a)

b = np.zeros((3, 4)) # array of 0s
print(b)

c = np.random.random(3) # array of random values
print(c)

d = np.full((2, 2), 12) # array filled with constant values
print(d)

#d[1, 1] = 13
#print(d)

# np.empty() create an array of uninitialized elements

# np.eye() create identity arrays for matrix calculations

# np.indentity() create identity arrays for matrix calculations

d = np.full((2, 2), 12, dtype = np.float32) # specifying the type

a = b.copy() # copy()

# loadtxt() to load a file into an array

# save() is used to save a numpy array to a file

####
print("#"*20)

a = np.arange(0, 10, 2)
print(a)

b = np.arange(6)
print(b)

c = np.linspace(0, 10, 6)
print(c)

####
print("#"*20)

a = np.arange(10)
print(a.shape)
print(a)

a.resize(2, 5)
print(a.shape)
print(a)

b = np.array([[1, 2, 3], [4, 5, 6]])
print(b)

print(b.reshape(3, 2))

print(b.shape)

print(b.ravel())

print(b.shape)

a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.transpose())

x = np.arange(8)

y = x[1:5:2]
print(y)

a = np.array([[0,1,2],[3,4,5],[6,7,8]])
print(a[0:2, 1])
print(a[..., 1])
print(a[:, 1])

####
print("#"*20)

x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
y = np.array([[9,10,11,12], [13,14,15,16]])

print(np.add(x,y))
print(np.remainder(y,x))
print(x**2)
print(y - x)
print(x < 5)

# np.dot() - matrix dot product

####
print("#"*20)

x = np.array([[1, 2], [3, 4]])
print(np.exp(x)) # exponential
print(np.sqrt(x)) # exponential

print(x.sum())
print(np.min(x))
print(x.max())
print(np.cumsum(x))
print(x.mean())
# np.median
print(np.corrcoef(x)) # correlation coefficient for the array
print(x.std()) # standard deviation for the array

####
print("#"*20)

two_d = np.array([[1,2,3,4], [5,6,7,8]]) # 2 x 4
one_d = np.array([[10]]) # 1
three_d = np.ones((3, 2)) # 3 x 2

print(np.add(two_d, one_d))
print(np.add(one_d, three_d))
#print(np.add(two_d, three_d)) - ERROR
