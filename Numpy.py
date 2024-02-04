import numpy as np 
import time



# a1 = np.arange(10)      ### vector

# a2 = np.arange(12,dtype=float).reshape(3,4)      ### matrix

# a3 = np.arange(12).reshape(3,2,2)       ### tensors  this tells you there are 3, 2d array with 2 rows and 2 columns 


# a3 = a3.astype(np.int32)

# print(a3.itemsize)
 



# a1 = np.random.random((3,3))

# a1 = np.round(a1*100)


# print(a1)
# print(np.max(a1,axis=1))



### indexing and slicing 


# a1 = np.arange(12).reshape(3,4)

# print(a1[::2,::3])

# print(a1[::2,1::2])

# print(a1[1,::3])

# print(a1[0:2,1:])

# print(a1[0:2,1::2])


# a3 = np.arange(27).reshape(3,3,3)

# print(a3[1])

# print(a3[1,1,1:])

# print(a3[::2])

# print(a3[0,1,:])

# print(a3[1,:,1])

# print(a3[2,1:,1:])

# print(a3[::2,0,::2])



### iterating

# for i in np.nditer(a3):
#     print(i)


# a = [i for i in range(10000000)]
# b = [i for i in range(10000000,20000000)]

# c = []
# import time 

# start = time.time()
# for i in range(len(a)):
#   c.append(a[i] + b[i])
# print(time.time()-start)




# a = np.arange(10000000)
# b = np.arange(10000000,20000000)

# start = time.time()
# c = a + b
# print(time.time()-start)






### fancy indexing 

# a = np.arange(64).reshape(4,4,4)


# print(a[2,[0,2],1:3])



### boolean indexing

# b = np.random.randint(1,100,24).reshape(6,4)

# print(b[b>50])

# print(b[(b%2==0) & (b>50)])


 


### mathematical calculation in numpy 
### sigmoid function 


# def sigmoid(array):
#     return 1 /(1 + np.exp(array))


# a = np.arange(10)

# print(sigmoid(a))




# b = np.random.randint(1,100,24).reshape(6,4)

# print(b)
# print(np.min(b,axis=1))

# b = np.append(b,np.random.randint(1, 11, 6).reshape(6, 1), axis=1)

# print(b)




# my_array = np.array([[5, 10, 3],
#                      [8, 15, 7],
#                      [12, 6, 9]])

# # Find the minimum value in each column
# min_values_per_column = np.min(my_array, axis=0)

# print(min_values_per_column)



### where 

a = np.random.randint(1,100,30)


# a[np.where(a>30)]


## replace all the values with zero if any value > 50 

# np.where(a>50,0,a)

print(np.argmax(a))
