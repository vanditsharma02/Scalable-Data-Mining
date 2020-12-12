import matplotlib.pyplot as plt
import random
import time
import numpy as np
from sklearn import metrics
from math import sqrt
from sklearn.model_selection import train_test_split

def diffa(y, ypred,x):
    return (y-ypred)*(-x)

def diffb(y, ypred):
    return (y-ypred)*(-1)

def shuffle_data(x,y):
    # shuffle x，y，while keeping x_i corresponding to y_i
    seed = random.random()
    random.seed(seed)
    random.shuffle(x)
    random.seed(seed)
    random.shuffle(y)

def get_batch_data(x, y, batch):
    shuffle_data(x, y)
    x_batch = x[0:batch]
    y_batch = y[0:batch]
    return [x_batch, y_batch]

data = np.loadtxt('LinearRegdata.txt')
x = data[:, 1]
y = data[:, 2]

# Normalize the data
x_max = max(x)
x_min = min(x)
y_max = max(y)
y_min = min(y)
for i in range(0, len(x)):
    x[i] = (x[i] - x_min)/(x_max - x_min)
    y[i] = (y[i] - y_min)/(y_max - y_min)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=109)

def batch_gd(X_train, y_train, X_test, y_test, learning_rate = 0.008):

    a = 10.0
    b = -20.0

    all_bgdloss = []
    all_ep = []
    start = time.time()
    for ep in range(1,100):
        loss = 0
        losst = 0
        all_da = 0
        all_db = 0
        for i in range(0, len(X_train)):
            y_pred = a*X_train[i] + b
            loss = loss + (y_train[i] - y_pred)*(y_train[i] - y_pred)/2
            all_da = all_da + diffa(y_train[i], y_pred, X_train[i]) #gradients accumulated
            all_db = all_db + diffb(y_train[i], y_pred) #gradients accumulated

        loss = loss/len(X_train)
        all_bgdloss.append(loss)
        all_ep.append(ep)

        #parameters updated
        a = a - learning_rate * all_da
        b = b - learning_rate * all_db

        for i in range(0, len(X_test)):
            y_pred = a*X_test[i] + b
            losst = losst + (y_test[i] - y_pred)*(y_test[i] - y_pred)/2

        #Saving best parameters for final reporting on test set
        if ep==1:
            prevloss = losst
        else:
            if losst < prevloss:
                prevloss=losst
                param1 = a
                param2 = b
    stop = time.time()
    print('time taken:' + str(stop-start))
    #Complete the function to obtain the plots and RMSE
    plt.figure()
    plt.plot(all_bgdloss)
    plt.title('Training Loss vs Updates')
    plt.xlabel('Update number')
    plt.ylabel('Training Loss')
    plt.show()
    plt.figure()
    plt.plot(all_bgdloss)
    plt.title('Training Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.show()

    return [param1, param2, prevloss]

def minibatch_gd(X_train, y_train, X_test, y_test, batch_size = 10, learning_rate = 0.008):

	a = 10.0
	b = -20.0

	all_mbgdloss = []
	all_updateloss = []
	all_ep = []
	start = time.time()
	for ep in range(1,25):
		loss_batch = 0
		losst = 0
		permutation = list(np.random.permutation(len(X_train)))
		shuffled_X_train = []
		shuffled_y_train = []
		for p in permutation:
			shuffled_X_train.append(X_train[p])
			shuffled_y_train.append(y_train[p])
		shuffled_X_train = np.array(shuffled_X_train)
		shuffled_y_train = np.array(shuffled_y_train)
		for batch in range(0, len(X_train), batch_size):
			loss_update = 0
			all_da = 0
			all_db = 0
			batch_X_train = shuffled_X_train[batch : batch + batch_size]
			batch_y_train = shuffled_y_train[batch : batch + batch_size]
			for i in range(0, len(batch_X_train)):
				y_pred = a * batch_X_train[i] + b
				loss_update = loss_update + (batch_y_train[i] - y_pred) * (batch_y_train[i] - y_pred) / 2
				all_da = all_da + diffa(batch_y_train[i], y_pred, batch_X_train[i])  # gradients accumulated
				all_db = all_db + diffb(batch_y_train[i], y_pred)  # gradients accumulated
			loss_update = loss_update / len(batch_X_train)
			all_updateloss.append(loss_update)
			loss_batch = loss_batch + loss_update
			# parameters updated
			a = a - learning_rate * all_da
			b = b - learning_rate * all_db

		all_mbgdloss.append(loss_batch * batch_size/len(X_train))
		all_ep.append(ep)

		for i in range(0, len(X_test)):
			y_pred = a * X_test[i] + b
			losst = losst + (y_test[i] - y_pred) * (y_test[i] - y_pred) / 2

		# Saving best parameters for final reporting on test set
		if ep == 1:
			prevloss = losst
		else:
			if losst < prevloss:
				prevloss = losst
				param1 = a
				param2 = b

	stop = time.time()
	print('time taken:' + str(stop - start))
	# Complete the function to obtain the plots and RMSE
	plt.figure()
	plt.plot(all_updateloss)
	plt.title('Training Loss vs Updates')
	plt.xlabel('Update number')
	plt.ylabel('Training Loss')
	plt.show()
	plt.figure()
	plt.plot(all_mbgdloss)
	plt.title('Training Loss vs Epochs')
	plt.xlabel('Epoch')
	plt.ylabel('Training Loss')
	plt.show()

	return [param1, param2, prevloss]

def stochastic_gd(X_train, y_train, X_test, y_test, learning_rate = 0.008):

	# sgd same as mbgd with batch_size = 1
	batch_size = 1
	a = 10.0
	b = -20.0

	all_sgdloss = []
	all_updateloss = []
	all_ep = []
	start = time.time()
	for ep in range(1, 100):
		loss_batch = 0
		losst = 0
		permutation = list(np.random.permutation(len(X_train)))
		shuffled_X_train = []
		shuffled_y_train = []
		for p in permutation:
			shuffled_X_train.append(X_train[p])
			shuffled_y_train.append(y_train[p])
		shuffled_X_train = np.array(shuffled_X_train)
		shuffled_y_train = np.array(shuffled_y_train)
		for batch in range(0, len(X_train), batch_size):
			loss_update = 0
			all_da = 0
			all_db = 0
			batch_X_train = shuffled_X_train[batch: batch + batch_size]
			batch_y_train = shuffled_y_train[batch: batch + batch_size]
			for i in range(0, len(batch_X_train)):
				y_pred = a * batch_X_train[i] + b
				loss_update = loss_update + (batch_y_train[i] - y_pred) * (batch_y_train[i] - y_pred) / 2
				all_da = all_da + diffa(batch_y_train[i], y_pred, batch_X_train[i])  # gradients accumulated
				all_db = all_db + diffb(batch_y_train[i], y_pred)  # gradients accumulated
			loss_update = loss_update / len(batch_X_train)
			all_updateloss.append(loss_update)
			loss_batch = loss_batch + loss_update
			# parameters updated
			a = a - learning_rate * all_da
			b = b - learning_rate * all_db

		all_sgdloss.append(loss_batch * batch_size / len(X_train))
		all_ep.append(ep)

		for i in range(0, len(X_test)):
			y_pred = a * X_test[i] + b
			losst = losst + (y_test[i] - y_pred) * (y_test[i] - y_pred) / 2

		# Saving best parameters for final reporting on test set
		if ep == 1:
			prevloss = losst
		else:
			if losst < prevloss:
				prevloss = losst
				param1 = a
				param2 = b

	stop = time.time()
	print('time taken:' + str(stop - start))
	# Complete the function to obtain the plots and RMSE
	plt.figure()
	plt.plot(all_updateloss)
	plt.title('Training Loss vs Updates')
	plt.xlabel('Update number')
	plt.ylabel('Training Loss')
	plt.show()
	plt.figure()
	plt.plot(all_sgdloss)
	plt.title('Training Loss vs Epochs')
	plt.xlabel('Epoch')
	plt.ylabel('Training Loss')
	plt.show()

	return [param1, param2, prevloss]

def momentum_gd(X_train, y_train, X_test, y_test, V = [0, 0], gamma = 0.9, learning_rate = 0.008):

	batch_size = 1
	a = 10.0
	b = -20.0
	Va = V[0]
	Vb = V[1]

	all_msgdloss = []
	all_updateloss = []
	all_ep = []
	start = time.time()
	for ep in range(1, 100):
		loss_batch = 0
		losst = 0
		permutation = list(np.random.permutation(len(X_train)))
		shuffled_X_train = []
		shuffled_y_train = []
		for p in permutation:
			shuffled_X_train.append(X_train[p])
			shuffled_y_train.append(y_train[p])
		shuffled_X_train = np.array(shuffled_X_train)
		shuffled_y_train = np.array(shuffled_y_train)
		for batch in range(0, len(X_train), batch_size):
			loss_update = 0
			all_da = 0
			all_db = 0
			batch_X_train = shuffled_X_train[batch: batch + batch_size]
			batch_y_train = shuffled_y_train[batch: batch + batch_size]
			for i in range(0, len(batch_X_train)):
				y_pred = a * batch_X_train[i] + b
				loss_update = loss_update + (batch_y_train[i] - y_pred) * (batch_y_train[i] - y_pred) / 2
				all_da = all_da + diffa(batch_y_train[i], y_pred, batch_X_train[i])  # gradients accumulated
				all_db = all_db + diffb(batch_y_train[i], y_pred)  # gradients accumulated
			loss_update = loss_update / len(batch_X_train)
			all_updateloss.append(loss_update)
			loss_batch = loss_batch + loss_update
			# parameters updated
			Va = gamma * Va + learning_rate * all_da
			Vb = gamma * Vb + learning_rate * all_db
			a = a - Va
			b = b - Vb

		all_msgdloss.append(loss_batch * batch_size / len(X_train))
		all_ep.append(ep)

		for i in range(0, len(X_test)):
			y_pred = a * X_test[i] + b
			losst = losst + (y_test[i] - y_pred) * (y_test[i] - y_pred) / 2

		# Saving best parameters for final reporting on test set
		if ep == 1:
			prevloss = losst
		else:
			if losst < prevloss:
				prevloss = losst
				param1 = a
				param2 = b

	stop = time.time()
	print('time taken:' + str(stop - start))
	# Complete the function to obtain the plots and RMSE
	plt.figure()
	plt.plot(all_updateloss)
	plt.title('Training Loss vs Updates')
	plt.xlabel('Update number')
	plt.ylabel('Training Loss')
	plt.show()
	plt.figure()
	plt.plot(all_msgdloss)
	plt.title('Training Loss vs Epochs')
	plt.xlabel('Epoch')
	plt.ylabel('Training Loss')
	plt.show()

	return [param1, param2, prevloss]

def adam_gd(X_train, y_train, X_test, y_test, M = [0, 0], V = [0, 0], beta1 = 0.9, beta2 = 0.999, learning_rate = 0.2, t = 2, epsilon = 0.001):

	batch_size = len(X_train)
	a = 0
	b = 0
	Va = V[0]
	Vb = V[1]
	Ma = M[0]
	Mb = M[1]

	all_adamloss = []
	all_updateloss = []
	all_ep = []
	start = time.time()
	for ep in range(1, 100):
		loss_batch = 0
		losst = 0
		permutation = list(np.random.permutation(len(X_train)))
		shuffled_X_train = []
		shuffled_y_train = []
		for p in permutation:
			shuffled_X_train.append(X_train[p])
			shuffled_y_train.append(y_train[p])
		shuffled_X_train = np.array(shuffled_X_train)
		shuffled_y_train = np.array(shuffled_y_train)
		for batch in range(0, len(X_train), batch_size):
			loss_update = 0
			all_da = 0
			all_db = 0
			batch_X_train = shuffled_X_train[batch: batch + batch_size]
			batch_y_train = shuffled_y_train[batch: batch + batch_size]
			for i in range(0, len(batch_X_train)):
				y_pred = a * batch_X_train[i] + b
				loss_update = loss_update + (batch_y_train[i] - y_pred) * (batch_y_train[i] - y_pred) / 2
				all_da = all_da + diffa(batch_y_train[i], y_pred, batch_X_train[i])  # gradients accumulated
				all_db = all_db + diffb(batch_y_train[i], y_pred)  # gradients accumulated
			loss_update = loss_update / len(batch_X_train)
			all_updateloss.append(loss_update)
			loss_batch = loss_batch + loss_update
			# parameters updated
			Ma = beta1 * Ma + (1 - beta1) * all_da
			Mb = beta1 * Mb + (1 - beta1) * all_db
			Ma = Ma / (1 - pow(beta1, t))
			Mb = Mb / (1 - pow(beta1, t))
			Va = beta2 * Va + (1 - beta2)*pow(all_da, 2)
			Vb = beta2 * Vb + (1 - beta2)*pow(all_db, 2)
			Va = Va / (1 - pow(beta2, t))
			Vb = Vb / (1 - pow(beta2, t))
			a = a - learning_rate*Ma/(sqrt(Va) + epsilon)
			b = b - learning_rate*Mb/(sqrt(Vb) + epsilon)

		all_adamloss.append(loss_batch * batch_size / len(X_train))
		all_ep.append(ep)

		for i in range(0, len(X_test)):
			y_pred = a * X_test[i] + b
			losst = losst + (y_test[i] - y_pred) * (y_test[i] - y_pred) / 2

		# Saving best parameters for final reporting on test set
		if ep == 1:
			prevloss = losst
			param1 = a
			param2 = b
		else:
			if losst < prevloss:
				prevloss = losst
				param1 = a
				param2 = b

	stop = time.time()
	print('time taken:' + str(stop - start))
	# Complete the function to obtain the plots and RMSE
	plt.figure()
	plt.plot(all_updateloss)
	plt.title('Training Loss vs Updates')
	plt.xlabel('Update number')
	plt.ylabel('Training Loss')
	plt.show()
	plt.figure()
	plt.plot(all_adamloss)
	plt.title('Training Loss vs Epochs')
	plt.xlabel('Epoch')
	plt.ylabel('Training Loss')
	plt.show()

	return [param1, param2, prevloss]

def main():

	p1 = batch_gd(X_train, y_train, X_test, y_test, 0.008)
	p2 = minibatch_gd(X_train, y_train, X_test, y_test, 10, 0.02)
	p3 = stochastic_gd(X_train, y_train, X_test, y_test, 0.02)
	p4 = momentum_gd(X_train, y_train, X_test, y_test, [0, 0], 0.9, 0.02)
	p5 = adam_gd(X_train, y_train, X_test, y_test, [0, 0], [0, 0], 0.9, 0.999, 0.02, 2, 0.001)

	errors = [sqrt(p1[2]*2/len(X_test)), sqrt(p2[2]*2/len(X_test)), sqrt(p3[2]*2/len(X_test)), sqrt(p4[2]*2/len(X_test)), sqrt(p5[2]*2/len(X_test))]

	loc = np.arange(5)
	plt.bar(loc, errors)
	plt.ylabel('RMSE')
	plt.title('RMSE vs Optimizer')
	plt.xticks(loc, ('Batch', 'Mini Batch', 'Stochastic', 'Momentum', 'Adam'))
	plt.show()

if __name__=='__main__':
	main()