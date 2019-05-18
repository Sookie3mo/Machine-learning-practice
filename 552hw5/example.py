
import math

def load_pgm_image(pgm):
    with open(pgm, 'rb') as f:
        f.readline()   # skip P5
        f.readline()   # skip the comment line
        xs, ys = f.readline().split()  # size of the image
        xs = int(xs)
        ys = int(ys)
        max_scale = int(f.readline().strip())

        image = []
        for _ in range(xs * ys):

            image.append(f.read(1)[0] /max_scale)
        return image

images = []
labels = []

with open('downgesture_train.list') as f:
    for training_image in f.readlines():
        training_image = training_image.strip()
        images.append(load_pgm_image(training_image))
        if 'down' in training_image:
            labels.append(1)
        else:
            labels.append(0)
print (images)

#sigmoid(logistic) function: 1/(1+e*(-(y*wT*x)))
def sigmoidFunction(x):
    result = 1.0 / (1 + math.exp(-x))
    return result

def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m
hiddenSize = 100

#initial weight = 0
weightIJ = makeMatrix(len(images),hiddenSize,0.0)
weightJK = makeMatrix(hiddenSize,len(images),0.0)

#propagate the input forward, calculate the outputs of all units
def forwardPropagate(input):

    for i in range(len(input) - 1):
        inputJ = input[i]

    for j in range(hiddenSize):
        sum = 0.0
        for i in range(len(input)):

            sum = sum + inputJ[i] * weightIJ[i][j]
        outputJ = sigmoidFunction(sum)

    for k in range(len(images)):
        sum = 0.0
        for j in range(hiddenSize):
            sum = sum + outputJ[j] * weightJK[j][k]
        outputK = sigmoidFunction(sum)

    return outputK,outputJ


def backPropagate(outputK,outputJ,target,alpha):
    error = 0.0
    for k in range(len(outputJ)):
        error = outputK[k]*(1-outputK[k])*(target-outputK[k])
        for j in range(0,100):
            weightJK = weightJK + alpha * error * outputK[k]

    for j in range(0,100):
        sum = 0.0
        for k in range(len(outputJ)):
            sum = sum + weightJK[j][k] * error
            error = outputJ[j] * (1-outputJ[j]) * sum
        for i in range(len(images)):
            weightIJ = weightIJ + alpha * error * input
    return error

epoch = 1000

for i in range(epoch):
    error = 0.0
    outputK,outputJ = forwardPropagate(images)
    error = error + backPropagate(outputK,outputJ,labels,0.1)


print (error)

# c = MLPClassifier(solver='sgd', alpha=0,
#                   hidden_layer_sizes=(100,), activation='logistic', learning_rate_init=0.1,
#                   max_iter=10000)
# c.fit(images, labels)
#
# total = 0
# correct = 0
# with open('downgesture_test.list') as f:
#     total += 1
#     for test_image in f.readlines():
#         test_image = test_image.strip()
# load =  [load_pgm_image(test_image),]

#         p = c.predict([load_pgm_image(test_image),])[0]
#         print('{}: {}'.format(test_image, p))
#         if (p != 0) == ('down' in test_image):
#             correct += 1
# print('correct rate: {}'.format(correct / total))
