
import pygame


# pre defined colors, pen radius and font color
black = [0, 0, 0]
white = [255, 255, 255]
red = [255, 0, 0]
green = [0, 255, 0]
draw_on = False
last_pos = (0, 0)
color = (255, 128, 0)
radius = 7
font_size = 500

#image size
width = 1000
height = 640

# initializing screen
screen = pygame.display.set_mode((width, height))
screen.fill(white)
pygame.font.init()




def crope(orginal):
    cropped = pygame.Surface((width-5, height-5))
    cropped.blit(orginal, (0, 0), (0, 0, width-5, height-5))
    return cropped


def roundline(srf, color, start, end, radius=1):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = max(abs(dx), abs(dy))
    for i in range(distance):
        x = int(start[0] + float(i) / distance * dx)
        y = int(start[1] + float(i) / distance * dy)
        pygame.draw.circle(srf, color, (x, y), radius)


def draw_partition_line():
    pygame.draw.line(screen, black, [width, 0], [width,height ], 8)


try:
    while True:
        # get all events
        e = pygame.event.wait()
        draw_partition_line()

        # clear screen after right click
        if(e.type == pygame.MOUSEBUTTONDOWN and e.button == 3):
            #screen.fill(white)
            draw_on = False
            fname = "out.png"

            img = crope(screen)
            pygame.image.save(img, fname)
           
            raise StopIteration

        # quit
        if e.type == pygame.QUIT:
            raise StopIteration

        # start drawing after left click
        if(e.type == pygame.MOUSEBUTTONDOWN and e.button != 3):
            color = black
            pygame.draw.circle(screen, color, e.pos, radius)
            draw_on = True

        # stop drawing after releasing left click
        if e.type == pygame.MOUSEBUTTONUP and e.button != 3:
            draw_on = False
            fname = "out.png"

            img = crope(screen)
            pygame.image.save(img, fname)


        # start drawing line on screen if draw is true
        if e.type == pygame.MOUSEMOTION:
            if draw_on:
                pygame.draw.circle(screen, color, e.pos, radius)
                roundline(screen, color, e.pos, last_pos, radius)
            last_pos = e.pos

        pygame.display.flip()

except StopIteration:
 
    pass
#~~~1. Data set ~~~
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import seaborn as sns
import tensorflow as tf 
physical_devices = tf.config.list_physical_devices('GPU') 

tf.config.experimental.set_memory_growth(physical_devices[0], True)
np.random.seed(2)

'load the dataset'
dataset = pd.read_csv("/home/olivergv/DeepLearning/dataset.csv")

"creating label"
y = dataset["label"]

"dropping label"
X = dataset.drop(labels = ["label"], axis = 1)

"deleting dataset to reduce memory usage"
del dataset

'overview of dataset'
g = sns.countplot(y)
y.value_counts()

'Grayscale normalization to reduce the effect of illumination differences.'
X = X / 255.0

'reshaping the dataset to fit standard of a 4D tensor of shape [mini-batch size, height = 28px, width = 28px, channels = 1 due to grayscale].'
X = X.values.reshape(-1,28,28,1)

'categorical conversion of label'
y = to_categorical(y, num_classes = 14)

'90% Training and 10% Validation split'
random_seed = 2
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1 , random_state = random_seed, stratify = y)


#-------------------------------

#~~~2. Model~~~
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from tensorflow import keras

'creating the instance of the model'
model = keras.models.load_model('/home/olivergv/dl/model.h5')




#-------------------------------

#~~~3. Prediction~~~

from PIL import Image
from itertools import groupby

'loading image'
image = Image.open("/home/olivergv/dl/out.png").convert("L")

'resizing to 28 height pixels'
w = image.size[0]
h = image.size[1]
r = w / h # aspect ratio
new_w = int(r * 28)
new_h = 28
new_image = image.resize((new_w, new_h))

'converting to a numpy array'
new_image_arr = np.array(new_image)

'inverting the image to make background = 0'
new_inv_image_arr = 255 - new_image_arr

'rescaling the image'
final_image_arr = new_inv_image_arr / 255.0

'splitting image array into individual digit arrays using non zero columns'
m = final_image_arr.any(0)
out = [final_image_arr[:,[*g]] for k, g in groupby(np.arange(len(m)), lambda x: m[x] != 0) if k]


'''
iterating through the digit arrays to resize them to match input 
criteria of the model = [mini_batch_size, height, width, channels]
'''
num_of_elements = len(out)
elements_list = []

for x in range(0, num_of_elements):

    img = out[x]
    
    #adding 0 value columns as fillers
    width = img.shape[1]
    filler = (final_image_arr.shape[0] - width) / 2
    
    if filler.is_integer() == False:    #odd number of filler columns
        filler_l = int(filler)
        filler_r = int(filler) + 1
    else:                               #even number of filler columns
        filler_l = int(filler)
        filler_r = int(filler)
    
    arr_l = np.zeros((final_image_arr.shape[0], filler_l)) #left fillers
    arr_r = np.zeros((final_image_arr.shape[0], filler_r)) #right fillers
    
    #concatinating the left and right fillers
    help_ = np.concatenate((arr_l, img), axis= 1)
    element_arr = np.concatenate((help_, arr_r), axis= 1)
    
    element_arr.resize(28, 28, 1) #resize array 2d to 3d

    #storing all elements in a list
    elements_list.append(element_arr)


elements_array = np.array(elements_list)

'reshaping to fit model input criteria'
elements_array = elements_array.reshape(-1, 28, 28, 1)

'predicting using the model'
model = keras.models.load_model("...//model.h5")
elements_pred =  model.predict(elements_array)
elements_pred = np.argmax(elements_pred, axis = 1)



#-------------------------------

#~~~4. Mathematical Operation~~~

def math_expression_generator(arr):
    
    op = {
              10,   # = "/"
              11,   # = "+"
              12,   # = "-"
              13    # = "*"
                  }   
    
    m_exp = []
    temp = []
        
    'creating a list separating all elements'
    for item in arr:
        if item not in op:
            temp.append(item)
        else:
            m_exp.append(temp)
            m_exp.append(item)
            temp = []
    if temp:
        m_exp.append(temp)
        
    'converting the elements to numbers and operators'
    i = 0
    num = 0
    for item in m_exp:
        if type(item) == list:
            if not item:
                m_exp[i] = ""
                i = i + 1
            else:
                num_len = len(item)
                for digit in item:
                    num_len = num_len - 1
                    num = num + ((10 ** num_len) * digit)
                m_exp[i] = str(num)
                num = 0
                i = i + 1
        else:
            m_exp[i] = str(item)
            m_exp[i] = m_exp[i].replace("10","/")
            m_exp[i] = m_exp[i].replace("11","+")
            m_exp[i] = m_exp[i].replace("12","-")
            m_exp[i] = m_exp[i].replace("13","*")
            
            i = i + 1
    
    
    'joining the list of strings to create the mathematical expression'
    separator = ' '
    m_exp_str = separator.join(m_exp)
    
    return (m_exp_str)

'creating the mathematical expression'
m_exp_str = math_expression_generator(elements_pred)

'calculating the mathematical expression using eval()'
while True:
    try:
        answer = eval(m_exp_str)    #evaluating the answer
        answer = round(answer, 2)
        equation  = m_exp_str + " = " + str(answer)
        print(equation)   #printing the equation
        break

    except SyntaxError:
        print("Invalid predicted expression!!")
        print("Following is the predicted expression:")
        print(m_exp_str)
        break

#-------------------------------

#~~~5. Model Update~~~

#++++++++++++++++++++++++++++++++++++++++++++++++++++++
def model_update(X, y, model):
    
    from tensorflow.keras.optimizers import RMSprop
    from keras.utils.np_utils import to_categorical
    from keras.preprocessing.image import ImageDataGenerator
      
    y = to_categorical(y, num_classes = 14)
    
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(X)

    #freezing layers 0 to 4
    for l in range(0, 5):
        model.layers[l].trainable = False

    optimizer = RMSprop(lr = 0.0001, rho = 0.9, epsilon = 1e-08, decay=0.0 )
    model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])
        
    history = model.fit(
                            datagen.flow(X,y, batch_size = 1),
                            epochs = 50,
                            verbose = 1
                        )
    
    'saving the model'
    model.save("...//updated_model.h5") 
    
    print("Model has been updated!!")
#++++++++++++++++++++++++++++++++++++++++++++++++++++++











pygame.quit()
