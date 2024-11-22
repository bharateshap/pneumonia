#Importing Datasets 
import tensorflow 
from keras.models import Model  
from keras.layers import Flatten,Dense  
from keras.applications.vgg16 import VGG16  
from keras.callbacks import EarlyStopping, ModelCheckpoint 
import matplotlib.pyplot as plot  
from glob import glob 
 
#Defining Constants and Datapaths 
IMAGESHAPE = [224, 224, 3]  
training_data = 'chest_xray/train' 
testing_data = 'chest_xray/test' 
 
#Loading Pre-trained VGG16 Model 
vgg_model = VGG16(input_shape=IMAGESHAPE, weights='imagenet', include_top=False)  
for each_layer in vgg_model.layers:  
    each_layer.trainable = False 
 
#Adding Custom Layers 
classes = glob('chest_xray/train/*')  
flatten_layer = Flatten()(vgg_model.output)  
prediction = Dense(len(classes), activation='softmax')(flatten_layer) 
 
#Creating the Final Model 
final_model = Model(inputs=vgg_model.input, outputs=prediction)  
final_model.summary()  
 
#Compiling the Model 
final_model.compile(  
loss='categorical_crossentropy',  
optimizer='adam',  
metrics=['accuracy']  
) 
 
#Data Augmentation and Loading 
ImageFlow= tensorflow.keras.preprocessing.image.ImageDataGenerator(rescale = 
1/255,  
                                shear_range = 0.2,  
                                zoom_range = 0.2,  
                                horizontal_flip = True, 
 
                                 validation_split=0.2)  
testdatagen= tensorflow.keras.preprocessing.image.ImageDataGenerator(rescale = 
1/255,  
                                shear_range = 0.2,  
                                zoom_range = 0.2,  
                                horizontal_flip = True, 
                                )  
training_set = testdatagen.flow_from_directory('chest_xray/train',   
                                                 target_size = (224, 224),  
                                                 batch_size = 16, 
                                                 class_mode = 'categorical')  
"""val_set = ImageFlow.flow_from_directory('chest_xray/train',   
                                                 target_size = (224, 224),  
                                                 batch_size = 16, 
                                                 subset='validation', 
                                                 class_mode = 'categorical') """ 
test_set = testdatagen.flow_from_directory('chest_xray/test',   
                                                 target_size = (224, 224),  
                                                 batch_size = 16, 
                                                 class_mode = 'categorical', 
                                                 )  
print(f"Number of training samples: {training_set.samples}") 
print(f"Number of validation samples: {test_set.samples}") 
steps_per_epoch = training_set.samples // training_set.batch_size 
validation_steps = test_set.samples // test_set.batch_size 
 
#Training the Model 
early_stopping = EarlyStopping(monitor='val_loss', patience=5, 
restore_best_weights=True) 
model_checkpoint = ModelCheckpoint('model.h5', save_best_only=True) 
fitted_model = final_model.fit( 
  x=training_set,  
  validation_data=test_set,  
  epochs=10,  
  steps_per_epoch=steps_per_epoch, 
  validation_steps=validation_steps, 
  callbacks=[early_stopping, model_checkpoint]) 
 
#Evaluating the Model 
test_loss, test_accuracy = final_model.evaluate(test_set, steps=39)  # 624 / 16 
 
print(f'Test Loss: {test_loss}') 
print(f'Test Accuracy: {test_accuracy}') 