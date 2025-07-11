#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import re
import nltk
import nltk.corpus 
import stopwords
import string
import os
import json
from time import time
import pickle
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate


# In[53]:


def readTextFile(path):
    with open(path) as f:
        captions = f.readlines()
    return captions


# In[55]:


captions = readTextFile('/Data/Flickr8k_text/Flickr8k.token.txt'.replace(" ","/"))


# In[4]:


first,second = captions[0].split('\t')
#print(first.split(".")[0])
#print(second)


# In[5]:


descriptions  = {}

for x in captions:
    first,second = x.split('\t')
    img_name = first.split(".")[0]

    if descriptions.get(img_name) is None:
        descriptions[img_name] = []

    
    descriptions[img_name].append(second.split("\n")[0])


# In[6]:


#descriptions["1000268201_693b08cb0e"]


# In[7]:


IMG_PATH = "/Study Related all DATA/Major Project (MCA_NEW)/Jupyter Notebook/Major Project (Caption Craft)/Data/Flickr8k_Dataset/"
import cv2
import matplotlib.pyplot as plt

img = cv2.imread(IMG_PATH+"1000268201_693b08cb0e.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#plt.imshow(img)
#plt.axis("off")
#plt.show()


# In[8]:


def clean_text(sentence):
    sentence = sentence.lower()
    sentence = re.sub("[^a-z]+"," ",sentence)
    sentence = sentence.split()

    sentence = [s for s in sentence if len(s)>1]
    sentence = " ".join(sentence)
    return sentence


# In[9]:


#clean_text("A cat is sitting over the house #64.")


# In[10]:


for key,caption_list in descriptions.items():
    for i in range(len(caption_list)):
        caption_list[i] = clean_text(caption_list[i])


# In[11]:


#descriptions["1000268201_693b08cb0e"]


# In[12]:


with open("descriptions_1.txt","w") as f:
    f.write(str(descriptions))


# In[13]:


descriptions = None
with open("descriptions_1.txt",'r') as f:
    descriptions = f.read()

json_acceptable_string = descriptions.replace("'","\"")
descriptions = json.loads(json_acceptable_string)


# In[14]:


#vocabula

vocab = set()

for key in descriptions.keys():
    [vocab.update(sentence.split()) for sentence in descriptions[key]]

print("Vocab Size : %d"%len(vocab))


# In[15]:


total_words = []

for key in descriptions.keys():
    [total_words.append(i) for des in descriptions[key] for i in des.split()]

print("Total Words %d"%len(total_words))


# In[16]:


import collections

counter = collections.Counter(total_words)
freq_cnt = dict(counter)
sorted_freq_cnt = sorted(freq_cnt.items(),reverse=True,key=lambda x:x[1])
threshold = 10
sorted_freq_cnt = [x for x in sorted_freq_cnt if x[1]>threshold]
total_words = [x[0] for x in sorted_freq_cnt]


# In[17]:


train_file_data = readTextFile("/Study Related all DATA/Major Project (MCA_NEW)/Jupyter Notebook/Major Project (Caption Craft)/Data/Flickr8k_text/Flickr_8k.trainImages.txt")
test_file_data = readTextFile("/Study Related all DATA/Major Project (MCA_NEW)/Jupyter Notebook/Major Project (Caption Craft)/Data/Flickr8k_text/Flickr_8k.testImages.txt")


# In[18]:


train = [row.split(".")[0] for row in train_file_data]
test = [row.split(".")[0] for row in test_file_data]


# In[19]:


train_descriptions = {}

for img_id in train:
    train_descriptions[img_id] = []
    for cap in descriptions[img_id]:
        cap_to_append = "startseq " + cap + " endseq"
        train_descriptions[img_id].append(cap_to_append)


# In[20]:


model = ResNet50(weights="imagenet",input_shape=(224,224,3))
#model.summary()


# In[21]:


model_new = Model(model.input,model.layers[-2].output)


# In[22]:


def preprocess_img(img):
    img = image.load_img(img,target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    img = preprocess_input(img)
    return img


# In[23]:


img = preprocess_img(IMG_PATH+"1000268201_693b08cb0e.jpg")
#plt.imshow(img[0]/255.0)
#plt.axis("off")
#plt.show()


# In[24]:


def encode_image(img):
    img = preprocess_img(img)
    feature_vector = model_new.predict(img)
    
    feature_vector = feature_vector.reshape((-1,))
    #print(feature_vector.shape)
    return feature_vector


# In[25]:


#encode_image(IMG_PATH+"1000268201_693b08cb0e.jpg")


# In[26]:


"""start = time()
encoding_train = {}

for ix,img_id in enumerate(train):
    img_path = IMG_PATH+"/"+img_id+".jpg"
    encoding_train[img_id] = encode_image(img_path)

    if ix%100==0:
        print("Encoding in Progress Time Step %d "%ix)

end_time = time()
print("Total Time Taken :",end_time-start)"""


# In[27]:


"""with open("/Study Related all DATA/Major Project (MCA_NEW)/Jupyter Notebook/Major Project (Caption Craft)/Data/encoded_train_features.pkl","wb") as f:
    pickle.dump(encoding_train,f)"""


# In[28]:


"""start = time()
encoding_test = {}

for ix,img_id in enumerate(test):
    img_path = IMG_PATH+"/"+img_id+".jpg"
    encoding_test[img_id] = encode_image(img_path)

    if ix%100==0:
        print("Test Encoding in Progress Time Step %d "%ix)

end_time = time()
print("Total Time Taken(Test) :",end_time-start)"""


# In[29]:


"""with open("/Study Related all DATA/Major Project (MCA_NEW)/Jupyter Notebook/Major Project (Caption Craft)/Data/encoded_test_features.pkl","wb") as f:
    pickle.dump(encoding_test,f)"""


# In[30]:


encoded_train_features = "/Study Related all DATA/Major Project (MCA_NEW)/Jupyter Notebook/Major Project (Caption Craft)/Data/encoded_train_features.pkl"
encoded_test_features = "/Study Related all DATA/Major Project (MCA_NEW)/Jupyter Notebook/Major Project (Caption Craft)/Data/encoded_test_features.pkl"

with open(encoded_train_features, "rb") as f:
    encoding_train_file = pickle.load(f)

with open(encoded_test_features, "rb") as f:
    encoding_test_file = pickle.load(f)


# In[31]:


word_to_index = {}
index_to_word = {}

for i,word in enumerate(total_words):
    word_to_index[word] = i+1
    index_to_word[i+1] = word


# In[32]:


index_to_word[1846] = 'startseq'
word_to_index['startseq'] = 1846
index_to_word[1847] = 'endseq'
word_to_index['endseq'] = 1847
vocab_size = len(word_to_index) + 1
#print("Vocab Size", vocab_size)


# In[33]:


max_len = 0
for key in train_descriptions.keys():
    for cap in train_descriptions[key]:
        max_len = max(max_len,len(cap.split()))

#print(max_len)


# In[34]:


def data_generator(train_descriptions,encoding_train_file,word_to_index,max_len,vocab_size,batch_size):
    X1,X2,Y = [],[],[]
    n = 0
    
    while True:
        for key,desc_list in train_descriptions.items():
            n += 1

            photo = encoding_train_file[key]
            for desc in desc_list:
                
                seq = [word_to_index[word] for word in desc.split() if word in word_to_index]
                for i in range(1,len(seq)):
                    xi = seq[0:i]
                    yi = seq[i]
                    xi = pad_sequences([xi],maxlen=max_len,value=0,padding='post')[0]
                    yi = to_categorical([yi],num_classes=vocab_size)[0]
                    X1.append(photo)
                    X2.append(xi)
                    Y.append(yi)
                    
                if n==batch_size:
                    yield[[np.array(X1),np.array(X2)],np.array(Y)]
                    X1,X2,Y = [],[],[]
                    n = 0


# In[35]:


f = open("/Study Related all DATA/Major Project (MCA_NEW)/Jupyter Notebook/Major Project (Caption Craft)/Data/glove.6B.50d.txt",encoding='utf8')


# In[36]:


embedding_index = {}

for line in f:
    values = line.split()
    word = values[0]
    word_embedding = np.array(values[1:],dtype = 'float')
    embedding_index[word] = word_embedding


# In[37]:


f.close()


# In[38]:


def get_embedding_matrix():
    emb_dim = 50
    matrix = np.zeros((vocab_size,emb_dim))
    for word,index in word_to_index.items():
        embedding_vector = embedding_index.get(word)

        if embedding_vector is not None:
            matrix[index] = embedding_vector

    return matrix


# In[39]:


embedding_matrix = get_embedding_matrix()
embedding_matrix.shape


# In[40]:


input_img_features = Input(shape=(2048,))
input_img1 = Dropout(0.3)(input_img_features)
input_img2 = Dense(256,activation='relu')(input_img1)


# In[41]:


input_captions = Input(shape=(max_len,))
input_cap1 = Embedding(input_dim=vocab_size,output_dim=50,mask_zero=True)(input_captions)
input_cap2 = Dropout(0.3)(input_cap1)
input_cap3 = LSTM(256)(input_cap2)


# In[42]:


decoder1 = add([input_img2,input_cap3])
decoder2 = Dense(256,activation='relu')(decoder1)
outputs = Dense(vocab_size,activation='softmax')(decoder2)

model = Model(inputs=[input_img_features,input_captions],outputs=outputs)


# In[43]:


model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False
model.compile(loss='categorical_crossentropy', optimizer=Adam())


# ## Training Model

# In[44]:


epochs = 50
batch_size = 2
steps = len(train_descriptions)//batch_size


# In[45]:


"""
for i in range(epochs):
    generator = data_generator(train_descriptions, encoding_train_file, word_to_index, max_len, vocab_size, batch_size)
    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
"""


# ### To save Training Data

# In[46]:


#model.save('./model_weights/model_test(50epochs)'+'.h5')


# ## Predictions

# In[47]:


def predict_caption(photo):
    model_load = load_model('/Study Related all DATA/Major Project (MCA_NEW)/Jupyter Notebook/Major Project (Caption Craft)/model_weights/model_test(50epochs).h5')
    in_text = "startseq"
    
    for i in range(max_len):
        sequence = [word_to_index[w] for w in in_text.split() if w in word_to_index]
        sequence = pad_sequences([sequence], maxlen=max_len, padding='post')

        ypred =  model_load.predict([photo,sequence])
        ypred = ypred.argmax()
        word = index_to_word[ypred]
        in_text += (' ' + word)
        
        if word =='endseq':
            break
        
        
    final_caption = in_text.split()
    final_caption = final_caption[1:-1]
    final_caption = ' '.join(final_caption)
    
    return final_caption


# In[48]:


"""
for i in range(5):
    rn =  np.random.randint(0, 1000)

    img_name = list(encoding_test_file.keys())[rn]
    photo = encoding_test_file[img_name].reshape((1,2048))

    i = plt.imread(IMG_PATH +"/" +img_name+".jpg")
    plt.imshow(i)
    plt.axis("off")
    plt.show()

    caption = predict_caption(photo)
    print(caption)
"""


# In[57]:


import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


def open_image():
    global file_path
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpeg *.jpg *.png")]
    )
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((800, 800))
        img = ImageTk.PhotoImage(img)
        label.config(image=img)
        label.image = img


def generate_caption():
    global file_path
    if not hasattr(label, 'image'):
        messagebox.showerror("Error: Image Not Found", "Uh-oh! We can't proceed without an image. Please select an image. then click 'Generate Caption'.")
        return

    photo = encode_image(file_path)
    photo = np.reshape(photo, (1, 2048))
    caption = predict_caption(photo)
    caption_var.set("Caption: " + caption)
    

root = tk.Tk()
root.geometry("1350x900")
root.resizable(0, 0)
root.title("CAPTION CRAFT: ADVANCED AI IMAGE CAPTIONING SYSTEM")

project_img = Image.open("/Study Related all DATA\Major Project (MCA_NEW)/Jupyter Notebook/Major Project (Caption Craft)/Main/AI_Logo.png")
project_img = project_img.resize((60, 60))
project_img = ImageTk.PhotoImage(project_img)
project_label = tk.Label(root, image=project_img)
project_label.pack(padx=0, pady=0)

heading_label = tk.Label(root, text="CAPTION CRAFT: ADVANCED AI IMAGE CAPTIONING SYSTEM", font=("Helvetica", 14, "bold"))
heading_label.pack(padx=0, pady=0)

predefined_img = Image.open("/Study Related all DATA\Major Project (MCA_NEW)/Jupyter Notebook/Major Project (Caption Craft)/Main/No_Image_Available.jpg")
predefined_img.thumbnail((500, 500))
predefined_img = ImageTk.PhotoImage(predefined_img)
label = tk.Label(root, image=predefined_img)
label.pack(padx=10, pady=10)


button = tk.Button(root, text="Open Image", command=open_image, height=2, width=30, font=("Helvetica", 15, "bold"), borderwidth=4, relief="raised")
button.pack(padx=10, pady=5)

button_generate = tk.Button(root, text="Generate Caption", command=generate_caption, height=2, width=30, font=("Helvetica", 15, "bold"), borderwidth=4, relief="raised")
button_generate.pack(padx=10, pady=5)

caption_var = tk.StringVar()
caption_label = tk.Label(root, textvariable=caption_var, font=("Helvetica", 18, "italic"), bg="lightgrey")
caption_label.pack(padx=10, pady=20)

root.mainloop()


# In[ ]:




