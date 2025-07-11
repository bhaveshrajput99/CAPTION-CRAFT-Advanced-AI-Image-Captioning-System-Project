#!/usr/bin/env python
# coding: utf-8

# In[1]:


import import_ipynb
import importlib
import matplotlib.pyplot as plt
import Caption_Craft_Main as Cap_bot
importlib.reload(Cap_bot)
from Caption_Craft_Main import encoding_test_file
IMG_PATH = "H:/Study Related all DATA/Major Project (MCA_NEW)/Jupyter Notebook/Major Project (Caption Craft)/Data/Flickr8k_Dataset/"


# In[2]:


"""
for i in range(1):
    rn =  Cap_bot.np.random.randint(0, 1000)

    img_name = list(encoding_test_file.keys())[rn]
    photo = encoding_test_file[img_name].reshape((1,2048))

    #i = plt.imread(IMG_PATH +"/" +img_name+".jpg")
    #plt.imshow(i)
    #plt.axis("off")
    #plt.show()

    caption = Cap_bot.predict_caption(photo)
    print(caption)
"""


# In[4]:


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

    photo = Cap_bot.encode_image(file_path)
    photo = np.reshape(photo, (1, 2048))
    caption = Cap_bot.predict_caption(photo)
    caption_var.set("Caption: " + caption)
    

root = tk.Tk()
root.geometry("1350x900")
root.resizable(0, 0)
root.title("CAPTION CRAFT: ADVANCED AI IMAGE CAPTIONING SYSTEM")

project_img = Image.open("H:/Study Related all DATA\Major Project (MCA_NEW)/Jupyter Notebook/Major Project (Caption Craft)/Main/AI_Logo.png")
project_img = project_img.resize((60, 60))
project_img = ImageTk.PhotoImage(project_img)
project_label = tk.Label(root, image=project_img)
project_label.pack(padx=0, pady=0)

heading_label = tk.Label(root, text="CAPTION CRAFT: ADVANCED AI IMAGE CAPTIONING SYSTEM", font=("Helvetica", 14, "bold"))
heading_label.pack(padx=0, pady=0)

predefined_img = Image.open("H:/Study Related all DATA\Major Project (MCA_NEW)/Jupyter Notebook/Major Project (Caption Craft)/Main/No_Image_Available.jpg")
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





# In[ ]:




