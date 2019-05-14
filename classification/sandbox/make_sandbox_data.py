import numpy as np
import pandas as pd  
from collections import Counter      
       
data = pd.read_csv('../data/train.csv')
cntr = Counter(data["Id"])
del cntr["new_whale"]

train_images, val_images = [], []
train_labels, val_labels = [], []
for k, v in cntr.most_common(8):
    images = data.loc[data["Id"] == k, "Image"].values.tolist()
    curr_train_images, curr_val_images = images[:-10], images[-10:]
    train_images += curr_train_images
    val_images += curr_val_images
    train_labels += [k for _ in range(len(curr_train_images))]
    val_labels += [k for _ in range(len(curr_val_images))]

train_data = pd.DataFrame({'Image': train_images, 'Id': train_labels})
val_data = pd.DataFrame({'Image': val_images, 'Id': val_labels})

train_data.to_csv('train.csv', index=False, columns=['Image', 'Id'])
val_data.to_csv('val.csv', index=False, columns=['Image', 'Id'])
