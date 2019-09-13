import os
import pandas as pd

BASE_DIR = '/content/drive/My Drive/Coin_dataset_31/'
train_folder = BASE_DIR+'train/'


files_in_train = sorted(os.listdir(train_folder))


images=[i for i in files_in_train ]


df = pd.DataFrame()
df['images']=[train_folder+str(x) for x in images]



df.to_csv('files_path.csv', header=None)
