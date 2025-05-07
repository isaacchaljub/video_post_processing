import pandas as pd
from tkinter import Tk
import tkinter.filedialog
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import joblib
import os
import numpy as np

#Read the CSV file
# file1=tkinter.filedialog.askopenfilename(title="Select the output_stem CSV file", filetypes=[("CSV files", "*.csv")])
# data=pd.read_csv(file1)

# file2=tkinter.filedialog.askopenfilename(title="Select the output_config CSV file", filetypes=[("CSV files", "*.csv")])
# config=pd.read_csv(file2)

data=pd.read_csv('A04 FILA 01B_output_stem_trackable_objects.csv')
config=pd.read_csv('A04 FILA 01B_output_config.csv')

#Get the velocity for each object and observation
def compute_vel(group):
    # Compute velocity as the difference in centroids_x over difference in frames
    vel = group['centroids_x'].diff() / group['frames'].diff()
    # Replace the first observation (NaN) with the group's mean velocity
    if not vel.empty:
        vel.iloc[0] = vel.mean()  # .mean() ignores NaN by default
    return vel

def assign_velocities(df):
    # Use groupby to apply compute_vel for each object_id
    df['velocity'] = df.groupby('object_id', group_keys=False)[['centroids_x','frames']].apply(compute_vel)
    return df

#Get the velocity for each object and observation
assign_velocities(data)

#Correct in case the video goes from left to right
if data['velocity'].mean()<0:
        data['velocity']=-data['velocity']

#Get the frames persisted and the first frame for each object
frame_counts = data.groupby('object_id')['frames'].apply(lambda x: x.max() - x.min())
frame_info=data.groupby('object_id')['frames'].min()

#Assign the values to the data frame
data['frames']=[frame_counts.get(data.loc[i,'object_id']) for i in range(len(data))]
data['first_frame']=[frame_info.get(data.loc[i,'object_id']) for i in range(len(data))]

#Get mean velocity and std area for each object, and adjust std area by frames
data['mean_velocity_fps']=[data.groupby('object_id')['velocity'].mean().get(data.loc[i,'object_id']) for i in range(len(data))]
data['std_area']=[data.groupby('object_id')['area'].std().get(data.loc[i,'object_id']) for i in range(len(data))]
data['adj_std_area']=data['std_area']/data['frames']

#Work only with stems that were counted in the original detection
positive_stems=data[data['counted']==1].reset_index(drop=True)
positive_stems.drop(columns=['counted'], inplace=True)

#Check if there are the same number of stems in data and config, if not, reassign the stem with the lowest count to the next one in config
while len(positive_stems.groupby('object_id').count()) < len(config.groupby('stem').count()):
    config['stem']=[config['stem'][i]+1 if config['stem'][i]==config.groupby('stem').size().idxmin() else config['stem'][i] for i in range(len(config))]
    #Adjust the stem number to the new amount using factorization
    config['stem']=pd.factorize(config['stem'])[0]+1

#Get the first frame for each object in positive_stems, then the distance between the first frame of each object and the first frame of the previous object
ffp=pd.DataFrame(positive_stems.groupby('object_id')['first_frame'].mean())
ffp['dist']=[ffp.iloc[i,0] if i==0 else ffp.iloc[i,0]-ffp.iloc[i-1,0] for i in range(len(ffp))]
positive_stems['dist_between_apps']=[ffp['dist'].get(positive_stems.loc[i,'object_id']) for i in range(len(positive_stems))]

#Drop the first frame column, as it is not needed anymore
positive_stems.drop('first_frame',axis=1,inplace=True)

#Get the adjusted frames as the frames tracked times the normalized speed
normalized_speed=MinMaxScaler().fit_transform(pd.DataFrame(positive_stems['mean_velocity_fps']))
positive_stems['adjusted_frames']=[positive_stems['frames'][i]*normalized_speed[i][0] for i in range(len(normalized_speed))]

#Use robust scaler for data to be able to use the model trained with the same scaler
scaled_data=RobustScaler().fit_transform(positive_stems.drop(['object_id'],axis=1))

#Get the columns that were scaled
scaled_cols=positive_stems.drop(['object_id'],axis=1).columns

#Get the scaled data for the positive stems
scaled_positive_stems=positive_stems.copy()
scaled_positive_stems.set_index('object_id', inplace=True)
scaled_positive_stems[scaled_cols]=scaled_data

#Import the model and predict the anomalies
model=joblib.load('random_forest_model.pkl')
predictions=model.predict(scaled_positive_stems)

#Add the predictions to the positive stems data frame
scaled_positive_stems['predictions']=predictions

#Get the average predictions for each stem, to be able to classify the stem as an anomaly or not
positive_stems['predictions']=[1 if scaled_positive_stems.groupby('object_id')['predictions'].mean().get(positive_stems.loc[i,'object_id'])>0.5 else 0 for i in range(len(positive_stems))]

#Use factorization to get the stem number for each object
positive_stems['stem']=pd.factorize(positive_stems['object_id'])[0] + 1

#Create a dictionary with the stem number and the predictions
predictions_dict = {positive_stems['stem'][i]: positive_stems['predictions'][i] for i in range(len(positive_stems))}

#Add the predictions to the config data frame
config['pred']=[predictions_dict.get(stem) for stem in config['stem']]

#Now, assign the values of stems that are negative to the next stem number unless it's the last one, in which case assign it to the previous one
for i in range(len(config)):
    if config['pred'][i]==0:
        if config['stem'][i]!=config['stem'].max():
            config.at[i,'stem']=config['stem'][i]+1
        else:
            config.at[i,'stem']=config['stem'].max()-1

#Drop the pred column from the config data frame
config.drop('pred',axis=1,inplace=True)

#Factorize config one last time to get the stem number for each object
config['stem']=pd.factorize(config['stem'])[0] + 1

#Fix the product mismatch that occurs when reassigning the stem number
#1) get the unique product values for each stem
tmp = config[['stem', 'product']].drop_duplicates()
# 2) sum the unique product values that now belong to the same stem
stem_product_sum = tmp.groupby('stem')['product'].sum()
# 3) give every row in that stem the same summed value
config['product'] = config['stem'].map(stem_product_sum)

#Save the config file to a new CSV file
# new_file_name=os.path.splitext(file2)[0] + '_filtered.csv'
new_file_name='A04 FILA 01B_output_config_filtered.csv'
config.to_csv(new_file_name, index=False)
print(f"Filtered config file saved as {new_file_name}")