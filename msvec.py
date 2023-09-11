import pickle
import os
import subprocess 
import cv2 
from tqdm import tqdm
import numpy as np


def run_v2e(input,output):
    #create the output directory if not exist
    os.makedirs(output,exist_ok=True)

    with open(input,'rb') as f:
        file = pickle.load(f)
        images = file['images']
        timestamps = file['timestamps']
    # output images to folder
    frames_dir = output+'/imgs'
    os.makedirs(frames_dir,exist_ok=True)

    for index,image in enumerate(images):
        image = np.pad(image,((14,14),(3,3)),'constant', constant_values=0))
        cv2.imwrite(os.path.join(frames_dir,str(index).zfill(6)+'.png'),image)
    
    #determine the fps
    timestamp_diff = timestamps[1:]-timestamps[:-1]
    fps = 1e6/timestamp_diff.mean()
    #print(output+'/original/seq/fps.txt')
    # write fps to file
    with open(output+'/fps.txt','w+') as f:
        f.write(str(fps))
    #'python upsampling/upsample.py --input_dir=example/original --output_dir=example/upsampled'
    
    '''
    python esim_torch/generate_events.py --input_dir=example/upsampled \
                                     --output_dir=example/events \
                                     --contrast_threshold_neg=0.2 \
                                     --contrast_threshold_pos=0.2 \
                                     --refractory_period_ns=0
                                     '''
    


if __name__ == '__main__':
    if not os.path.exists('/tmp/esim'):
        os.mkdir('/tmp/esim')
    files = pickle.load(open('/tsukimi/datasets/MVSEC/data_paths.pkl','rb'))['test']
    for file in tqdm(files):
        run_v2e('/tsukimi/datasets/MVSEC/event_chunks_processed/'+file,'/tmp/esim/original/'+file)
        
    
    # cmd = ['python','upsampling/upsample.py','--input_dir='+output+'/original','--output_dir='+output+'/upsampled']
    # subprocess.run(cmd)

    # cmd = ['python','esim_torch/scripts/generate_events.py','--input_dir='+output+'/upsampled','--output_dir='+output+'/events','--contrast_threshold_neg=0.2','--contrast_threshold_pos=0.2','--refractory_period_ns=0']
    # subprocess.run(cmd)
