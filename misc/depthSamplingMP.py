"""
Tutorial script for depth sampling
"""

import os
import numpy as np
import math
import time
import zlib
import cv2
import random
import multiprocessing

FETCH_BATCH_SIZE=32
HEIGHT=192
WIDTH=256

thread_1 = list(random.sample(range(0,299),4))


folderLst = list(map(str,thread_1))
datasetdir = "/home/ShapeNet/data/"
datasavedir = "/home/dataset/data.ShapeNetDepth/train/"

#index of pointer
p_start = FETCH_BATCH_SIZE*HEIGHT*WIDTH*3 # 3 channel uint8 image ends
p_end = p_start + FETCH_BATCH_SIZE*HEIGHT*WIDTH*2 # 1 channel uint16 ends

def image_save(img, id):
    
    #get the paths
    hr_path = os.path.join(datasavedir,'image_hr/%07d.png'%id)
    lr_path = os.path.join(datasavedir,'image_lr/%07d.png'%id)
    
    #save high resolution(hr) image 
    cv2.imwrite(hr_path, img)
    
    #get non-zero indices
    #transpose the index tuple matrix since random.sample() works along axis=0
    img_nnz_ids = np.transpose(np.nonzero(img))
    # print img_nnz_ids
    
    #get total nnz element no
    img_nnz_length = img_nnz_ids.shape[0]
    # print img_nnz_length
    
    #sample using random.sample()
    no_of_sample = int(round(img_nnz_length*0.75)) #this tells us how many nnz have to be zerod
    sample = random.sample(img_nnz_ids,no_of_sample)
    
    #transpose and convert to tuples
    sample = tuple(np.transpose(sample))
    
    #make the selection zero
    img[sample] = 0
    
    #save low resolution(hr) image 
    cv2.imwrite(lr_path, img)

def multiprocessing_folder(folder):
    
    print "Now in folder {}. Starting... Local Time: {}".format(folder, time.asctime(time.localtime(time.time())))
        
    start_folder_time = time.time()
    
    #datasetdir is the Dataset Folder
    #data_dir will return directories within datasetdir
    data_dir = os.path.join(datasetdir,'%s/'%(folder))
    
    f = []
    
    for _,_,filename in os.walk(data_dir):
        f.extend(filename)
    
    folder_mean = []
    folder_std = []

    for file in f:        
        path = os.path.join(data_dir,file)
        
        file_n = file[:-3]
        
        # Reading the zipped file
        binfile=zlib.decompress(open(path,'r').read())

        # Saving the binary values into corresponding arrays + PreProcessing
        depth=np.fromstring(binfile[p_start:p_end],dtype='uint16').reshape((FETCH_BATCH_SIZE,HEIGHT,WIDTH))
        
        batch_mean = np.mean(depth)
        batch_std = np.std(depth)
        
        folder_mean.append(batch_mean)
        folder_std.append(batch_std)
        
        for i in range(FETCH_BATCH_SIZE):
            
            image_id = 32000*int(folder) + 32*int(file_n)
            
            #sample and store the image
            image_save(depth[i,:,:], image_id + i) #sending each image separately
        
    
    folder_mean_avg = np.mean(folder_mean)
    folder_std_avg = np.mean(folder_std)
    
    end_folder_time = time.time()
    folder_time = end_folder_time - start_folder_time
    
    print "The folder {} took {} mins for preprocessing. ok. Local Time: {}".format(folder, folder_time/60.0, time.asctime(time.localtime(time.time())))
    print "Folder {} has average mean: {} and std: {}".format(folder, folder_mean_avg, folder_std_avg)

def main():
    
    processes = []
    
    for folder in folderLst:
        p = multiprocessing.Process(target=multiprocessing_folder, args=(folder,))
        processes.append(p)
        p.start()
    
    # The join() call ensures that subsequent lines of your code are not called before all the multiprocessing processes are completed.
    for process in processes:
        process.join()
    
    total_time = time.time() - global_start_time
    print "Total preprocessing time for {} folders is {} hrs! Adios!!".format(len(folderLst),total_time/3600.0)

if __name__ == '__main__':
    global_start_time = time.time()
    main()