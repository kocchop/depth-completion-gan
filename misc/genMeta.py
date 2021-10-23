"""
Script to generate the meta_info file
"""

import os
import time

def main():
    
    f = []
    for (dirpath, dirnames, filenames) in os.walk("./image_lr"):
        f.extend(filenames)
    
    f = sorted(f)
    
    with open("meta_info.txt", 'w') as metafile:
        for image in f:
            metafile.write("%s (192, 256, 1)\n"%image)
    
    print("Total time taken is {} seconds".format(time.time() - global_start_time))

if __name__=='__main__':
    print("Started processing the folder. ok...")
    global_start_time  = time.time()
    main()
