import numpy as np


from os import path, mkdir, listdir, fsync



#### Concat all beta files
all_files = listdir('./')
all_res = []
for file in all_files:
    print(file)
    if 'beta' in file:
        try:
            all_res.append(np.load(file))
        except:
            print("File \"{}\" could not be loaded due to some reason".format(file))
            continue

all_stacked = np.vstack(all_res)
np.save('beta_all', all_stacked)
print(all_stacked.shape)



#### Concat all eval files
all_files = listdir('../eval/')
all_res = []
for file in all_files:
    if 'eval' in file:
        print(file)
        all_res.append(np.load(path.join('../eval/', file)))

all_stacked = np.hstack(all_res)
np.save('../eval/eval_all', all_stacked)
print(all_stacked.shape)