import numpy as np
def generate_mask_path():
    a=np.random.randint(4)+1
    mask_path = ("masks/mask{:d}.mat".format(a))
    return mask_path
#mask_path=generate_mask_path()
#print(mask_path)
