from scipy.io import loadmat
import numpy as np
import os
import skimage.morphology as morph


#batch_size = 100
SHOW_PLOTS = False


if SHOW_PLOTS:
    import matplotlib.pyplot as plt

data_dir = 'data/'
seg_filename = data_dir + 'manual_seg/manual_seg_32points_pat{}.mat'
img_filename = data_dir + 'mrimages/sol_yxzt_pat{}.mat'

input_dir = data_dir + 'input/'
if not os.path.exists(input_dir):
    os.makedirs(input_dir)
output_dir = data_dir + 'output/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
bbox_dir = data_dir + 'bbox/'
if not os.path.exists(bbox_dir):
    os.makedirs(bbox_dir)

batch_filename = 'batch{}.npy'

def save_batch(batch):
    '''
    print(i.shape)
    print(o.shape)
    print(i[0])
    print(o[0])
    '''
    np.save((input_dir + batch_filename).format(batch_count), batch[0])
    np.save((output_dir + batch_filename).format(batch_count), batch[1])
    np.save((bbox_dir + batch_filename).format(batch_count), batch[2])


def fill_poly(poly_y, poly_x, shape):
    bbox = np.zeros((4), dtype=np.int32)
    bbox[0] = np.min(poly_y)
    bbox[1] = np.min(poly_x)
    bbox[2] = np.max(poly_y)
    bbox[3] = np.max(poly_x)
    
    mask = np.zeros(shape, dtype = np.bool_)
    mask[poly_y.astype(np.int), poly_x.astype(np.int)] = True
    mask = morph.convex_hull_image(mask).astype(np.int8)
    return mask, bbox


max_bbox = [0, 0]


cur_batch = [None, None, None]
count = 0
batch_count = 0
i = 1
while True:
    seg = seg_filename.format(i)
    img = img_filename.format(i)
    i += 1
 
    print(seg)
    if not os.path.isfile(seg):
        break

    slices = loadmat(img)['sol_yxzt']
    segmentations = loadmat(seg)['manual_seg_32points']
    
    for z in range(slices.shape[2]):
        for t in range(slices.shape[3]):
            
            slice = slices[:, :, z, t]
            segmentation = segmentations[z, t]

            if segmentation.shape[0] > 1:
                segm = np.zeros((2, 33, 2))
                segm[0] = segmentation[:33, :]
                segm[0, 32] = segm[0, 0]

                segm[1] = segmentation[32:, :]
                segm[1, 0] = segm[1, -1]
                
                mask, skip = fill_poly(segm[0, :, 1], segm[0, :, 0], slice.shape[:2])
                t, bbox = fill_poly(segm[1, :, 1], segm[1, :, 0], slice.shape[:2])
                mask += t
                count += 1
                
                max_bbox = [max(max_bbox[0], bbox[2] - bbox[0]),
                            max(max_bbox[1], bbox[3] - bbox[1])]
                

                if cur_batch[0] is None:
                    cur_batch[0] = np.array([slice], dtype=np.float16)
                    cur_batch[1] = np.array([mask], dtype=np.int8)
                    cur_batch[2] = np.array([bbox], dtype=np.int32)
                else:
                    cur_batch[0] = np.append(cur_batch[0], [slice], axis=0)
                    cur_batch[1] = np.append(cur_batch[1], [mask], axis=0)
                    cur_batch[2] = np.append(cur_batch[2], [bbox], axis=0)
                '''
                if count % batch_size == 0:                    
                    save_batch(cur_batch[1], cur_batch[0])
                    cur_batch = [None, None]
                    batch_count += 1
                '''

                if SHOW_PLOTS:
                    plt.subplot(121)
                    plt.plot(segm[0, :, 0], segm[0, :, 1])
                    plt.plot(segm[1, :, 0], segm[1, :, 1])
                    plt.imshow(slice, cmap='gray')
                    
                    plt.subplot(122)
                    plt.imshow(mask, cmap='gray', interpolation='None')
                
                    mngr = plt.get_current_fig_manager()
                    # to put it into the upper left corner for example:
                    mngr.window.setGeometry(100,100, 1000, 800)

                    plt.show()

    save_batch(cur_batch)
    cur_batch = [None, None, None]
    batch_count += 1

    print('count:', count)
print('max_bbox:', max_bbox)
'''
if cur_batch[0] is not None:
    np.save(batch_filename.format(batch_count), cur_batch)
    cur_batch = None
'''
