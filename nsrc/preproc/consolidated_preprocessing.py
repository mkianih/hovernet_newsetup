import os
import numpy as np
import cv2
import scipy.io as sio

from misc.utils import rm_n_mkdir

def scale_image(path_img, scale, verbose=False, interpol = cv2.INTER_AREA):
    img = cv2.cvtColor(cv2.imread(path_img, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    shape = img.shape[:2]
    new_dim = get_new_dim(shape, scale)
    res_img = cv2.resize(img, new_dim, interpolation = interpol)
    if verbose:
        print('Before:')
        print(np.shape(img))
        plot_image(img)
        
        print('After:')
        print(np.shape(res_img))
        plot_image(res_img)
    
    return res_img
        
def scale_npy(path_npy, scale, verbose=False, interpol = cv2.INTER_NEAREST):
    npy = np.load(path_npy, allow_pickle=True)
    mask_inst = npy.item().get('inst_map')
    mask_type = npy.item().get('type_map')
    centroids = npy.item().get('inst_centroid')
    assert np.shape(mask_inst) == np.shape(mask_type), "Inconsistency in `mask_inst` and `mask_type` shapes"
    
    shape = np.shape(mask_inst)
    new_dim = get_new_dim(shape, scale)
    
    ## mask_inst and mask_type
    res_mask_inst = cv2.resize(mask_inst, new_dim, interpolation = interpol) 
    res_mask_type = cv2.resize(mask_type, new_dim, interpolation = interpol)
    
    ## centroids
    res_centroids = centroids * scale
    ### TODO: actually recalculate centroids
    
    result_npy = {'inst_map': res_mask_inst,
                  'type_map': res_mask_type,
                  'inst_type': npy.item().get('inst_type'),
                  'inst_centroid': res_centroids
                 }
    if verbose:
        print (f'Shape before: {np.shape(mask_inst)}')
        print(f'Uniques inst before: {np.unique(mask_inst)}')
        print(f'Uniques type before: {np.unique(mask_type)}')
        print (f'Centroids [0], [1]: {centroids[0], centroids[1]}')
        plot_image(mask_inst)
        
        print(f'Shape after: {np.shape(res_mask_inst)}')
        print(f'Uniques inst after: {np.unique(res_mask_inst)}')
        print(f'Uniques type after: {np.unique(res_mask_type)}')
        print (f'Centroids [0], [1]: {res_centroids[0], res_centroids[1]}')
        plot_image(res_mask_inst)
        
    return result_npy

def transform(data_path, output_masks, type_folders, result_folder, verbose=False):
    count = 0
    patients = glob(f"{data_path}/*")
    for patient_loc in tqdm(patients):
        patient_name = patient_loc[len(data_path): ]
        if verbose: print(patient_name)

        ## To make patient's name directory in the destination folder
        rm_n_mkdir(f'{output_masks}/{patient_name}')

        ## Read sub-images of each patient in the data path        
        sub_images = glob(patient_loc + '/*.svs')
        for sub_image_loc in sub_images:
            gt = 0
            sub_image_name = sub_image_loc[len(data_path)+len(patient_name)+1:-4]        

            ## To make sub_image directory under the patient's folder
            sub_image = f'{output_masks}/{patient_name}/{sub_image_name}' #Destination path
            rm_n_mkdir(sub_image)

            image_name = sub_image_loc
            img = openslide.OpenSlide(image_name)

            # If svs image needs to save in tif
            cv2.imwrite(sub_image_loc[:-4] + '.tif', np.array(img.read_region((0,0),0,img.level_dimensions[0])))      

            # Read xml file
            xml_file_name  = image_name[:-4]
            xml_file_name = xml_file_name + '.xml'
            tree = ET.parse(xml_file_name)
            root = tree.getroot()
            # Generate n-ary mask for each cell-type                         
            for k in range(len(root)):
                label = [x.attrib['Name'] for x in root[k][0]]
                label = label[0]

                for child in root[k]:
                    for x in child:
                        r = x.tag
                        if r == 'Attribute':
                            count = count+1
                            label = x.attrib['Name']
                            n_ary_mask = np.transpose(np.zeros((img.read_region((0,0),0,img.level_dimensions[0]).size))) 
                            sub_path = f'{sub_image}/{label}'
                            rm_n_mkdir(sub_path)
                            
                        if r == 'Region':
                            regions = []
                            vertices = x[1]
                            coords = np.zeros((len(vertices), 2))
                            for i, vertex in enumerate(vertices):
                                coords[i][0] = vertex.attrib['X']
                                coords[i][1] = vertex.attrib['Y']        
                            regions.append(coords)
                            poly = Polygon(regions[0])  
                            vertex_row_coords = regions[0][:,0]
                            vertex_col_coords = regions[0][:,1]
                            fill_row_coords, fill_col_coords = draw.polygon(vertex_col_coords, vertex_row_coords, n_ary_mask.shape)
                            gt = gt + 1 #Keep track of giving unique valu to each instance in an image
                            n_ary_mask[fill_row_coords, fill_col_coords] = gt
                            mask_path = f'{sub_path}/{str(count)}_mask.tif'
                            cv2.imwrite(mask_path, n_ary_mask)
                            
    ###---------------- next substep: mat files
    for patient in (patients_names):
        rm_n_mkdir(f'{result_folder}/{patient}')
        rm_n_mkdir(f'{result_folder}/{patient}/Class_maps')
        rm_n_mkdir(f'{result_folder}/{patient}/Images')
        rm_n_mkdir(f'{result_folder}/{patient}/Labels')

        patient_imgs_full = sorted(glob(f'{data_path}/{patient}/*.tif'))
        for patient_img_full in patient_imgs_full:
            patient_img = patient_img_full.split('/')[-1]
            img = tifffile.imread(f'{patient_img_full}')
            img = img[..., 0:3]
            mat_file = dict.fromkeys(type_folders)
            for typ in type_folders:
                path_to_masks = f'{output_masks}/{patient}/{patient_img.split(".tif")[0]}/{typ}/*.tif'
                if len(glob(path_to_masks)) > 0:
                    mask_path = glob(path_to_masks)[0]
                    mask = tifffile.imread(f'{mask_path}')
                else:
                    mask = np.zeros((np.shape(img)[0], np.shape(img)[1]))
                mat_file[typ] = mask
            sio.savemat(f'{result_folder}/{patient}/Class_maps/{patient_img.split(".tif")[0]}.mat', mat_file)
            cv2.imwrite(f'{result_folder}/{patient}/Images/{patient_img.split(".tif")[0]}.png', img)
            
    ###---------------- next substep: transform to npy
    patients = glob(f'{result_folder}/*')
    for patient in (patients):
        for img_path in glob(f'{patient}/Images/*.png'):
            img_png = imread(f'{img_path}')
            inst_types_npy = sio.loadmat(f'{patient}/Class_maps/{os.path.basename(img_path).split(".png")[0]}.mat')
            iter_to_transform = inst_types_npy.copy()
            for key in iter_to_transform.keys():
                if str(key).startswith("__"):
                    inst_types_npy.pop(key, None)
            mapping = {k: v for v,k in enumerate(inst_types_npy.keys(), 1)} # no 'Background': 0
            # {'Epithelial': 1, 'Lymphocyte': 2, 'Macrophage': 3, 'Neutrophil': 4}
            ## inst_map and type_map
            inst_map = np.zeros((np.shape(img_png)[0], np.shape(img_png)[1]))
            type_map = np.zeros((np.shape(img_png)[0], np.shape(img_png)[1]))

            for k,v in mapping.items(): # Epithelial: 1
                uniques = np.unique(inst_types_npy[k])[1:] # exclude 0
                for val in (uniques):
                    inst_map[inst_types_npy[k] == val] = val
                    type_map[inst_types_npy[k] == val] = v
            
            for i, inst in enumerate(np.unique(inst_map)):
                inst_map[inst_map == inst] = i
                    
            inst_type = [type_map[np.where(inst_map == x)[0][0], np.where(inst_map == x)[1][0]] 
                         for x in list(np.unique(inst_map))[1:]]

            inst_centroid_list = []
            inst_id_list = list(np.unique(inst_map))
            for inst_id in inst_id_list[1:]:  # avoid 0 i.e background
                mask = np.array(inst_map == inst_id, np.uint8)
                inst_moment = cv2.moments(mask)
                inst_centroid = [
                    (inst_moment["m10"] / inst_moment["m00"]),
                    (inst_moment["m01"] / inst_moment["m00"]),
                ]
                inst_centroid_list.append(inst_centroid)

            result_dict = {
                    "inst_map": inst_map.astype(np.int16), # conv uint16?
                    "type_map": type_map.astype(np.int8), # conv uint8?
                    "inst_type": (np.array([inst_type]).T).astype(np.int8), # conv uint8?
                    "inst_centroid": np.array(inst_centroid_list),
                }
            np.save(
                    f'{patient}/Labels/{os.path.basename(img_path).split(".png")[0]}.npy',
                    result_dict,
                )

def compose_dataset(monusac_data, crit, scaling, verbose=False):

    label_id = f'c{crit}{"_scaling" if scaling else "_cut"}'
    data_folder = f'{monusac_data}/data_monusac'
    output_folder = f'{monusac_data}/data_hv_monusac{label_id}'
    rm_n_mkdir(output_folder)

    for mode in ['train', 'valid', 'test']:
        rm_n_mkdir(f'{output_folder}/{mode}/Labels')
        rm_n_mkdir(f'{output_folder}/{mode}/Images')

    for cmode in ['train', 'test']:
        patches = []
        to_transform = []

        count = 0
        # filter out outliers
        for pname_img in glob(f"{data_folder}/{cmode}/*/Images/*.png"):
            img_size = np.shape(imread(pname_img))
            pname = os.path.basename(pname_img).split('.png')[0]

            if (img_size[0]) >= crit and (img_size[1] >= crit):
                patches.append(pname)
            else:
                count += 1
                if scaling: 
                    to_transform.append(pname) # comment if transformation should be avoided
        if verbose:
            print ()            
            if scaling:        
                print (f"Skipped for {cmode} for further transform: {len(to_transform)}")
            else:
                print (f"Cut images for {cmode}: {count} ")

        def copy_patches(mode, patches_list):
            for patch in patches_list:
                img = glob(f'{data_folder}/{cmode}/*/Images/{patch}.png')
                label = glob(f'{data_folder}/{cmode}/*/Labels/{patch}.npy')
                if patch not in to_transform:
                    shutil.copy(img[0], f'{output_folder}/{mode}/Images/{patch}.png')
                    shutil.copy(label[0], f'{output_folder}/{mode}/Labels/{patch}.npy')
                elif patch in to_transform:
                    scale = get_scale(np.shape(plt.imread(img[0])), crit)

                    image = scale_image(img[0], scale)
                    npy = scale_npy(label[0], scale)
                    
                    if verbose: print (f"Shape: {np.shape(plt.imread(img[0]))}, scale: {scale}, crit: {crit}. Result: {np.shape(image)} ")

                    cv2.imwrite(f'{output_folder}/{mode}/Images/{patch}_s.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    np.save(f'{output_folder}/{mode}/Labels/{patch}_s.npy', npy)

        full_patches = [*patches, *to_transform]
        if verbose: print (f"For mode {cmode}: {len(patches)} + {len(to_transform)} = {len(full_patches)}")

        if cmode == 'train':
            train, valid = train_test_split(full_patches, test_size=0.2, random_state=42, shuffle=True)
            if verbose: print (f"Train + valid: {len(train)} + {len(valid)} = {len(train)+len(valid)}")
            copy_patches('train', train)    
            copy_patches('valid', valid)
        elif cmode == 'test':
            copy_patches('test', full_patches)
            
    if verbose:
        print ()
        for cmode in ['train', 'valid', 'test']:
            path = glob(f'{output_folder}/{cmode}/Labels/*')
            print (f'{cmode} - {len(path)}')
    
    
if __name__ == "__main__":

    ### Unzip MoNuSAC.zip and 2 zip archives inside: MoNuSAC_images_and_annotations.zip and MoNuSAC Testing Data and Annotations.zip
    monusac_data = "/data/input/data_monusac/MoNuSAC/"

    for mode in ['train', 'test']:
        result_folder = f'{monusac_data}/data_monusac/{mode}'
        if mode == 'train':
            data_path = f'{monusac_data}/MoNuSAC_images_and_annotations'
            output_masks = f'{monusac_data}/MoNuSAC_images_and_annotations_masks'
            type_folders = ['Epithelial', 'Lymphocyte', 'Macrophage', 'Neutrophil']
        elif mode == 'test':
            data_path = f'{monusac_data}/MoNuSAC Testing Data and Annotations'
            output_masks = f'{monusac_data}/MoNuSAC Testing Data and Annotations_masks'
            type_folders = ['Epithelial', 'Lymphocyte', 'Macrophage', 'Neutrophil', 'Ambiguous']
        patients_full_path = glob(f'{data_path}/*')
        patients_names = [x.split('/')[-1] for x in patients_full_path]
        
        ### Save svs as tif and generate masks from xml file annotations
        ### Save tif as png and generate .mat files
        ### Save hover numpy from mat
        transform(data_path, output_masks, type_folders, result_folder, verbose=False)
        print (f'Finished for {mode}')

    crit = 164
    scaling = False
    compose_dataset(monusac_data, crit, scaling, verbose=True)



# Consolidated PanNuke preprocessing
import os
import numpy as np
import cv2
import scipy.io as sio


def transform(input_dir, output_dir, folds, use_opencv=True, fix_masks=True):
    for fold in folds:
        data = np.load(
            os.path.join(input_dir, fold, "Arrays", "images.npy")
        )  # f'{input_dir}/{fold}/Arrays/images.npy'
        masks = np.load(os.path.join(input_dir, fold, "Masks", "masks.npy"))
        count_img = np.shape(data)[0]
        assert count_img == np.shape(masks)[0]

        for img in range(count_img):
            if use_opencv:
                imageCV = cv2.cvtColor(np.float32(data[img]), cv2.COLOR_BGR2RGB)
                cv2.imwrite(
                    os.path.join(output_dir, fold, "Images", f"{fold}_{img}.png"),
                    imageCV,
                )  # f'{output_dir}/{fold}/Images/{fold}_{img}.png', imageCV
            if fix_masks:
                # switch 0 and 5 types
                for xpix in range(np.shape(masks[img])[0]):
                    for ypix in range(np.shape(masks[img])[1]):
                        masks[img][xpix][ypix][0], masks[img][xpix][ypix][5] = (
                            masks[img][xpix][ypix][5],
                            masks[img][xpix][ypix][0],
                        )
                # fix 6 channel types to instances, types
                result = np.zeros((np.shape(masks[img])[0], np.shape(masks[img])[1], 2))
                for i in range(
                    1, np.shape(masks[img])[2]
                ):  # skip filling background with 1's
                    # instance info
                    type_channel_pannuke = masks[img][:, :, i]
                    for val in np.unique(type_channel_pannuke):
                        if val != 0.0:
                            result[:, :, 0][np.where(type_channel_pannuke == val)] = val
                    # type info
                    result[:, :, 1][np.where(type_channel_pannuke > 0)] = i

            # remap labels
            for num, val in enumerate(np.unique(result[:, :, 0])):
                result[:, :, 0][result[:, :, 0] == val] = float(num)

            # create inst type
            inst_type = []
            for i in np.unique(np.array(result[:, :, 0]))[1:]:
                fpos = list(zip(*np.where(result[:, :, 0] == int(i))))[0]
                inst_type.append(result[:, :, 1][fpos[0]][fpos[1]])

            # compute instance centroids

            inst_centroid_list = []
            inst_map = result[:, :, 0]
            inst_id_list = list(np.unique(inst_map))
            for inst_id in inst_id_list[1:]:  # avoid 0 i.e background
                mask = np.array(inst_map == inst_id, np.uint8)
                inst_moment = cv2.moments(mask)
                inst_centroid = [
                    (inst_moment["m10"] / inst_moment["m00"]),
                    (inst_moment["m01"] / inst_moment["m00"]),
                ]
                inst_centroid_list.append(inst_centroid)

            #'inst_map', 'type_map', 'inst_type', 'inst_centroid'
            result_dict = {
                "inst_map": result[:, :, 0],
                "type_map": result[:, :, 1],
                "inst_type": np.array([inst_type]).T,
                "inst_centroid": np.array(inst_centroid_list),
            }
            np.save(
                os.path.join(output_dir, fold, "Labels", f"{fold}_{img}.npy"),
                result_dict,
            )
            print(f"{fold}_{img}.npy saved")


if __name__ == "__main__":
    folds_to_use = ["test", "valid", "train"]
    transform("/data/input/data_pannuke/", "/data/input/data_hv_pannuke/", folds_to_use)


# Consolidated CoNSeP preprocessing
import os
import scipy.io as sio
import glob
import numpy as np


def transform(input_dir, output_dir, modes):
    for mode in modes:
        file_list = glob.glob(
            os.path.join(input_dir, mode, "Labels", "*{}".format(".mat"))
        )
        file_list.sort()
        assert len(file_list) != 0
        for file_mat in file_list:
            filename = os.path.basename(file_mat)
            basename = filename.split(".")[0]
            to_transform = sio.loadmat(file_mat)
            iter_to_transform = to_transform.copy()
            for key in iter_to_transform.keys():
                if str(key).startswith("__"):
                    to_transform.pop(key, None)

            ## * for converting the GT types in CoNSeP (see paper for explanations)
            for key in ["type_map", "inst_type"]:
                to_transform[key][
                    (to_transform[key] == 3) | (to_transform[key] == 4)
                ] = 3
                to_transform[key][
                    (to_transform[key] == 5)
                    | (to_transform[key] == 6)
                    | (to_transform[key] == 7)
                ] = 4
                assert np.max(to_transform[key]) <= 4

            # save only maps
            # result = np.dstack((to_transform['inst_map'], to_transform['type_map']))
            np.save(
                f"{os.path.join(output_dir, mode.lower(), 'Labels', basename)}.npy",
                to_transform,
            )
            print(
                f"{os.path.join(output_dir, mode.lower(), 'Labels', basename)}.npy saved."
            )


if __name__ == "__main__":
    transform(
        "/data/input/data_consep/CoNSeP/",
        "/data/input/data_hv_consep/",
        ["Train", "Test"],
    )
