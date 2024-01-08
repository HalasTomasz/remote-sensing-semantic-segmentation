import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.wkt import loads as wkt_loads
import os
import rasterio as rio
import cv2

color_dict = {
    1: [0, 128, 0], # Green
    2: [255, 0, 0], # Red
    3: [0, 0, 255], # Blue
    4: [255, 255, 0], # Yellow
    5: [255, 192, 203], # Pink
    6: [128, 0, 128], # Purple
    7: [255, 165, 0], # Orange
    8: [0, 255, 0], # Lime Green
    9: [0, 0, 0], # Black
    10: [255, 255, 255] # White
}



def _convert_coordinates_to_raster(coords, img_size, xymax):
    Xmax,Ymax = xymax
    H,W = img_size
    W1 = 1.0*W*W/(W+1)
    H1 = 1.0*H*H/(H+1)
    xf = W1/Xmax
    yf = H1/Ymax
    coords[:,1] *= yf
    coords[:,0] *= xf
    coords_int = np.round(coords).astype(np.int32)
    return coords_int


def _get_xmax_ymin(grid_sizes_panda, imageId):
    xmax, ymin = grid_sizes_panda[grid_sizes_panda.ImageId == imageId].iloc[0,1:].astype(float)
    return (xmax,ymin)


def _get_polygon_list(wkt_list_pandas, imageId, cType):
    df_image = wkt_list_pandas[wkt_list_pandas.ImageId == imageId]
    multipoly_def = df_image[df_image.ClassType == cType].MultipolygonWKT
    polygonList = None
    if len(multipoly_def) > 0:
        assert len(multipoly_def) == 1
        polygonList = wkt_loads(multipoly_def.values[0])
    return polygonList


def _get_and_convert_contours(polygonList, raster_img_size, xymax):
    perim_list = []
    interior_list = []
    if polygonList is None:
        return None
    
    for poly in polygonList.geoms:
        perim = np.array(list(poly.exterior.coords))
        perim_c = _convert_coordinates_to_raster(perim, raster_img_size, xymax)
        perim_list.append(perim_c)
        for pi in poly.interiors:
            interior = np.array(list(pi.coords))
            interior_c = _convert_coordinates_to_raster(interior, raster_img_size, xymax)
            interior_list.append(interior_c)
    return perim_list,interior_list


def _plot_mask_from_contours(raster_img_size, contours, class_value):
    img_mask = np.zeros((raster_img_size[0],raster_img_size[1],3),np.uint8)
    if contours is None:
        return img_mask
    print(class_value)
    perim_list,interior_list = contours
    cv2.fillPoly(img_mask,perim_list, color_dict[class_value] )
    cv2.fillPoly(img_mask,interior_list,0)
    return img_mask


def generate_mask_for_image_and_class(raster_size, imageId, class_type, grid_sizes_panda,
                                     wkt_list_pandas, color):
    xymax = _get_xmax_ymin(grid_sizes_panda,imageId)
    polygon_list = _get_polygon_list(wkt_list_pandas,imageId,class_type)
    contours = _get_and_convert_contours(polygon_list,raster_size,xymax)
    mask = _plot_mask_from_contours(raster_size,contours,color)
    return mask


def create_mask_for_every_image(inDir, filenames):
    print(inDir)
    df = pd.read_csv(inDir + '/train_wkt_v4.csv')
    gs = pd.read_csv(inDir + '/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
    #img_filename = "./Data/three_band/6120_2_2.tif"
    print(filenames)
    for img_filename in filenames:
        dataset = rio.open(inDir+img_filename)
        mask_list = []
        for category in range(1,10):
        #print('Size is ',dataset.RasterXSize,'x',dataset.RasterYSize,'x',dataset.RasterCount)
            img_dirname = os.path.splitext(os.path.basename(img_filename))[0]
            mask_list.append(generate_mask_for_image_and_class((dataset.RasterYSize,dataset.RasterXSize ),img_dirname,
                                                   category,gs,df, category))
       
     
        dir_path = os.path.join(inDir,"Masks", img_dirname)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        mask = np.sum(mask_list, axis=0).astype(np.uint8)
        print(mask.shape)
        plt.imshow(mask)
        plt.show()
       # output_directory = os.path.join(dir_path, f"mask_{str(img_dirname)}_final.tif")
        #cv2.imwrite(output_directory,mask)


####
inDir = '/home/halas/Inzynierka/Data'
train_images = pd.read_csv('/home/halas/Inzynierka/Data/train_wkt_v4.csv')
train_images_list = train_images['ImageId'].unique().tolist()
# abc = os.path.join(inDir,"three_band")
file_names = [os.path.join(inDir,str("/Train/"+image_id+".tif"))  for image_id in train_images_list]
create_mask_for_every_image(inDir,file_names)