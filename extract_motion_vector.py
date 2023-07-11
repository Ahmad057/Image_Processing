import pandas as pd
import json

df = pd.read_csv("output.csv", index_col=None)


def get_frame_object(filename, frame_no):
    frame_no = f"{frame_no}"
    with open(filename) as file:
        json_string = file.read()
        json_object = json.loads(json_string)
        json_object = json.loads(json_object)


    return [[obj['x'], obj['y'], obj['w'], obj['h']]
               for frame_data in json_object["meta-data"].values()
               for obj in frame_data['objects']]


def getMV(frame_no):
    return df[df['frame_no']==frame_no]

def extractMV(frame_no, coordinates_list):
    YoloObjs = []
    mvobj = getMV(frame_no)

    # YoloObjs = get_frame_object("meta_data.json", frame_no)
    YoloObjs = coordinates_list
    lenght_yoloObj = len(YoloObjs)
    MV = [[] for _ in range(lenght_yoloObj)]
    jj = 0
    for k in range(len(mvobj)):
        j = 0
        while j < len(YoloObjs) >= 0:
            # (mvobj.delta_x[k]) != 0) and (mvobj.delta_y[k] != 0) and)
            if mvobj.src_x.iloc[k] >= YoloObjs[j][0] and (mvobj.src_y.iloc[k] >= YoloObjs[j][1]) and (mvobj.src_x.iloc[k] + mvobj.w.iloc[k] <= YoloObjs[j][2]) and (mvobj.src_y.iloc[k] + mvobj.h.iloc[k] <= YoloObjs[j][3]):
                MV[j+jj].append(mvobj.values[k])
    
            elif (mvobj.src_x.iloc[k] >= YoloObjs[j][2]) and (mvobj.src_y.iloc[k] >= YoloObjs[j][3]):
                YoloObjs.pop(j)
                jj += 1
                j -= 1

            
            j += 1
                  
        if YoloObjs == []: 
            return MV
        

def get_motion_vector(frame_no, coordinates_list):
    mv_coordinates = []
    motion_file_coor = []

    _df = df[df['frame_no'] == frame_no]
    # print("frame-no: ", frame_no)

    if not _df.empty:
        for coordinates in coordinates_list:
            x1, y1 = coordinates[0], coordinates[1]
            x2, y2 = coordinates[2], coordinates[3]

            if x1 >= 0 and y1 >= 0:
                mv_df =  _df[(_df["src_x"]+_df["w"]>= x1) & (_df['src_x']<= x2) & (_df['src_y']+_df["h"]>= y1) & (_df['src_y']<= y2)]
                if not mv_df.empty:
                    _x1, _y1 = (x1 - int(mv_df.delta_x.iloc[0]/mv_df.delta_scale.iloc[0])), (y1 - int(mv_df.delta_y.iloc[0]/mv_df.delta_scale.iloc[0]))
                    _x2, _y2 = (x2 - int(mv_df.delta_x.iloc[-1]/mv_df.delta_scale.iloc[-1])), (y2 - int(mv_df.delta_y.iloc[-1]/mv_df.delta_scale.iloc[-1]))
                    mv_coordinates.append([_x1, _y1, _x2, _y2])
                    # print("Actual Coordinate: ",[x1, y1, x2, y2], [_x1, _y1, _x2, _y2], [x1, y1, x2-x1, y2-y1], [_x1, _y1, _x2-_x1, _y2-_y1])
                    # print("Process: ",  x1-_x1, y1-_y1, x2-_x2, y2-_y2)
                    # motion_file_coor.append([mv_df["src_x"].iloc[0], mv_df['src_y'].iloc[0], mv_df["src_x"].iloc[-1], mv_df['src_y'].iloc[-1], mv_df.delta_x.iloc[0], mv_df.delta_y.iloc[0], mv_df.delta_x.iloc[-1], mv_df.delta_y.iloc[-1], mv_df.delta_scale.iloc[0], mv_df.delta_scale.iloc[-1], x1-_x1, y1-_y1, x2-_x2, y2-_y2]) 
                    motion_file_coor.append([x1, y1, x2, y2, mv_df["src_x"].iloc[0], mv_df['src_y'].iloc[0], mv_df["src_x"].iloc[-1], mv_df['src_y'].iloc[-1], mv_df.delta_x.iloc[0], mv_df.delta_y.iloc[0], mv_df.delta_x.iloc[-1], mv_df.delta_y.iloc[-1], (x1-_x1), (y1-_y1), (x2-_x2), (y2-_y2)])
    
    return mv_coordinates, motion_file_coor
        