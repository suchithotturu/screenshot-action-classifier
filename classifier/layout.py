"""22 spatial features derived only from automatic OCR line boxes."""
import numpy as np

NAMES = ([f"text_grid_{y}_{x}" for y in range(3) for x in range(3)] +
         ["text_left","text_center","text_right","box_width_mean","box_width_std",
          "box_height_mean","box_height_std","center_y_mean","center_y_std",
          "vertical_span","line_gap_mean","line_gap_std","ocr_confidence_mean"])


def spatial_features(row):
    boxes,confidence = [],[]
    for block in row.get("ocr",{}).get("blocks",[]):
        box = block.get("box",[])
        if len(box)!=4 or not np.isfinite(box).all():
            continue
        x,y,w,h = np.clip(np.asarray(box,dtype=float),0,1)
        if w<=0 or h<=0:
            continue
        boxes.append([x,y,min(w,1-x),min(h,1-y)])
        confidence.append(float(np.clip(block.get("confidence",0),0,1)))
    if not boxes:
        return np.zeros(len(NAMES),dtype=float)
    a=np.asarray(boxes)
    centers=np.clip(a[:,:2]+a[:,2:]/2,0,1)
    grid=np.zeros((3,3))
    for x,y in centers:
        grid[min(int(y*3),2),min(int(x*3),2)]+=1
    align=np.bincount(np.minimum((centers[:,0]*3).astype(int),2),minlength=3)/len(a)
    gaps=np.diff(np.sort(centers[:,1]))
    rest=[a[:,2].mean(),a[:,2].std(),a[:,3].mean(),a[:,3].std(),
          centers[:,1].mean(),centers[:,1].std(),
          np.max(a[:,1]+a[:,3])-np.min(a[:,1]),
          gaps.mean() if len(gaps) else 0,gaps.std() if len(gaps) else 0,
          np.mean(confidence)]
    return np.concatenate([grid.ravel()/len(a),align,rest])
