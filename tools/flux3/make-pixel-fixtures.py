#!/usr/bin/env python3
"""Generate tiny CPU-only Pillow/mflux reference fixtures; never imports MLX.
Run with the existing mflux venv and pass its flux3_reference_images.py path.
"""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageOps

reference = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('flux3_reference_geometry', reference)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def pixels(w, h):
    return np.array([((x * 17 + y * 31 + c * 73) % 256) for y in range(h) for x in range(w) for c in range(3)], dtype=np.uint8).reshape(h, w, 3)

rows = []
for w,h,ow,oh in [(13,11,7,5),(13,11,19,17),(13,11,13,6),(13,11,8,11),(13,11,13,11)]:
    original = pixels(w,h)
    result = np.array(Image.fromarray(original).resize((ow,oh), Image.Resampling.LANCZOS))
    rows.append(dict(kind='resize', width=w,height=h,outWidth=ow,outHeight=oh,input=original.reshape(-1).tolist(),output=result.reshape(-1).tolist()))
for w,h in [(30,21),(13,11),(1,1)]:
    original = pixels(w,h)
    result, box = module.Flux3ReferenceImages.prepare_reference(Image.fromarray(original))
    rows.append(dict(kind='single',width=w,height=h,outWidth=result.width,outHeight=result.height,input=original.reshape(-1).tolist(),output=np.array(result).reshape(-1).tolist()))
original = pixels(35,29)
result = module.Flux3ReferenceImages.prepare_multi_reference(Image.fromarray(original))
rows.append(dict(kind='multiple',width=35,height=29,outWidth=result.width,outHeight=result.height,input=original.reshape(-1).tolist(),output=np.array(result).reshape(-1).tolist()))
for orientation in range(1,9):
    original=pixels(3,2)
    image=Image.fromarray(original); image.getexif()[274]=orientation
    result=ImageOps.exif_transpose(image)
    rows.append(dict(kind=f'orientation{orientation}',width=3,height=2,outWidth=result.width,outHeight=result.height,input=original.reshape(-1).tolist(),output=np.array(result).reshape(-1).tolist()))
out=Path(__file__).resolve().parents[2]/'Tests/vMLXFluxTests/Resources/Flux3/pixel-parity.json'
out.write_text(json.dumps(rows,separators=(',',':'))+'\n')
print(f'{len(rows)} CPU Pillow/mflux pixel fixtures -> {out}')
read_rows=[]
for mode in ['RGB','RGBA','L']:
    original=pixels(3,2)
    if mode=='RGBA':
        alpha=np.array([0,64,128,255,200,255],dtype=np.uint8).reshape(2,3,1)
        original=np.concatenate([original,alpha],axis=-1)
    elif mode=='L':
        original=original[:,:,0]
    image=Image.fromarray(original)
    path=out.parent/f'read-{mode.lower()}.png'
    image.save(path)
    read_rows.append(dict(file=path.name,width=image.width,height=image.height,output=np.array(image.convert('RGB')).reshape(-1).tolist()))
image=Image.fromarray(pixels(3,2)); image.getexif()[274]=6
path=out.parent/'read-exif6.png'; image.save(path,exif=image.getexif())
transposed=ImageOps.exif_transpose(image)
read_rows.append(dict(file=path.name,width=transposed.width,height=transposed.height,output=np.array(transposed.convert('RGB')).reshape(-1).tolist()))
(out.parent/'image-read-parity.json').write_text(json.dumps(read_rows,separators=(',',':'))+'\n')
