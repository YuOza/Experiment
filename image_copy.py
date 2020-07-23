import pathlib
from PIL import Image

dataset = "bedroom"  #bedroom, Celeb
img_path = "./"+dataset
path = pathlib.Path(img_path)
files = list(path.glob('**/*.jpg')) + list(path.glob('**/*.png'))
images = []
i = 0
for fn in files:
    Image.open(fn).save("./learn/forFIRTrue/"+dataset+"/"+str(i)+".png")
    i += 1
    if i > 10000:
        break
