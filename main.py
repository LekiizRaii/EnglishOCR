from character_segmentation import CharacterSegmentation
from utils import *

path = "../iam_words/words/r06/r06-126/r06-126-09-04.png"
# path = "../iam_words/words/a01/a01-000u/a01-000u-01-02.png"
module = CharacterSegmentation(path)
module.run()
