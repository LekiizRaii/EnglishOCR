from character_segmentation import character_segmentation

path = "./iam_words/words/c03/c03-016d/c03-016d-02-05.png"
returnPath = "result/char"
module = character_segmentation.CharacterSegmentation2(path, returnPath)
module.run()
