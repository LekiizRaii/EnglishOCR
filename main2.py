import character_segmentation2

path = "./iam_words/words/c03/c03-016d/c03-016d-02-05.png"
returnPath = "result/char"
module = character_segmentation2.CharacterSegmentation2(path, returnPath)
module.run()
