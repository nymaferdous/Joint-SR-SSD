import glob
import os

annotpath = '/home/nyma/PycharmProjects/JointSSD/pascal-voc/trainval/VOCdevkit/VOC2007/Annotations'
imagepath = '/home/nyma/PycharmProjects/JointSSD/Vedai_Augmented'
# f = open("train.txt", "w")

# list = os.listdir(path)  # dir is your directory path
# number_files = len(list)
# print(number_files)


# def process(annotfile):
#     # base = os.path.basename(annotfile)
#     # base1 = os.path.splitext(base)[0]
#     # print(os.path.getsize(annotfile))
#     # if os.path.getsize(annotfile) == 0:
#     #     os.remove(annotfile)
#     #     os.remove('./FlipPatches/' + base1 + '.jpg')
#
#     for line in fd.readlines():
#
#         words = line.rstrip('\n').split(' ')
#
#         base = os.path.basename(annotfile)
#         base1 = os.path.splitext(base)[0]
#
#         if words[4] == '0':
#             print(file)
#             os.remove(annotfile)
#             os.remove('./Patches512/' + base1 + '.jpg')

path = '/home/nyma/PycharmProjects/JointSSD/Vedai_Augmented'
f = open("train.txt", "w")

for file in sorted(glob.glob(path + '/*.jpg')):
    base = os.path.basename(file)
    base1 = os.path.splitext(base)[0]
    f.write(base1 + '\n')


# for file in sorted(glob.glob(path + '/*.txt')):
#     fd = open(file, 'r')
#     process(file)

# for file in sorted(glob.glob(imagepath + '/*.jpg')):
#     base = os.path.basename(file)
#     imagebase = os.path.splitext(base)[0]
#
#     for file1 in sorted(glob.glob(annotpath + '/*.xml')):
#         file1base = os.path.basename(file1)
#         annotbases= os.path.splitext(file1base)[0]
#         # print("Image Path", imagepath + '/' + annotbase + '.jpg')
#         exist = os.path.isfile(annotpath + '/' + imagebase + '.xml')
#         if exist:
#            print("Ok")
#         else:
#             print(file)
                # os.remove(file)


            # print(os.remove(file))
            # print(exist)
            # print("Found file1", file1)
            # print("Found", annotpath + '/' + imagebase + '.jpg')
        # else:
        #
        #     os.remove(file1)
            # break
        # if (os.path.exists(imagepath+'/'+ annotbase+'.jpg'):
        #     print("Jpeg Image", imagepath+annotbase+'.jpg')
        #     print("Nai")
        #     break
            # os.remove(file1)


    # f.write(base1 + '\n')