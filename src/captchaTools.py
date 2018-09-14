import cv2
import numpy as np

import os

resizeRate = 4
ponit_min_width = 2

def getSpecColorMask(img, hsvRanges) :
    # change to hsv model
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # get mask
    mask =  cv2.inRange(hsv, hsvRanges[0][0], hsvRanges[0][1])
    if len(hsvRanges) > 1 : 
        for index in range(len(hsvRanges)):
            if index > 0 :
                maskTemp = cv2.inRange(
                    hsv, hsvRanges[index][0], hsvRanges[index][1])
                mask = cv2.bitwise_or(mask, maskTemp)

    return mask

# todo 没用，考虑删除
def removeNoise(img):
    #定义结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    # 腐蚀
    erosion = cv2.erode(img,kernel,iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    opened_mask = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel)    
    return img

def removeIsolatePoints(binaryImg):
    # img.shape[0]为height,img.shape[1]为width
    img = np.copy(binaryImg)
    region_half_size = 3
    region_total_ponits = 3
    for x in range(binaryImg.shape[1]):
        for y in range(binaryImg.shape[0]):
            sum = sumRegionPoints(binaryImg, x, y, region_half_size)
            if sum < region_total_ponits:
                img[y,x] = 255
    return img


def sumRegionPoints(img, x, y, region_half_size):
    '''
    统计以二值化的img[白底黑字]中以(x, y)为中心，region_half_size为半径的正方形内所有像素为黑色的个数
    '''
    min_x = x - region_half_size
    min_x = min_x if(min_x>=0) else 0
    max_x = x + region_half_size
    max_x = max_x if(max_x <= img.shape[1]) else img.shape[1]

    min_y = y - region_half_size
    min_y = min_y if(min_y>=0) else 0
    max_y = y + region_half_size
    max_y = max_y if(max_y <= img.shape[0]) else img.shape[0]

    sum = 0
    for yy in range(min_y, max_y): 
        for xx in range(min_x, max_x):
            if img[yy, xx] == 0:
                sum = sum + 1
    return sum


def getSplitImageScope(binaryImg):
    # img.shape[0]为height,img.shape[1]为width
    # 创建x轴的投影
    xWidth = binaryImg.shape[1]
    xProjection = [0]*xWidth
    for x in range(xWidth):
        for y in range(binaryImg.shape[0]):
            if binaryImg[y, x] == 0:
                xProjection[x] += 1
    # print('touying', xProjection)
    
    # 两个字符间的最小间隔, 需要 > 0
    CONFIG_MIN_GAP_SIZE = 1
    # X轴投影后的最小值，大于这个值才会算做分隔的一条槽, 需要 >= 0
    CONFIG_MIN_VERTICAL_PONIT_COUNT = 0
    # 一段字符的最小宽度
    CONFIG_MIN_SCOPE_WIDTH = 3
    splitScope = []
    gap = {}
    for index, item in enumerate(xProjection):
        if item > CONFIG_MIN_VERTICAL_PONIT_COUNT:
            if 'start' not in gap:
                gap['start'] = index
        else:
            if 'start' in gap:
                endCount = 0
                # 如果当前index后面连续CONFIG_MIN_GAP_SIZE项内，有不是0的值，就不应该结束当前字符
                for tempIndex in range(index, index + CONFIG_MIN_GAP_SIZE + 1):
                    if tempIndex < xWidth and xProjection[tempIndex] <= CONFIG_MIN_VERTICAL_PONIT_COUNT :
                        endCount = endCount + 1
                if endCount > CONFIG_MIN_GAP_SIZE:
                    gap['end'] = index
                    splitScope.append(gap.copy())
                    gap.clear()
    optimizedScope = []
    for scope in splitScope:
        if scope['end'] - scope['start'] >= CONFIG_MIN_SCOPE_WIDTH:
            optimizedScope.append(scope)
    return optimizedScope

def splitImg(img, scopes, imageName, path):
    for scope in scopes:
        blockImg = img[0:img.shape[0], scope['start']:scope['end']]
        width = scope['end'] - scope['start']
        cv2.imwrite(path + imageName + '-split-' + str(width) + '.png', blockImg)

def processImage(imagePath, hsvRanges):
    iamgeName = imagePath[imagePath.rfind('/')+1:].replace('.png','')
    originalImage = cv2.imread(imagePath)
    resizedImage = cv2.resize(originalImage, (0, 0), fx=resizeRate, fy=resizeRate)

    cv2.imwrite(image_out_path+iamgeName +
                '-00-resizedImage.png', resizedImage)

    imageOnlyWithSpecColor = getSpecColorMask(resizedImage, hsvRanges)

    # cv2.imwrite(image_out_path+iamgeName+'-01-imageOnlyWithSpecColor.png', imageOnlyWithSpecColor)

    # # cvtColor：将彩色图转为灰度图
    # grayImg = cv2.cvtColor(imageOnlyWithSpecColor, cv2.COLOR_RGB2GRAY)

    # # # 去燥
    prueImg = removeNoise(imageOnlyWithSpecColor)
    # cv2.imwrite(image_out_path+iamgeName+'-02-prueImg.png', prueImg)

    # 恢复缩放之前的大小
    normalSizeImage = cv2.resize(
        prueImg, (0, 0), fx=1/resizeRate, fy=1/resizeRate)
    # cv2.imwrite(image_out_path+iamgeName+'-03-normalSizeImage.png', normalSizeImage)

    # threshold：将图像二值化为黑白图片
    _ret, binaryImage = cv2.threshold(normalSizeImage, 250, 255, cv2.THRESH_BINARY_INV)
    # cv2.imwrite(image_out_path+iamgeName+'-04-binaryImage.png', binaryImage)

    removedPointsImg = removeIsolatePoints(binaryImage)
    cv2.imwrite(image_out_path+iamgeName +
                '-05-removedPointsImg.png', removedPointsImg)

    splitImageScope = getSplitImageScope(removedPointsImg)

    print('splitImageScope', splitImageScope)

    splitImg(removedPointsImg, splitImageScope, iamgeName, image_out_path)



# set red thresh
# hsvRanges = [[np.array([0,70,50]), np.array([10,255,255])], 
            #  [np.array([170, 70, 50]), np.array([180, 255, 255])]]

# set blue thresh
hsvRanges = [[np.array([100,43,46]), np.array([124,255,255])]]

# processImage('./resources/red/red01.png', lower_color_red, upper_color_red)


image_path = '/Users/leallee/Downloads/CaptchaSummary/blue-en-1/'
image_out_path = image_path + 'out/'

for root, dirs, files in os.walk(image_path, topdown=False):
    for name in files:
        if os.path.splitext(name)[1] == '.jpg': 
            print('--- start processing ', os.path.join(root, name))
            processImage(os.path.join(root, name), hsvRanges)
