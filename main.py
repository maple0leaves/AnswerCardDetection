# 答题卡图像识别

import cv2
import numpy as np
import pandas as pd
from imutils.perspective import four_point_transform
from matplotlib import pyplot as plt


# from matplotlib import pyplot as plt

# import matplotlib
# matplotlib.rc("font",family='AR PL UKai CN')
# 增强亮度
def imgBrightness(img1, c, b):
    rows, cols = img1.shape
    blank = np.zeros([rows, cols], img1.dtype)
    # cv2.addWeighted 实现两副相同大小的图像融合相加
    rst = cv2.addWeighted(img1, c, blank, 1 - c, b)
    return rst


def judgeX(x, mode):
    if mode == "point":
        if x < 600:
            return int(x / 100) + 1
        elif x > 600 and x < 1250:
            return int((x - 650) / 100) + 6
        elif x > 1250 and x < 1900:
            return int((x - 1250) / 100) + 11
        elif x > 1900:
            return int((x - 1900) / 100) + 16
    elif mode == "ID":
        # round 离哪个整数近取哪个整数 --221
        # print('x',round((x-35)/80))
        return round((x-35)/80)+1
        # return int((x) / 50) + 1
        # return int((x-110)/260)+1
    elif mode == "subject":
        if x < 1500:
            return False


def judgeY(y, mode):
    if mode == "point":
        if y % 560 > 180 and y % 560 < 240:
            return 'A'
        elif y % 560 > 260 and y % 560 < 320:
            return 'B'
        elif y % 560 > 340 and y % 560 < 380:
            return 'C'
        elif y % 560 > 420 and y % 560 < 480:
            return 'D'
        else:
            return False

    elif mode == "ID":
        if y > 135:

            # return int((y - 135) / 30)
            return round((y - 255) /48) + 1
            # return int((y-950)/180)
        else:
            return False

    elif mode == "subject":
        print(y, mode)
        # if int((y - 140) / 25) == 0:
        #     return "政治"
        # elif int((y - 140) / 25) == 1:
        #     return "语文"
        # elif int((y - 140) / 25) == 2:
        #     return "数学"
        # elif int((y - 140) / 25) == 3:
        #     return "物理"
        # elif int((y - 140) / 25) == 4:
        #     return "化学"
        # elif int((y - 140) / 25) == 5:
        #     return "英语"
        # elif int((y - 140) / 25) == 6:
        #     return "历史"
        # elif int((y - 140) / 25) == 7:
        #     return "地理"
        # elif int((y - 140) / 25) == 8:
        #     return "生物"
        # else:
        #     return "科目代号填涂有误"
        if round((y - 255) /48)  == 0:
            return "政治"
        elif round((y - 255) /48) == 1:
            return "语文"
        elif round((y - 255) /48)  == 2:
            return "数学"
        elif round((y - 255) /48) == 3:
            return "物理"
        elif round((y - 255) /48) == 4:
            return "化学"
        elif round((y - 255) /48) == 5:
            return "英语"
        elif (round((y - 255) /48) + 1) == 6:
            return "历史"
        elif round((y - 255) /48) == 7:
            return "地理"
        elif round((y - 255) /48) == 8:
            return "生物"
        else:
            return "科目代号填涂有误"


def judge(x, y, mode):
    print('judgeX(x, mode)',judgeX(x, mode))
    if judgeY(y, mode) != False and judgeX(x, mode) != False:
        if mode == "point":
            return (int(y / 560) * 20 + judgeX(x, mode), judgeY(y, mode))
        elif mode == "ID":
            return (judgeX(x, mode), judgeY(y, mode))
    elif mode == "subject":
        print(y, mode)
        return judgeY(y, mode)
    else:
        return 0


def judge_point(answers, mode):
    IDAnswer = []
    for answer in answers:
        if (judge(answer[0], answer[1], mode) != 0):
            IDAnswer.append(judge(answer[0], answer[1], mode))
        else:
            continue
    IDAnswer.sort()
    return IDAnswer


def judge_ID(IDs, mode):
    student_ID = []
    for ID in IDs:
        if (judge(ID[0], ID[1], mode) != False):
            student_ID.append(judge(ID[0], ID[1], mode))
        else:
            continue
    student_ID.sort()
    print('student_ID:', student_ID)
    return student_ID


def judge_Subject(subject, mode):
    return judge(subject[0][0], subject[0][1], mode)


# 边缘检测，获取轮廓信息
def get_contours(img_contours):
    '''
        输入：一张二值化的图片，图片
        输出：查找到的轮廓信息，列表list
    '''
    # canny边缘检测
    edged_ = cv2.Canny(img_contours, 0, 255)

    # 查找检测物体的轮廓信息
    cnts_, hierarchy = cv2.findContours(edged_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 确保至少有一个轮廓被找到
    if len(cnts_) > 0:
        # 将轮廓按照大小排序
        cnts_ = sorted(cnts_, key=cv2.contourArea, reverse=True)
        l = len(cnts_)
        return cnts_

    print('获取轮廓信息出错')
    return 'error'


# 获取图像腐蚀、膨胀处理
def get_erode(img_warped, blockSize, c, iterations_dilate, iterations_erode):
    '''
        输入：
        输出：
        膨胀：增加图片白色区域，抹除细节
        腐蚀：减小图片白色区域，增加细节
  '''
    #这个自适应二值化的参数要认真的调，涉及到能否识别出填涂的选项，这个特别重要！！！！--221
    # thresh_ = cv2.adaptiveThreshold(img_warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, c)
    thresh_ = cv2.adaptiveThreshold(img_warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 81, 2)
    imgplot = plt.imshow(thresh_, cmap='gray')
    plt.title("thresh")
    plt.show()
    
    # 膨胀处理 cv2.dilate
    kernel = np.ones((5, 5), dtype=np.uint8)
    pengzang_ = cv2.dilate(thresh_.copy(), kernel, iterations=iterations_dilate)
    plt.imshow(pengzang_,cmap="gray")
    plt.title("pengzhang")
    plt.show()

    kernel = np.ones((5, 5), np.uint8)
    fushi_ = cv2.erode(pengzang_.copy(), kernel, iterations=iterations_erode)
    plt.imshow(fushi_,cmap="gray")
    plt.title("fushi")
    plt.show()
    return fushi_
    # return pengzang_


# 获取三个主要检测区域
def get_area(img, gray, blurred_border):
    '''
        输入：
        输出：
    '''
    # 查找检测物体的轮廓信息
    cnts = get_contours(blurred_border)
    docCnt = []
    count = 0

    # 对排序后的轮廓进行循环处理
    for c in cnts:

        # 获取近似的轮廓
        # 轮廓周长。也称为弧长
        peri = cv2.arcLength(c, True)
        # 一个重新采s样的轮廓，所以它仍然会返回一组 (x, y) 点
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # 如果近似轮廓有四个顶点，那么就认为找到了答题卡
        # 名字的边框比科目代码的边框更大，所以docCnt要取4个，第三个是名字的边框
        if len(approx) == 4:
            docCnt.append(approx)
            count += 1
            if count == 4:
                break

    # 四点变换, 划出选择题区域
    paper = four_point_transform(img, np.array(docCnt[0]).reshape(4, 2))
    warped = four_point_transform(gray, np.array(docCnt[0]).reshape(4, 2))
    plt.imshow(paper,cmap="gray")
    plt.title("paper")
    plt.show()
    # 四点变换, 划出准考证区域
    ID_Area = four_point_transform(img, np.array(docCnt[1]).reshape(4, 2))
    ID_Area_warped = four_point_transform(gray, np.array(docCnt[1]).reshape(4, 2))

    # 四点变换, 划出科目区域,科目代码是docCnt[3]，名字的边框是docCnt[2]
    # Subject_Area = four_point_transform(img, np.array(docCnt[2]).reshape(4, 2))
    # Subject_Area_warped = four_point_transform(gray, np.array(docCnt[2]).reshape(4, 2))    # 四点变换, 划出科目区域,科目代码是docCnt[3]，名字的边框是docCnt[2]
    Subject_Area = four_point_transform(img, np.array(docCnt[3]).reshape(4, 2))
    Subject_Area_warped = four_point_transform(gray, np.array(docCnt[3]).reshape(4, 2))

    # 图像存储
    cv2.imwrite('select.png', paper)
    cv2.imwrite('number.png', ID_Area)
    cv2.imwrite('course.png', Subject_Area)

    # fig = plt.figure(figsize=(16, 12))
    #
    # # Subplot for original image
    #
    # a = fig.add_subplot(2, 3, 1)
    # imgplot = plt.imshow(paper)
    # a.set_title('select')
    #
    # a = fig.add_subplot(2, 3, 2)
    # imgplot = plt.imshow(ID_Area)
    # a.set_title('number')
    #
    # a = fig.add_subplot(2, 3, 3)
    # imgplot = plt.imshow(Subject_Area)
    # a.set_title('course')
    # plt.show()


    # # 图像存储
    # cv2.imwrite('选择题区域.png', paper)
    # cv2.imwrite('准考证区域.png', ID_Area)
    # cv2.imwrite('科目区域.png', Subject_Area)

    return paper, warped, ID_Area, ID_Area_warped, Subject_Area, Subject_Area_warped


# 处理选择题区域
def get_area_1(paper, warped):
    '''
    '''
    # 处理选择题区域统计答题结果
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    plt.imshow(thresh,cmap="gray")
    plt.title("1111111111thresh")
    plt.show()
    # 图像放大
    thresh = cv2.resize(thresh, (2400, 2800), cv2.INTER_LANCZOS4)
    paper = cv2.resize(paper, (2400, 2800), cv2.INTER_LANCZOS4)
    warped = cv2.resize(warped, (2400, 2800), cv2.INTER_LANCZOS4)

    # 查找检测物体的轮廓信息
    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    questionCnts = []
    answers = []

    # 对每一个轮廓进行循环处理
    for c in cnts:
        # 计算轮廓的边界框，然后利用边界框数据计算宽高比
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        # 判断轮廓是否是答题框
        if w >= 40 and h >= 15 and ar >= 1 and ar <= 1.8:
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            questionCnts.append(c)
            answers.append((cX, cY))
            cv2.circle(paper, (cX, cY), 7, (255, 255, 255), -1)
    print('检测到的选择题目信息长度：', len(answers))

    # 根据像素坐标获取选择题答案
    ID_Answer = judge_point(answers, mode="point")
    # 将像素坐标位置画到原图上
    paper_drawcontours = cv2.drawContours(paper, questionCnts, -1, (255, 0, 0), 3)
    cv2.imwrite('selectresult.jpg', paper)

    return ID_Answer


# 处理准考证号区域
# 这里是要判断涂的选项框的大小，看是不是在范围内从而判断哪个选项  --221
def get_area_2(ID_Area, ID_Area_warped, blockSize, c, iterations_dilate, iterations_erode):
    fushi = get_erode(ID_Area_warped, blockSize, c, iterations_dilate, iterations_erode)
    # plt.subplot(2,1,1)
    # plt.imshow(ID_Area_warped)
    # plt.title("ID_Area")
    #
    # plt.subplot(2,1,2)
    # plt.imshow(fushi)
    # plt.title("fushi")
    # plt.show()
    '''
    cv2.findContours函数返回的轮廓是一组表示图像中对象边界的点的列表。
    轮廓是由一系列的(x, y)坐标组成的，这些坐标连接在一起以形成对象的边界。
    '''
    cnts, hierarchy = cv2.findContours(fushi.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # thresh = cv2.threshold(ID_Area_warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # thresh = cv2.resize(thresh, (2400, 2800), cv2.INTER_LANCZOS4)
    # cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ID_Cnts = []
    IDs = []
    # 对每一个轮廓进行循环处理
    for c in cnts:
        # 计算轮廓的边界框，然后利用边界框数据计算宽高比
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        print("准考证号",w, h, ar, y, x)

        # 判断轮廓是否是答题框
        if w >= 15 and w < 300 and h >= 5 and ar >= 0.9 and ar <= 3.0 and y > 100:
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            if cY < 265:
                ID_Cnts.append(c)
                IDs.append((cX, cY))
                cv2.circle(ID_Area, (cX, cY), 7, (255, 255, 255), -1)

    # 从像素坐标转换成学生的准考证号
    student_IDlist = judge_ID(IDs, mode="ID")
    # 判断准考证号是否有误
    ID_index = 1
    student_ID = ""
    for tuple_ in student_IDlist:
        if tuple_[0] == ID_index:
            student_ID += str(tuple_[1])
            ID_index += 1
        else:
            student_ID = "准考证号填涂有误"
            break
    # 像素坐标信息画到原图上
    ID_Area_drawContours = cv2.drawContours(ID_Area, ID_Cnts, -1, (255, 0, 0), 3)

    cv2.imwrite('numberresult.jpg', ID_Area)
    cv2.imwrite("222.jpg",ID_Area_drawContours)
    return student_ID


# 科目代码区域
def get_area_3(Subject_Area, Subject_Area_warped, blockSize, c, iterations_dilate, iterations_erode):
    '''
        输入：
        输出：
    '''
    Subject_thresh_fushi = get_erode(Subject_Area_warped, blockSize, c, iterations_dilate, iterations_erode)
    cnts, hierarchy = cv2.findContours(Subject_thresh_fushi.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # cnts, hierarchy = cv2.findContours(Subject_thresh_fushi.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    ID_Cnts = []
    IDs = []
    # 对每一个轮廓进行循环处理
    for c in cnts:
        # 计算轮廓的边界框，然后利用边界框数据计算宽高比
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        # 判断轮廓是否是答题框
        if w >= 15 and w < 150 and h >= 5 and ar >= 1 and ar <= 3.0 and x > 50:
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            if cY < 2650:
                ID_Cnts.append(c)
                IDs.append((cX, cY))
                cv2.circle(Subject_Area, (cX, cY), 7, (255, 255, 255), -1)

    student_IDlist = judge_Subject(IDs, mode="subject")
    cv2.imwrite('courseresult.jpg', Subject_Area)
    return student_IDlist


# 存储 表格
def save_csv(answer_path, save_path, ID_Answer, student_ID, student_IDlist):
    df = pd.read_excel(answer_path)
    index_list = df[["题号"]].values.tolist()
    true_answer_list = df[["答案"]].values.tolist()
    index = []
    true_answer = []
    score = 0

    # 去括号
    for i in range(len(index_list)):
        index.append(index_list[i][0])
    for i in range(len(true_answer_list)):
        true_answer.append(true_answer_list[i][0])

    answer_index = []
    answer_option = []
    for answer in ID_Answer:
        answer_index.append(answer[0])
        answer_option.append(answer[1])
    for i in range(len(index)):
        if answer_option[i] == true_answer[i]:
            score += 1
        if i + 1 == len(answer_option):
            break

    # 写入表格
    info = {"试卷类型": ["A"], "准考证号": [student_ID], "科目代号": [student_IDlist]}
    df1 = pd.DataFrame(info)
    df2 = pd.DataFrame(np.array(true_answer).transpose(), index=index, columns=["正确答案"])
    df2["学生答案"] = ''

    for i in range(len(answer_option)):
        df2["学生答案"][i + 1] = answer_option[i]

    df2["总得分"] = ''
    df2["总得分"][1] = score
    with pd.ExcelWriter(save_path) as writer:
        df1.to_excel(writer, index=False, sheet_name="type")
        df2.to_excel(writer, sheet_name="score")
        writer._save()
        print("导出excel成功！")

    return


# 答题卡图片处理主要步骤
def img_1(image_path, answer_path, save_path):
    # 读取图片 RGB \ BGR
    img = cv2.imread(image_path)
    # 转换为灰度图 COLOR_BGR2GRAY、COLOR_BGR2RGB
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # 高斯滤波 中值滤波
    blurred_gauss = cv2.GaussianBlur(gray, (3, 3), 0)

    # 增强亮度 选择开启
    blurred_bright = imgBrightness(blurred_gauss, 1.5, 3)

    # 自适应二值化
    blurred_threshold = cv2.adaptiveThreshold(blurred_bright, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 2)
    # blurred_threshold = cv2.adaptiveThreshold(blurred_gauss, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 2)
    # 显示原来的和缩放后的图像  Create a figure
    fig = plt.figure(figsize=(16, 12))

    # Subplot for original image
    a = fig.add_subplot(2, 3, 1)
    imgplot = plt.imshow(img)
    a.set_title('origin')

    a = fig.add_subplot(2, 3, 2)
    imgplot = plt.imshow(gray, cmap='gray')
    a.set_title('gray')

    a = fig.add_subplot(2, 3, 3)
    imgplot = plt.imshow(blurred_gauss, cmap='gray')
    a.set_title('blurred_gauss')

    a = fig.add_subplot(2, 3, 4)
    imgplot = plt.imshow(blurred_bright, cmap='gray')
    a.set_title('blurred_bright')

    a = fig.add_subplot(2, 3, 5)
    imgplot = plt.imshow(blurred_threshold, cmap='gray')
    a.set_title('blurred_threshold')

    plt.show()
    # plt.savefig('5张图片.png', bbox_inches='tight')

    # 用来给图片添加边框
    blurred_border = cv2.copyMakeBorder(blurred_threshold, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 255, 0))
    # bordered_img_rgb = cv2.cvtColor(blurred_border, cv2.COLOR_BGR2RGB)
    #
    # plt.imshow(bordered_img_rgb)
    # plt.title("666")
    # plt.show()
    # 获取主要区域
    paper, warped, ID_Area, ID_Area_warped, Subject_Area, Subject_Area_warped = get_area(img, gray, blurred_border)

    # 处理选择题区域
    ID_Answer = get_area_1(paper, warped)

    # 处理准考证号区域
    student_ID = get_area_2(ID_Area, ID_Area_warped, 53, 30, 3, 3)

    # 科目代码区域
    student_IDlist = get_area_3(Subject_Area, Subject_Area_warped, 53, 30, 4, 3)

    # 存储表格
    save_csv(answer_path, save_path, ID_Answer, student_ID, student_IDlist)

    return


if __name__ == '__main__':
    image_path = 'img/img7.jpg'
    answer_path = "answer.xlsx"
    save_path = "test.xlsx"

    img_1(image_path, answer_path, save_path)
    print('预测结束')
