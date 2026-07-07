#####  modified 0703
from collections import deque
import numpy as np
import cv2
import json
import math
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')



###########计算两个向量的夹角###############
def angle_between_vectors(Vetor1, Vetor2):
    A=Vetor1
    B=Vetor2

    if A[0] != 0 or A[1] != 0:
        if B[0] != 0 or B[1] != 0:
            dot_product = A[0] * B[0] + A[1] * B[1]
            magnitude_A = math.sqrt(A[0] ** 2 + A[1] ** 2)
            magnitude_B = math.sqrt(B[0] ** 2 + B[1] ** 2)
            cos_theta = dot_product / (magnitude_A * magnitude_B)
            if (abs(cos_theta) < 1):
                theta = math.acos(cos_theta)
                angel = math.degrees(theta)
                return angel
    else:
        angel0 = 180
        return angel0
def angle_between_lines(A, B, C):
    # 计算向量AB和BC
    BA = [A[0]-B[0], A[1]-B[1]]
    BC = [C[0]-B[0], C[1]-B[1]]

    # 计算向量BA和BC之间的夹角
    angle = angle_between_vectors(BA, BC)

    return angle




# def is_near_region(y_coordinate, image_height):
#     """
#     判断车辆是否位于近处区域。
#     假设图像分为两部分：上半部分（远处），下半部分（近处）。
#     """
#     # 设置分界线为图像高度的2/3处
#     boundary_line = int(image_height * 2 / 3)
#     return y_coordinate > boundary_line

# 异常事件的判断的函数
def trackVehicleInSeqpre(inTrackList, segmentation_map):
    abnormalResListpre = [0, 0, 0, 0, 0, 0, 0]
    allidlist = []

    data_deque = {}
    pltalltrack = {}
    zuoshangalltrack = {}
    zuoxiaalltrack = {}
    youxiaalltrack = {}
    zhongxiaalltrack = {}
    zhongshangalltrack = {}
    zhongshangalltrackspeed = {}
    xiangsualltrack = {}
    centeralltrack = {}

    duche_speeddict1 = defaultdict(list)
    duche_speeddict2 = defaultdict(list)

    duche_speedidlist1 = []
    duche_speedidlist2 = []

    k1 = 0
    k2 = 0
    k3 = 0
    shiguidlist = []
    duche_way2_idlist = []

    weitingid = []
    weitingidlist = []

    weitingboxlist = []
    biaoshiidlist = []
    guijiyichangout = []
    guijiyichangoutidlist = []
    allid = []




    # 下面是一帧一帧处理的
    for j, input in enumerate(inTrackList):
        if len(input) == 0:
            continue
        bbox = input[:, :4]  # 提取bbox，bbox是一帧中的所有坐标
        identities = input[:, -1]  # 将输入的ID放在一起；【12 11 10 9 8 7 6 5 4 3 2 1】，一帧中所有出现的id存起来

        # 如果对象丢失超过100帧，从data_deque中删除
        for key in list(data_deque):
            if key not in identities:
                k1 = k1 + 1
                if (k1 >= 100):
                    data_deque.pop(key)

        for key in list(zhongxiaalltrack):
            if key not in identities:
                k2 = k2 + 1
                if (k2 >= 100):
                    zhongxiaalltrack.pop(key)

        for key in list(zhongshangalltrack):
            if key not in identities:
                k3 = k3 + 1
                if (k3 >= 100):
                    zhongshangalltrack.pop(key)


        # 对每一帧的所有坐标bbox提取出，一个一个坐标box，进行处理；得到data_deque：每个id后面存储它的所有中心点的坐标；
        # for i, box in enumerate(bbox):
        #     x1, y1, x2, y2 = [int(i) for i in box]
        for i, box in enumerate(bbox):
            x1, y1, w, h = [int(i) for i in box]
            x2 = x1 + w
            y2 = y1 + h 
            # 得到中心点
            center = (int((x2 + x1) / 2), int((y1 + y2) / 2))

            # 得到左下角，右下角坐标，方便后面占用应急车道的判断
            zuoshang = (int(x1), int(y1))
            zuoxia = (int(x1), int(y2))
            youxia = (int(x2), int(y2))
            zhongxia = (int((x2 + x1) / 2), int(y2))
            zhongshang = (int((x2 + x1) / 2), int(y1))
            xiangsu = int(abs((x1 - x2) * (y1 - y2)))

            # 得到对应车的id
            id = int(identities[i]) if identities is not None else 0
            allid.append(id)
            allidlist = list(set(allid))

            # 为新对象创建新缓冲区    data_deque：三维列表，是每个id对应在不同帧中的中点坐标
            if id not in data_deque:
                data_deque[id] = deque(maxlen=200)
            # 数据形式：每个id后面存储它的所有中心点的坐标；
            data_deque[id].appendleft(center)

            if id not in pltalltrack:
                pltalltrack[id] = deque(maxlen=200)
            # 数据形式：每个id后面存储它的所有中心点的坐标；
            pltalltrack[id].appendleft(center)


            if id not in zuoshangalltrack:
                zuoshangalltrack[id] = deque(maxlen=200)
                # 数据形式：每个id后面存储它的所有左下的坐标；
            zuoshangalltrack[id].appendleft(zuoshang)

            if id not in zuoxiaalltrack:
                zuoxiaalltrack[id] = deque(maxlen=200)
                # 数据形式：每个id后面存储它的所有左下的坐标；
            zuoxiaalltrack[id].appendleft(zuoxia)

            if id not in youxiaalltrack:
                youxiaalltrack[id] = deque(maxlen=200)
                # 数据形式：每个id后面存储它的所有右下的坐标；
            youxiaalltrack[id].appendleft(youxia)

            if id not in zhongxiaalltrack:
                zhongxiaalltrack[id] = deque(maxlen=200)
                # 数据形式：每个id后面存储它的所有右下的坐标；
            zhongxiaalltrack[id].appendleft(zhongxia)

            if id not in zhongshangalltrack:
                zhongshangalltrack[id] = deque(maxlen=200)
                # 数据形式：每个id后面存储它的所有右下的坐标；
            zhongshangalltrack[id].appendleft(zhongshang)

            if id not in zhongshangalltrackspeed:
                zhongshangalltrackspeed[id] = deque(maxlen=200)
                # 数据形式：每个id后面存储它的所有右下的坐标；
            zhongshangalltrackspeed[id].appendleft(zhongshang)

            if id not in xiangsualltrack:
                xiangsualltrack[id] = deque(maxlen=200)
                # 数据形式：每个id后面存储它的所有右下的坐标；
            xiangsualltrack[id].appendleft(xiangsu)

            if id not in centeralltrack:
                centeralltrack[id] = deque(maxlen=200)
            # 数据形式：每个id后面存储它的所有中心点的坐标；
            centeralltrack[id].appendleft(center)


            ################# 判断违停
            # if (len(data_deque[id]) > 70):  # 出现在画面时间比较长进行判断
            #     a = np.sqrt((data_deque[id][0][0] - data_deque[id][69][0]) ** 2 + (
            #
            #                 # 城市里，像素点变动什么范围合适？一般是3-5
            #                 data_deque[id][0][1] - data_deque[id][69][1]) ** 2)  # 计算像素点的变动
            #
            #
            #     # 记录违规车辆
            #     if (a < 2):
            #         #abnormalResListpre[0] = 1
            #
            #         weitingboxlist.append(box)  # 记录违停所有车辆的所有目标框坐标
            #         weitingid.append(id)  # 记录违停车辆的所有id

            # TODO  200帧（大约8秒）可以过滤掉多余的异常停车
            #print(f"len(data_deque[id]): {len(data_deque[id])}")
            if (len(data_deque[id]) > 70):  # 出现在画面时间比较长进行判断200
                a = np.sqrt((data_deque[id][0][0] - data_deque[id][69][0]) ** 2 + (

                    # 城市里，像素点变动什么范围合适？一般是3-5
                        data_deque[id][0][1] - data_deque[id][69][1]) ** 2)  # 计算像素点的变动
                #print(f"valueofa: {a}")

                # 记录违规车辆
                if (a < 2):
                    # abnormalResListpre[0] = 1

                    weitingboxlist.append(box)  # 记录违停所有车辆的所有目标框坐标
                    weitingid.append(id)  # 记录违停车辆的所有id
                    print(f"weitingid: {weitingid}")
                    print(f"len_weitingid: {len(weitingid)}")
            # 添加远近区域分割逻辑


            # ################# 判断违停
            # if (len(data_deque[id]) > 70):  # 出现在画面时间比较长进行判断
            #     a = np.sqrt((data_deque[id][0][0] - data_deque[id][69][0]) ** 2 + (data_deque[id][0][1] - data_deque[id][69][1]) ** 2)  # 计算像素点的变动
                
            #     # 获取车辆中心点的 y 坐标，用于判断车辆是否在近处区域
            #     y_coordinate = data_deque[id][0][1]
            #     image_height = 1080
            #     # 判断车辆所在区域
            #     if is_near_region(y_coordinate, image_height):
            #         # 远处区域
            #         threshold = 1
            #     else:
            #         # 近处区域
            #         threshold = 2

            #     # 记录违规车辆
            #     if (a < threshold):
            #         # abnormalResListpre[0] = 1

            #         weitingboxlist.append(box)  # 记录违停所有车辆的所有目标框坐标
            #         weitingid.append(id)  # 记录违停车辆的所有id
      


                    #  #####警示标志#########################
                    # if biaoshinum == 0:
                    #     abnormalResListpre[3] = 1
                    #     biaoshiidlist = weitingidlist
                    
                    
                    ######判断堵车方法二############
                    # duche_way2_idlist.append(id)
                    # num1 = list(set(duche_way2_idlist))
                    # if (len(num1) >= 4) :
                    #     abnormalResListpre[6] = 1
                    
                    # # 判断交通事故发生
                    # shiguidlist.append(id)
                    # num2 = list(set(shiguidlist))
                    # if(len(num2) >= 2) and (len(num2) <= 3):
                    #     abnormalResListpre[5] = 1
                    # #先吧交通事故判断注销，后面添加颜色识别
                    # if (len(num2) >= 500) :
                    #     abnormalResListpre[5] = 1

                    


            ############判断堵车的先决条件之一速度：所有车的平均速度；1/4 ###################
            # 平均速度此种计算方式是否合适？
            if (len(data_deque[id]) > 40):

                # first1_y = zuoxiaalltrack[id][39][1]
                duche_y1_1 = zuoxiaalltrack[id][39][1]
                duche_y1_2 = zhongshangalltrackspeed[id][39][1]
                duche_y1_3 = abs(((duche_y1_1 - duche_y1_2) / 4) * 3)
                duche_first1_y = zuoxiaalltrack[id][39][1] - duche_y1_3

                if zuoxiaalltrack[id][1][1] > zuoxiaalltrack[id][39][1]:
                    i = 1
                    for j in range(38, -1, -1):
                        duche_ing1_y = zhongshangalltrack[id][j][1]
                        if duche_ing1_y < duche_first1_y:
                            i = i + 1
                        if duche_ing1_y >= duche_first1_y:
                            duche_speed = (4 * 25) / i
                            duche_speedidlist1.append(id)
                            duche_speedidlist1 = list(set(duche_speedidlist1))
                            duche_speeddict1[id].append(duche_speed)
                            break
                        if i == 37:
                            duche_speed = 2
                            duche_speedidlist1.append(id)
                            duche_speedidlist1 = list(set(duche_speedidlist1))
                            duche_speeddict1[id].append(duche_speed)
                            break


                # first2_y = zhongshangalltrack[id][39][1]
                duche_y2_1 = zhongshangalltrackspeed[id][39][1]
                duche_y2_2 = zuoxiaalltrack[id][39][1]
                duche_y2_3 = abs(((duche_y2_1 - duche_y2_2) / 4) * 3)
                duche_first2_y = zhongshangalltrackspeed[id][39][1] + duche_y2_3

                if zhongshangalltrack[id][1][1] < zhongshangalltrack[id][39][1]:
                    i = 1
                    for j in range(38, -1, -1):
                        duche_ing2_y = zuoxiaalltrack[id][j][1]
                        if duche_ing2_y > duche_first2_y:
                            i = i + 1
                        if duche_ing2_y <= duche_first2_y:
                            duche_speed = (4 * 25) / i
                            duche_speedidlist2.append(id)
                            duche_speedidlist2 = list(set(duche_speedidlist2))
                            duche_speeddict2[id].append(duche_speed)
                            break
                        if i == 37:
                            duche_speed = 2
                            duche_speedidlist2.append(id)
                            duche_speedidlist2 = list(set(duche_speedidlist2))
                            duche_speeddict2[id].append(duche_speed)
                            break




            # ###############车道检测前根据车辆夹角来 判断是否轨迹异常###############################
            if (len(data_deque[id]) > 10):
                a = np.sqrt((data_deque[id][0][0] - data_deque[id][9][0]) ** 2 + (data_deque[id][0][1] - data_deque[id][9][1]) ** 2)
                if(a>2):
                    A = (data_deque[id][5][0], data_deque[id][5][1])
                    B = (data_deque[id][2][0], data_deque[id][2][1])
                    C = (data_deque[id][0][0], data_deque[id][0][1])

                    angle = angle_between_lines(A, B, C)
                    if angle is not None:
                        if int(angle) < 10:
                            guijiyichangout.append(id)  ###记录轨迹异常所有车辆的所有id
                            count_dict = {}
                            for i in guijiyichangout:
                                if i not in count_dict:
                                    count_dict[i] = 1
                                else:
                                    count_dict[i] += 1

                            new_a = []
                            for i in guijiyichangout:
                                if count_dict[i] >= 100:
                                    new_a.append(i)
                                    abnormalResListpre[1] = 1
                            guijiyichangoutidlist = list(set(new_a))



    ##判断堵车的部分
    ##计算所有车平均速度#######
    average_duche_speeds1 = {}
    average_duche_speeds2 = {}
    average_duche_speeds = {}
    ducheidlist = []
    for id in duche_speedidlist1:
        duche_speeds = duche_speeddict1[id]
        average_duche_speed = sum(duche_speeds) / len(duche_speeds)
        average_duche_speeds1[id] = average_duche_speed


    for id in duche_speedidlist2:
        duche_speeds = duche_speeddict2[id]
        average_duche_speed = sum(duche_speeds) / len(duche_speeds)
        average_duche_speeds2[id] = average_duche_speed


    average_duche_speeds.update(average_duche_speeds1)
    average_duche_speeds.update(average_duche_speeds2)
    print(f"average_duche_speeds: {average_duche_speeds}")
    print(f"len_average_duche_speeds: {len(average_duche_speeds)}")
 

# ###########原堵车
    # if len(average_duche_speeds) >= 3:
    #     ##########
    #         sum_values = sum(v for k, v in average_duche_speeds.items())
    #         count = len(average_duche_speeds)
    #         # 计算平均值
    #         average_duche = sum_values / count
    #         print(f"average_duche: {average_duche}")
    #         print(f"len_allidlist: {len(allidlist)}")
    #         ###所有车平均速度低于3，且车数量大于等于15
    #         ####把这种情况规避了##############8 45 200
    #         if average_duche <= 3 and len(allidlist) >= 60:
    #             #print(len(allidlist))
    #             abnormalResListpre[6] = 1
    #             ducheidlist = allidlist






    ##############违停的过滤#############
    ###这里得到有三次异常判断为违停的；小于等于三次过滤#####
    count_dict = {}
    for num in weitingid:
        if num in count_dict:
            count_dict[num] += 1
        else:
            count_dict[num] = 1

    for key, value in count_dict.items():
        print(f"key: {key}, value: {value}")
        if value > 4:
            weitingidlist.append(key)

    if len(weitingidlist) > 0:
        for id in weitingidlist:
            count = sum(1 for num in xiangsualltrack[id] if num < 900)
            value_1 = count / len(xiangsualltrack[id])
            if value_1 >= 0.3:
                weitingidlist = [x for x in weitingidlist if x != id]



    #########20240403 wwj add road_reg
    # 选择的是第70帧这一点的车辆目标框来做判断的。0是黑色区域(非高速公路)、255是白色区域(高速公路)
    if len(weitingidlist) > 0:
        allblack_bool = np.all(segmentation_map == 0)  #true表示全黑
        if not allblack_bool:
            for id in weitingidlist:
                x1 = zuoshangalltrack[id][69][0]
                y1 = zuoshangalltrack[id][69][1]
                x2 = youxiaalltrack[id][69][0]
                y2 = youxiaalltrack[id][69][1]
                # 检查并交换坐标以确保 x1 < x2 和 y1 < y2
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                roi = segmentation_map[y1:y2, x1:x2]
                # roi = segmentation_map[y2:y1, x2:x1]
                non_zero_pixels = cv2.countNonZero(roi)  # 得到矩形框中白色像素的个数。(就是矩形框与高速道路区域的交集的像素个数)
                # print(f"id: {id}, x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, non_zero_pixels: {non_zero_pixels}")
                if non_zero_pixels == 0:
                    weitingidlist = [x for x in weitingidlist if x != id]

                # 使用OpenCV绘制矩形框
                # img_with_box = cv2.cvtColor(segmentation_map, cv2.COLOR_GRAY2BGR)  # 转换为BGR图像以便绘制彩色矩形框
                # cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色矩形框

                # 保存结果图像
                # cv2.imwrite(f"segmentation_map_{id}.png", img_with_box)
                # plt.figure()
                # plt.imshow(segmentation_map, cmap='gray')
                # plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='green', facecolor='none', linewidth=2))
                # plt.title(f"Segmentation Map with Bounding Boxes for id {id}")
                # plt.savefig(f"segmentation_map_{id}.png")
                # plt.close()


            
    #########################################

    print(f"lenofweitingidlist: {len(weitingidlist)}")
    if len(weitingidlist) > 0:
        # if len(weitingidlist) >= 1 and len(weitingidlist) <= 3 :
        #    abnormalResListpre[0] = 1

        #         ###交通事故和堵车的判断
        num1 = list(set(weitingidlist))







###############################异常停车筛选
        if len(weitingidlist) >= 0 and len(weitingidlist) <= 3 :
            centers = []
            for id in num1:
                if id in centeralltrack and len(centeralltrack[id]) > 69:
                    centers.append(centeralltrack[id][69])
                elif id in centeralltrack:
                    centers.append(centeralltrack[id][0])
            close_enough = True
            for i in range(len(centers)):
                for j in range(i+1, len(centers)):
                    dist = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
                    print(f"dist: {dist}")
                    if dist <= 100:
                        close_enough = False
                        break
                if not close_enough:
                    break
            if close_enough:
                # normal_ids = [id for id in allidlist if id not in num1]
                # normal_centers = []
                # for id in normal_ids:
                #     if id in centeralltrack and len(centeralltrack[id]) > 69:
                #         normal_centers.append(centeralltrack[id][69])
                #     elif id in centeralltrack:
                #         normal_centers.append(centeralltrack[id][0])
                # abnormalstop_like = True
                # min_dist_threshold = 200  # 阈值可调整
                # dist_shigu_list = []
                # for c in centers:
                #     for nc in normal_centers:
                #         dist_shigu = np.linalg.norm(np.array(c) - np.array(nc))
                #         dist_shigu_list.append(dist_shigu)
                #         print("dist_shigu:", dist_shigu)
                #         print("dist_shigu最小值:", min(dist_shigu_list))
                #         if min(dist_shigu_list) < min_dist_threshold:
                #             abnormalstop_like = False
                #             break
                #     if not abnormalstop_like:
                #         break
                # if abnormalstop_like:
                #     # 画面尺寸
                # img_w, img_h = 1920, 1080
                # edge_margin = 200  # 距离边缘200像素以内算边缘
                # all_on_edge = True
                # for c in centers:
                #     x, y = c
                #     if not (
                #         x <= edge_margin or x >= img_w - edge_margin #or
                #         #y <= edge_margin or y >= img_h - edge_margin
                #     ):
                #         all_on_edge = False
                #         break
                # if all_on_edge:
                    shiguidlist = weitingidlist
                    abnormalResListpre[0] = 1
                # shiguidlist = weitingidlist
                # abnormalResListpre[0] = 1
        #num1 = list(set(weitingidlist))
    ##################################
        # if (len(num1) >= 2) and (len(num1) <= 3):
        #     shiguidlist = weitingidlist
        #     abnormalResListpre[5] = 1
        #交通事故判断修改#######################
        if (len(num1) >= 2) and (len(num1) <= 3):
            centers = []
            for id in num1:
                if id in centeralltrack and len(centeralltrack[id]) > 69:
                    centers.append(centeralltrack[id][69])
                elif id in centeralltrack:
                    centers.append(centeralltrack[id][0])
            close_enough = True
            for i in range(len(centers)):
                for j in range(i+1, len(centers)):
                    dist = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
                    if dist >= 50:
                        close_enough = False
                        break
                if not close_enough:
                    break
            if close_enough:
                normal_ids = [id for id in allidlist if id not in num1]
                normal_centers = []
                for id in normal_ids:
                    if id in centeralltrack and len(centeralltrack[id]) > 69:
                        normal_centers.append(centeralltrack[id][69])
                    elif id in centeralltrack:
                        normal_centers.append(centeralltrack[id][0])
                shigu_like = True
                min_dist_threshold = 100  # 阈值可调整
                dist_shigu_list = []
                for c in centers:
                    for nc in normal_centers:
                        dist_shigu = np.linalg.norm(np.array(c) - np.array(nc))
                        dist_shigu_list.append(dist_shigu)
                        print("dist_shigu:", dist_shigu)
                        print("dist_shigu最小值:", min(dist_shigu_list))
                        if min(dist_shigu_list) < min_dist_threshold:
                            shigu_like = False
                            break
                    if not shigu_like:
                        break
                if shigu_like:
                    shiguidlist = weitingidlist
                    abnormalResListpre[5] = 1
        #交通事故判断修改###############################

        print(f"len(num1): {len(num1)}")
        #duchepanduan:4
        # if (len(num1) >= 14):
        #    duche_way2_idlist = weitingidlist
        #    abnormalResListpre[6] = 1

    #print(len(weitingidlist))



        # # 下面判断警示标志的代码
        # if biaoshinum == 0:
        #     abnormalResListpre[3] = 1
        #     biaoshiidlist = weitingidlist


###############堵车判断###########################
    if len(average_duche_speeds) >= 3:
        ##########
            sum_values = sum(v for k, v in average_duche_speeds.items())
            count = len(average_duche_speeds)
            # 计算平均值
            average_duche = sum_values / count
            print(f"average_duche: {average_duche}")
            print(f"len_allidlist: {len(allidlist)}")
            ###所有车平均速度低于3，且车数量大于等于15
            ####把这种情况规避了##############8 45 200
            if average_duche <= 20 and len(allidlist) >= 15:
            #if average_duche <= 3 and len(allidlist) >= 60 and len(weitingidlist)>=4:
                #print(len(allidlist))
                abnormalResListpre[6] = 1
                ducheidlist = allidlist
#############################################################
    ##############交通事故和堵车的过滤#############




    maybe_disuidlist = []
    disuidlist = []
    cesuidlist = allidlist
    car_num = 0
    max_count = 0
    max_id = None
    bool_value = 1
    ####################################################################测速方案2.0##################################################
    ###当监控与道路趋于平行的时候，不判断低速行驶。
    for id_max, coordinates in centeralltrack.items():
        count = len(coordinates)
        if count > max_count:
            max_count = count
            max_id = id_max
    if max_id is not None:
        max_y_value = abs(centeralltrack[max_id][0][1] - centeralltrack[max_id][-1][1])
        if max_y_value <= 10:
            bool_value = 0

    for id in cesuidlist:
        i = 0
        while i < len(centeralltrack[id]):
            if centeralltrack[id][i][1] < 720:
                del centeralltrack[id][i]
            else:
                i += 1
        if len(centeralltrack[id]) >= 200:
            b = np.sqrt((centeralltrack[id][0][0] - centeralltrack[id][199][0]) ** 2 + (
                        centeralltrack[id][0][1] - centeralltrack[id][199][1]) ** 2)
            print(f"b_value: {b}")
            if 10 < b and b < 40:
                car_num = car_num + 1
            if 10 <= b and b < 20:
                maybe_disuidlist.append(id)
    print(f"car_num: {car_num}")
    print(f"len(maybe_disuidlist): {len(maybe_disuidlist)}")
    print(f"disu_id: {id}")

    if car_num < 3 and bool_value == 1:
        if len(maybe_disuidlist) > 0:
            disuidlist = maybe_disuidlist
            abnormalResListpre[4] = 1




            
    if abnormalResListpre[6] == 1:
        abnormalResListpre[5] = 0
        if len(ducheidlist) == 0:
            ducheidlist = duche_way2_idlist

    # print(f"abnormalResListpre: {abnormalResListpre}")
    # print(f"shiguidlist:{shiguidlist}")
    return abnormalResListpre, weitingidlist,  guijiyichangoutidlist, biaoshiidlist,disuidlist,shiguidlist,ducheidlist



 
