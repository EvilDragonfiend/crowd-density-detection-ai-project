import cv2
from ultralytics import YOLO
import joblib
import numpy as np
import pandas as pd
import glob
import os
import time #나중에 제거
import statistics

from flask import Flask, render_template, request

MODEL_IMG_PATH = "./image_model/best_v14.pt"
my_image_model = YOLO(MODEL_IMG_PATH)
MODEL_PRED_PATH = './predict_model/sklearn_svr_model.pkl'
model_pred = joblib.load(MODEL_PRED_PATH)
app = Flask(__name__, template_folder='.')

@app.route('/')
def index():
    return render_template('/html/index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']

    # Save the uploaded file to a folder
    file_to_save = os.path.join('static', file.filename)
    file.save(file_to_save)

    image = cv2.imread(file_to_save)
    results = my_image_model(image)
    result_item = results[0]
    boxes = result_item.boxes.xyxy
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    fname = file.filename.split('.')
    fname = fname[len(fname)-1]
    output_file = os.path.join('static', 'result.'+fname)
    cv2.imwrite(output_file, image)

    height, width, _ = image.shape
    refined_data = run_data_refine(boxes, height, width)
    predictions = round(model_pred.predict(refined_data)[0], 2)
    time.sleep(2)
    return render_template('/html/uploaded.html', filename='result.'+fname, detected=len(boxes), myresult=predictions)


def run_data_refine(detected_boxes, height, width):
    box_field_names = [
                        #"id",  # 아이디
                       "image_width",  # 이미지 너비
                       "image_height",  # 이미지 높이
                       "box_count",  # 박스 갯수
                       "box_size_avg",  # 박스 평균 사이즈
                       "box_dist_avg",  # 박스간 거리
                       "box_normalized_count",  # 표준화된 박스갯수, 평균값에 가까운 박스들 (50%~150%)
                       "box_normalized_count (above average)",  # 평균보다 1.5배 큰 박스들
                       "box_normalized_count (below average)",  # 평균보다 0.5배 작은 박스들
                       "box_normalized_size_avg",  # 표준화된 박스갯수들로 계산한 박스크기
                       "box_normalized_dist_avg",  # 표준화된 박스갯수들로 계산한 박스간 거리
                       "normalization_warp_ratio",  # 이미지의 왜곡도 수준 - 수식: warp_ratio = box_count / box_normalized_count
                       "box_normalized_plus_dist_avg",  # 라벨값을 보정한 뒤에 계산한 박스간 거리
                       "d_box_norm_above_size",  # 평균보다 작은 박스들의 평균사이즈
                       "d_box_norm_below_size",  # 평균보다 큰 박스들의 평균사이즈
                       "d_box_normalized_points_far_distance",  # 평균적으로 큰 박스들과 평균적으로 작은 박스들간의 거리
                       "box_most_large_x",  # 가장 큰 박스의 x
                       "box_most_large_y",  # 가장 큰 박스의 y
                       "box_most_small_x",  # 가장 작은 박스의 x
                       "box_most_small_y"  # 가장 작은 박스의 y
                       ]
    box_data_total = []

    image_width = width
    image_height = height

    boxes = detected_boxes.tolist()
    points = []
    for each_box in boxes:
        x1, y1, x2, y2 = each_box
        points.append([(x2-x1)/2+x1, (y2-y1)/2+y1])

    d_box_count = 0
    d_box_size = 0
    d_box_dist_avg = 0
    temp_box_size_x = 0
    temp_box_size_y = 0

    box_most_large_x = 0
    box_most_large_y = 0
    box_most_small_x = 9999
    box_most_small_y = 9999

    for each_box in boxes:
        x1, y1, x2, y2 = each_box
        d_box_count += 1
        temp_box_size_x += x2 - x1
        temp_box_size_y += y2 - y1

        if box_most_large_y < y2 - y1:  # 세로를 기준으로
            box_most_large_x = x2 - x1
            box_most_large_y = y2 - y1
        if box_most_small_y > y2 - y1:  # 세로를 기준으로
            box_most_small_x = x2 - x1
            box_most_small_y = y2 - y1

    if d_box_count > 0:
        temp_box_size_x = temp_box_size_x / d_box_count
        temp_box_size_y = temp_box_size_y / d_box_count
        d_box_size = max(temp_box_size_x, temp_box_size_y)  # 정사각형으로 간주

    temp_box_distances = []
    for __q in range(len(points)):
        minimum = 999999
        for __w in range(len(points)):
            if __q == __w:
                continue
            center1 = (points[__q][0] + points[__q][1]) / 2
            center2 = (points[__w][0] + points[__w][1]) / 2
            result = abs(center1 - center2)
            if result < minimum:
                minimum = result
        if minimum == 0:  # 원래는 0이 나오면 안되는데 가끔 나옴. 라벨 문제로 보임.
            continue
        temp_box_distances.append(minimum)

    if len(temp_box_distances) > 0:
        d_box_dist_avg = statistics.mean(temp_box_distances)

    ##------------------------------------
    d_box_count_normalized = 0
    d_box_count_above_middle = 0
    d_box_count_below_middle = 0
    d_box_size_normalized = 0
    d_box_normalized_dist_avg = 0
    temp_box_size_x = 0  # 재계산
    temp_box_size_y = 0  # 재계산
    skip_count = -1

    box_above_list = []
    box_below_list = []
    point_above_list = []
    point_below_list = []
    pop_targets = []
    temp_box_size_x = 0
    temp_box_size_y = 0
    for each_box in boxes:  ## 다시 박스를 확인해서 평균에서 벗어나는지 확인
        skip_count += 1
        x1, y1, x2, y2 = each_box
        y_size_check = y2 - y1
        if y_size_check > d_box_size * 1.5:
            d_box_count_above_middle += 1
            box_above_list.append(boxes[skip_count])
            # point_above_list.append(points.pop(skip_count))
            skip_count -= 1
            continue
        elif y_size_check < d_box_size * 0.5:
            d_box_count_below_middle += 1
            # box_below_list.append(boxes.pop(skip_count))
            box_below_list.append(boxes[skip_count])
            # point_below_list.append(points.pop(skip_count))
            skip_count -= 1
            continue
        d_box_count_normalized += 1
        temp_box_size_x += x2 - x1
        temp_box_size_y += y2 - y1
    if d_box_count_normalized > 0:
        temp_box_size_x = temp_box_size_x / d_box_count_normalized
        temp_box_size_y = temp_box_size_y / d_box_count_normalized
        d_box_size_normalized = max(temp_box_size_x, temp_box_size_y)  # 정사각형으로 간주

    temp_box_size_x = 0
    temp_box_size_y = 0
    if d_box_size_normalized == 0:
        for each_box in box_above_list:  ## 다시 박스를 확인해서 평균에서 벗어나는지 확인
            skip_count += 1
            x1, y1, x2, y2 = each_box
            d_box_count_normalized += 1
            temp_box_size_x += x2 - x1
            temp_box_size_y += y2 - y1
        if d_box_count_normalized > 0:
            temp_box_size_x = temp_box_size_x / d_box_count_normalized
            temp_box_size_y = temp_box_size_y / d_box_count_normalized
            d_box_size_normalized = max(temp_box_size_x, temp_box_size_y)  # 정사각형으로 간주
        d_box_count_above_middle = 0

    # 포인트 부분 다시
    temp_box_distances = []
    for __q in range(len(points)):
        minimum = 999999
        for __w in range(len(points)):
            if __q == __w:
                continue
            center1 = (points[__q][0] + points[__q][1]) / 2
            center2 = (points[__w][0] + points[__w][1]) / 2
            result = abs(center1 - center2)
            if result < minimum:
                minimum = result
        if minimum == 0:  # 원래는 0이 나오면 안되는데 가끔 나옴. 라벨 문제로 보임.
            continue
        temp_box_distances.append(minimum)

    d_box_normalized_dist_avg = 0
    if len(temp_box_distances) > 0:
        d_box_normalized_dist_avg = statistics.mean(temp_box_distances)

    d_data_warp_ratio = 0
    if d_box_count > 0:  # 왜곡의 수준
        d_data_warp_ratio = 1 - (d_box_count_normalized / d_box_count)

        val_portion_to_adjust = 100 / d_box_size_normalized

    # 다시 또 재계산
    # boxes = data['boxes']
    # points = data['points'] # 새로 할 필요 없을듯
    # new_boxes = []
    new_points = []

    # 포인트 부분 재계산
    for each_point in points:
        px, py = each_point
        y_rate = py / image_height
        new_y = py * y_rate * val_portion_to_adjust
        new_x = px * y_rate * val_portion_to_adjust
        new_points.append([new_x, new_y])

    # 포인트 부분 다시 (3차)
    temp_box_distances = []
    d_box_normalized_plus_dist_avg = 0
    points = new_points
    for __q in range(len(points)):
        minimum = 999999
        for __w in range(len(points)):
            if __q == __w:
                continue
            center1 = (points[__q][0] + points[__q][1]) / 2
            center2 = (points[__w][0] + points[__w][1]) / 2
            result = abs(center1 - center2)
            if result < minimum:
                minimum = result
        if minimum == 0:  # 원래는 0이 나오면 안되는데 가끔 나옴. 라벨 문제로 보임.
            continue
        temp_box_distances.append(minimum)

    d_box_normalized_plus_dist_avg = 0
    if len(temp_box_distances) > 0:
        d_box_normalized_plus_dist_avg = statistics.mean(temp_box_distances)

    d_box_norm_above_size = 0
    d_box_norm_below_size = 0
    d_box_normalized_points_far_distance = 0

    if d_box_count > 0:
        # 박스 사이즈 계산을 평균이상이하의 평균을 계산
        temp_box_size_x = 0
        temp_box_size_y = 0
        ## 이하
        for each_box in box_below_list:
            x1, y1, x2, y2 = each_box
            d_box_count += 1
            temp_box_size_x += x2 - x1
            temp_box_size_y += y2 - y1

        if d_box_count > 0:
            temp_box_size_x = temp_box_size_x / d_box_count
            temp_box_size_y = temp_box_size_y / d_box_count
            d_box_norm_below_size = max(temp_box_size_x, temp_box_size_y)  # 정사각형으로 간주

        temp_box_size_x = 0
        temp_box_size_y = 0
        ## 이상
        for each_box in box_above_list:
            x1, y1, x2, y2 = each_box
            d_box_count += 1
            temp_box_size_x += x2 - x1
            temp_box_size_y += y2 - y1

        if d_box_count > 0:
            temp_box_size_x = temp_box_size_x / d_box_count
            temp_box_size_y = temp_box_size_y / d_box_count
            d_box_norm_above_size = max(temp_box_size_x, temp_box_size_y)  # 정사각형으로 간주

        # 포인트 다시 구하기
        skip_count = -1
        for each_box in boxes:  ## 다시 박스를 확인해서 평균에서 벗어나는지 확인
            skip_count += 1
            x1, y1, x2, y2 = each_box
            y_size_check = y2 - y1
            if y_size_check > d_box_size * 1.5:
                point_above_list.append(points.pop(skip_count))
                skip_count -= 1
                continue
            elif y_size_check < d_box_size * 0.5:
                point_below_list.append(points.pop(skip_count))
                skip_count -= 1
                continue
        # 포인트, 가장 작은 박스랑 큰 박스랑 비교
        if len(point_above_list) > 0 and len(point_below_list) > 0:
            temp_box_distances = []
            for __q in range(len(point_above_list)):
                maximum = 0
                for __w in range(len(point_below_list)):
                    if __q == __w:
                        continue
                    center1 = (point_above_list[__q][0] + point_above_list[__q][1]) / 2
                    center2 = (point_below_list[__w][0] + point_below_list[__w][1]) / 2
                    result = abs(center1 - center2)
                    if result > maximum:
                        maximum = result
                temp_box_distances.append(maximum)

            if len(temp_box_distances) > 0:
                d_box_normalized_points_far_distance = statistics.mean(temp_box_distances)

        box_data = []
        #box_data.append(1) # id는 여기서 쓰는 값이 아님
        box_data.append(image_width)
        box_data.append(image_height)
        box_data.append(d_box_count)
        box_data.append(d_box_size)
        box_data.append(d_box_dist_avg)
        box_data.append(d_box_count_normalized)
        box_data.append(d_box_count_above_middle)
        box_data.append(d_box_count_below_middle)
        box_data.append(d_box_size_normalized)
        box_data.append(d_box_normalized_dist_avg)
        box_data.append(d_data_warp_ratio)
        box_data.append(d_box_normalized_plus_dist_avg)
        box_data.append(d_box_norm_above_size)
        box_data.append(d_box_norm_below_size)
        box_data.append(d_box_normalized_points_far_distance)
        box_data.append(box_most_large_x)
        box_data.append(box_most_large_y)
        if box_most_small_x == 9999:
            box_most_small_x = 0
        if box_most_small_y == 9999:
            box_most_small_y = 0
        box_data.append(box_most_small_x)
        box_data.append(box_most_small_y)

        box_data_total.append(box_data)

    merged_data = pd.DataFrame([box_data], columns=box_field_names)
    merged_data = merged_data[model_pred.feature_names_in_]

    return merged_data

if __name__ == '__main__':
    print("RUNNING")
    app.run(debug=True, port=5505)