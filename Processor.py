import cv2
from ultralytics import YOLO
import numpy as np
from processor_config import *


def entry_point():
    pass

def q_criteria(arr):
    sorted_arr = sorted(arr)
    n = len(arr)
    mx = np.max(arr)




def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def scalar(x1, y1, x2, y2):
    return x1*x2 + y1*y2

def norm(x1, y1):
    return np.sqrt(x1**2+y1**2)

class Preprocessor:
    def __init__(self, image):
        self.bgr_image = image
        self.gray_image = None
        self.thresh_image = None

        self.gray(self.bgr_image)
        self.thresh(self.gray_image)

    def gray(self, bgr_image):
        self.gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        return self.gray_image

    def thresh(self, gray_image):
        self.thresh_image = cv2.adaptiveThreshold(
            gray_image,
            maxValue=preprocessor_thresh_adaptiveThreshold_maxValue,
            adaptiveMethod=preprocessor_thresh_adaptiveThreshold_adaptiveMethod,
            thresholdType=preprocessor_thresh_adaptiveThreshold_thresholdType,
            blockSize=preprocessor_thresh_adaptiveThreshold_blockSize,
            C=preprocessor_thresh_adaptiveThreshold_C
        )
        return self.thresh_image


class TabletFinder:
    def __init__(self, images):
        self.images = images


class CellsFinder:
    def __init__(self, images: list):
        self.images = images
        self.model = None
        self.results = None

        self.load_model()
        self.find()

    def load_model(self):
        self.model = YOLO(circles_model_path)

    def find(self):
        self.results = self.model(self.images)


class CellsPostprocessor:
    def __init__(self, result, image, image_type="ir"):
        self.result = result
        self.image = image
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image_width = self.image.shape[1]
        self.image_height = self.image.shape[0]
        self.image_type = image_type
        self.xyxyn_array = result.boxes.xyxyn.numpy()
        self.xywhn_array = result.boxes.xywhn.numpy()
        self.initial_count = len(self.xywhn_array)
        self.count = self.initial_count
        self.centers_id = set()
        self.centers = {}
        self.radii = {}
        self.rows_y = []
        self.four_shortests_distances_list = {}
        self.distance_between_centers = None
        self.radius = None
        self.angle = None
        self.rotated_image = None
        self.rotated_centers = {}
        self.table_of_centers = []
        self.pretty_table_of_centers = []
        self.table_of_blue_int = {}
        self.table_of_green_int = {}
        self.table_of_red_int = {}

        for i in range(self.count):
            self.centers_id.add(i)
            self.centers[i] = self.get_center(self.xyxyn_array[i])
            self.radii[i] = self.get_radius(self.xywhn_array[i])

        self.radius = round(np.mean(tuple(self.radii.values())))

        self.delete_miss_radii()
        self.find_distance()
        self.find_angle()
        self.rotate_image()
        #self.find_table_of_centers()

        #self.complete_table_of_centers()
        #self.get_colors()

        self._find_table_of_centers()
        self.find_table_of_colors()

    def get_center(self, xyxy):
        center = list(map(round, (np.mean(xyxy[[0, 2]])*self.image_width, np.mean((xyxy[[1, 3]]))*self.image_height)))
        return center

    def get_radius(self, xywh):
        radius = (xywh[2]*self.image_width + xywh[3]*self.image_height)/4
        return radius

    def delete_miss_radii(self):
        deleted_count = 0
        for i in range(self.initial_count):
            radii = tuple(self.radii.values())
            mean_radius = np.mean(radii)
            std_radius = np.std(radii)
            for j in self.centers_id:
                if abs(self.radii[j] - mean_radius) > 3*std_radius:
                    self.radii.pop(j)
                    self.centers.pop(j)
                    self.centers_id.remove(j)
                    deleted_count += 1
                    break
        self.count -= deleted_count
        self.radius = np.mean(tuple(self.radii.values()))

    def find_distance(self):

        shortest_distances = []
        nearest_neighbours = []
        for i, center1 in self.centers.items():
            distances = []

            for j, center2 in self.centers.items():
                if j != i:
                    distances.append([j, dist(*center1, *center2)])
            distances = sorted(distances, key=lambda x:x[1])
            #print(distances[0])
            if len(distances) > 4:
                four_shortest_distances = distances[:4]
            else:
                four_shortest_distances = distances
            four_shortest_distances_array = np.array([c[1] for c in four_shortest_distances])
            min_among_four = np.min(four_shortest_distances_array)

            j = 0
            while j < len(four_shortest_distances_array):
                if abs(four_shortest_distances_array[-1]-min_among_four)/min_among_four > cells_centers_distance_error:
                    four_shortest_distances.pop(-1)
                j += 1
            del j

            # deleted_four_count = 0
            # for j in range(4):
            #     four_shortest_distances_array = np.array([c[1] for c in four_shortest_distances])
            #     mean_among_four = np.mean(four_shortest_distances_array)
            #     std_among_four = np.std(four_shortest_distances_array)
            #
            #
            #     for k in range(4-deleted_four_count):
            #         if abs(four_shortest_distances_array[k]-mean_among_four) > 3*std_among_four:
            #             print("del 4 ", shortest_distances[k])
            #             four_shortest_distances.pop(k)
            #             deleted_four_count += 1
            #             break
            #print("four sh", four_shortest_distances)
            shortest_distances.extend(four_shortest_distances)
            self.four_shortests_distances_list[i] = four_shortest_distances
        #print(np.mean(shortest_distances))
        shortest_distances_count = len(shortest_distances)
        deleted_shortest_count = 0
        for i in range(shortest_distances_count):
            shortest_distances_array = np.array([c[1] for c in shortest_distances])
            mean_shortest = np.mean(shortest_distances_array)
            std_shortest = np.std(shortest_distances_array)
            for j in range(shortest_distances_count - deleted_shortest_count):
                if abs(shortest_distances_array[j] - mean_shortest) > 3*std_shortest:
                    #print("del sh", shortest_distances[j])
                    shortest_distances.pop(j)
                    deleted_shortest_count += 1
                    break
        #print("Deleted: ", deleted_shortest_count)
        if len(shortest_distances) > 0:
            #print(shortest_distances)
            mean_dist = np.mean([c[1] for c in shortest_distances])
            self.distance_between_centers = mean_dist
            #print(self.distance_between_centers)
            return mean_dist


    def find_angle(self):
        angs = []
        for i, center in self.centers.items():
            x0, y0 = center

            vectors = []
            for j, dst in self.four_shortests_distances_list[i]:
                if abs(dst-self.distance_between_centers)/np.mean(
                        (dst, self.distance_between_centers)) < cells_centers_distance_error:
                    vectors.append([self.centers[j][0]-x0, self.centers[j][1]-y0])

            angles = []
            #print(vectors1, vectors2)
            for j, vec in enumerate(vectors):
                angle_horizontal = np.arccos(scalar(*vec, 100, 0) / norm(*vec) / 100)
                if vec[0] <= 0 and vec[1] <= 0:
                    if angle_horizontal <= 3*np.pi/4:
                        angles.append(angle_horizontal-np.pi/2)
                    else:
                        angles.append(-np.pi+angle_horizontal)
                if vec[0] > 0 and vec[1] <= 0:
                    if angle_horizontal <= np.pi/4:
                        angles.append(angle_horizontal)
                    else:
                        angles.append(-np.pi/2+angle_horizontal)
                if vec[0] <= 0 and vec[1] > 0:
                    if angle_horizontal <= 3*np.pi/4:
                        angles.append(-angle_horizontal+np.pi/2)
                    else:
                        angles.append(np.pi-angle_horizontal)
                if vec[0] > 0 and vec[1] > 0:
                    if angle_horizontal <= np.pi/4:
                        angles.append(-angle_horizontal)
                    else:
                        angles.append(np.pi/2-angle_horizontal)
            deleted_angles_count = 0
            angles_count = len(angles)
            for j in range(angles_count):
                shortest_angles_array = np.array(angles)
                mean_angles = np.mean(shortest_angles_array)
                std_angles = np.std(shortest_angles_array)
                for k in range(angles_count - deleted_angles_count):
                    if abs(shortest_angles_array[k] - mean_angles) > 3 * std_angles:
                        print("del ang", angles[k])
                        angles.pop(k)
                        deleted_angles_count += 1
                        break
            if len(angles) > 0:
                angs.append(np.mean(angles))
            else:
                continue

        if len(angs) > 0:
            #print(angs)
            self.angle = np.mean(angs)
            #print("angs")
        return self.angle

    def rotate_image(self):
        angle = -self.angle/np.pi*180
        center = (self.image_width/2, self.image_height/2)
        rotation_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)

        for i, cnt in self.centers.items():
            x, y = cnt

            x_new, y_new = rotation_matrix@np.array([x, y, 1]).T
            self.rotated_centers[i] = [round(x_new), round(y_new)]
        #print(self.rotated_centers)
        cosine = np.abs(rotation_matrix[0, 0])
        sine = np.abs(rotation_matrix[0, 1])
        new_width = int((self.image_height * sine) + (self.image_width * cosine))
        new_height = int((self.image_height * cosine) + (self.image_width * sine))
        self.rotated_image = cv2.warpAffine(self.image, rotation_matrix, (new_width, new_height), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        #print(self.image.shape, self.rotated_image.shape)

    def _find_table_of_centers(self):
        x_min = min(tuple(self.rotated_centers.values()), key=lambda x:x[0])[0]
        x_max = max(tuple(self.rotated_centers.values()), key=lambda x:x[0])[0]
        y_min = min(tuple(self.rotated_centers.values()), key=lambda x:x[1])[1]
        y_max = max(tuple(self.rotated_centers.values()), key=lambda x:x[1])[1]
        columns_number = round((x_max-x_min)/self.distance_between_centers) + 1
        rows_number = round((y_max-y_min)/self.distance_between_centers) + 1
        table_of_centers = [[None for j in range(columns_number)] for i in range(rows_number)]
        pretty_table_of_centers = [[None for j in range(columns_number)] for i in range(rows_number)]
        extra_centers = []
        appended_count = 0
        for i, cnt in self.rotated_centers.items():
            x = cnt[0]
            y = cnt[1]
            row = round((y-y_min)/self.distance_between_centers)
            column = round((x-x_min)/self.distance_between_centers)
            if table_of_centers[row][column] == None :
                table_of_centers[row][column] = i
            else:
                extra_centers.append(i)
        #print(table_of_centers)
        for i, row in enumerate(table_of_centers):
            for j, center in enumerate(row):
                if center is None:
                    min_dist = -1
                    c = None
                    approximate_x = x_min + j*self.distance_between_centers
                    approximate_y = y_min + i*self.distance_between_centers
                    for cnt in extra_centers:
                        dst = dist(*self.rotated_centers[cnt], approximate_x, approximate_y)
                        if dst < min_dist and dst/self.distance_between_centers < cells_centers_left_distance_error:
                            c = cnt
                            min_dist = dst
                    if c is not None:
                        table_of_centers[i][j] = c
                    else:
                        x_coords = []
                        y_coords = []
                        for k in range(rows_number):

                            if table_of_centers[k][j] is not None:
                                x_coords.append(self.rotated_centers[table_of_centers[k][j]][0])
                                y_coords.append(self.rotated_centers[table_of_centers[k][j]][1])
                        if len(x_coords)*len(y_coords) > 1:

                            mod = np.polyfit(y_coords, x_coords, 1)
                            pol = np.poly1d(mod)
                            x_new = pol(approximate_y)
                            id_new = self.initial_count + appended_count + 1
                            self.centers_id.add(id_new)
                            table_of_centers[i][j] = id_new
                            self.rotated_centers[id_new] = [round(x_new), round(approximate_y)]
                            appended_count += 1
                        else:
                            id_new = self.initial_count + appended_count + 1
                            self.centers_id.add(id_new)
                            table_of_centers[i][j] = id_new
                            self.rotated_centers[id_new] = [round(approximate_x), round(approximate_y)]
                            appended_count += 1

        #print(table_of_centers)
        self.table_of_centers = table_of_centers

    def find_table_of_colors(self):
        table_of_blue_int = {}
        table_of_green_int = {}
        table_of_red_int = {}
        rng = round(cells_color_radius_fraction*self.radius)
        for row in self.table_of_centers:
            for cnt in row:
                x, y = self.rotated_centers[cnt]
                raw_area_1 = self.rotated_image[y - rng:y + rng, x - rng:x + rng]
                raw_area = raw_area_1.reshape((raw_area_1.shape[0] * raw_area_1.shape[1], 3))
                b, g, r = list(map(round, np.mean(raw_area, axis=0)))
                table_of_blue_int[cnt] = b
                table_of_green_int[cnt] = g
                table_of_red_int[cnt] = r

        self.table_of_blue_int = table_of_blue_int
        self.table_of_green_int = table_of_green_int
        self.table_of_red_int = table_of_red_int








    def find_table_of_centers(self):
        enumerated_centers = self.centers
        y_sorted_centers = sorted(enumerated_centers, key=lambda x: x[1][1])

        i = 1
        table_of_centers = []
        mean_distances = []
        while i < self.count:
            j = 0
            # print(i)
            row_data_list = []
            while i + j < self.count and abs(y_sorted_centers[i - 1 + j][1][1] - y_sorted_centers[i + j][1][1]) / \
                    y_sorted_centers[i + j][1][1] < cells_centers_relative_error:
                row_data_list.append(y_sorted_centers[i + j])
                j += 1

                # print(i+j)

            row = [y_sorted_centers[i - 1], ]
            row = sorted(row + row_data_list, key=lambda x: x[1][0])
            self.rows_y.append(np.mean(np.array([row[i][1][1] for i in range(len(row))])))
            raw_distances = []
            for k in range(1, len(row)):
                dist = abs(row[k - 1][1][0] - row[k][1][0])
                raw_distances.append(dist)
            mean_dist = np.array(raw_distances).mean()
            mean_distances.append(mean_dist)
            distances = [round(raw_distances[i]/mean_dist) for i in range(len(raw_distances))]
            appended_count = 0
            for k in range(len(distances)):
                for l in range(distances[k] - 1):
                    row.insert(k + 1 + appended_count, None)
                    appended_count += 1

            table_of_centers.append(row)

            i += j + 1
            # print()
        # print(np.array(table_of_circles, dtype=object))
        if len(mean_distances) > 0:
            self.distance_between_centers = np.array(mean_distances).mean()
        self.table_of_centers = table_of_centers
        for r in table_of_centers:
            print(r)

    
    def complete_table_of_centers(self):
        max_len_value = -1
        max_len_index = -1
        for i, row in enumerate(self.table_of_centers):
            if len(row) > max_len_value:
                max_len_value = len(row)
                max_len_index = i
        appended_count = 0
        for i, row in enumerate(self.table_of_centers):
            current_len = len(row)
            len_difference = max_len_value - current_len
            if len_difference > 0:
                #print("ok")

                if round(row[0][1][0]) > round(self.table_of_centers[max_len_index][0][1][0]):
                    left_extra_points_count = round((row[0][1][0]-self.table_of_centers[max_len_index][0][1][0])
                                                    / self.distance_between_centers)
                    for j in range(left_extra_points_count):
                        row.insert(0, None)
                        self.count += 1
                if round(row[-1][1][0]) < round(self.table_of_centers[max_len_index][-1][1][0]):
                    right_extra_points_count = round((self.table_of_centers[max_len_index][-1][1][0] - row[-1][1][0])
                                                    / self.distance_between_centers)
                    #print(i, right_extra_points_count)
                    for j in range(right_extra_points_count):
                        row.append(None)
                        self.count += 1

        for k, row in enumerate(self.table_of_centers):
            anchor = None
            anchor_index = None
            for i, center in enumerate(row):
                if center is not None:
                    anchor = center[1][0]
                    anchor_index = i
                    break
            for i, center in enumerate(row):
                if center is None:
                    if i < anchor_index:
                        row[i] = [self.count + k,
                                  (row[anchor_index][1][0] - self.distance_between_centers * (anchor_index - i),
                                   self.rows_y[k])]
                    else:
                        row[i] = [self.count + k,
                                  (row[anchor_index][1][0] + self.distance_between_centers * (i-anchor_index),
                                   self.rows_y[k])]
                else:
                    anchor = center[1][0]
                    anchor_index = i
                    """
                    j = i + 1
                    while j < len(row) and row[j] is None:
                        j += 1
                    row[i] = [self.count+k, (row[j][1][0]-self.distance_between_centers*(j-i), self.rows_y[k])]
                    """


        for k, row in enumerate(self.table_of_centers):
            for i, center in enumerate(row):

                center[1] = list(map(int, np.around(center[1])))

    def get_colors(self):
        if len(self.radii) > 0:
            radii = [self.radii[i][1] for i in range(len(self.radii))]
            self.radius = np.mean(radii)
            print(self.radius)
            rng = round(self.radius*cells_color_radius_fraction)
            work_area = []
            # добавить другие типы
            if self.image_type == "ir":
                image = self.gray_image
            else:
                image = self.image
        for row in self.table_of_centers:
            colors_row = []
            for center in row:
                x, y = center[1]
                raw_area_1 = image[y-rng:y+rng, x-rng:x+rng]
                if len(image.shape) == 3:
                    raw_area = raw_area_1.reshape((raw_area_1.shape[0]*raw_area_1.shape[1], 3))
                    colors_row.append([center[0], list(map(round, np.mean(raw_area, axis=0)))])
                else:
                    raw_area = raw_area_1.reshape((raw_area_1.shape[0] * raw_area_1.shape[1], ))
                    colors_row.append([center[0], round(np.mean(raw_area, axis=0))])
                #cv2.imshow(f"{row}", raw_area)

            self.table_of_colors.append(colors_row)








class Validator:
    def __init__(self, result):
        self.bbox = result.bbox
        self.conf = result.cnf