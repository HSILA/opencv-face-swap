import cv2
import numpy as np
import json

def find_index_landmark(array):
    index = None
    for num in array[0]:
        index = num
        break
    return index

def get_landmark_points(json_path):
    with open(json_path, 'r') as f:
        points = json.load(f)
    return points['point']

def get_triangles(points):
    convexhull = cv2.convexHull(np.array(points, dtype=np.int32))
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(points)
    triangles = subdiv.getTriangleList().astype(np.int32)
    return triangles, convexhull

def create_mask(triangle_points, image):
    triangle = np.array(triangle_points, dtype=np.int32)
    bounding_rect = cv2.boundingRect(triangle)
    (x, y, w, h) = bounding_rect
    cropped_triangle = image[y:y+h, x:x+w]
    mask = np.zeros((h, w), np.uint8)

    points = np.array((
                    (triangle_points[0][0] - x, triangle_points[0][1] - y),
					(triangle_points[1][0] - x, triangle_points[1][1] - y),
					(triangle_points[2][0] - x, triangle_points[2][1] - y)), np.int32)

    cv2.fillConvexPoly(mask, points, 255)
    roi = cv2.bitwise_and(cropped_triangle, cropped_triangle, mask=mask)
    return roi, points, bounding_rect

def warp(face_path, json1, body_path, json2):

    face = cv2.imread(face_path)
    body = cv2.imread(body_path)

    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    body_gray = cv2.cvtColor(body, cv2.COLOR_BGR2GRAY)

    body_new_face = np.zeros_like(body)

    points1 = get_landmark_points(json1)
    points1_np = np.array(points1, dtype=np.int32)
    points2 = get_landmark_points(json2)
    
    triangles1, _ = get_triangles(points1)
    _, convexhull2 = get_triangles(points2)
    indexes_triangles = []

    for triangle in triangles1:
        v1 = (triangle[0], triangle[1])
        v2 = (triangle[2], triangle[3])
        v3 = (triangle[4], triangle[5])

        index_v1 = np.where((points1_np == v1).all(axis=1))
        index_v1 = find_index_landmark(index_v1)
        index_v2 = np.where((points1_np == v2).all(axis=1))
        index_v2 = find_index_landmark(index_v2)
        index_v3 = np.where((points1_np == v3).all(axis=1))
        index_v3 = find_index_landmark(index_v3)

        # Saves coordinates if the triangle exists and has 3 vertices
        if index_v1 is not None and index_v2 is not None and index_v3 is not None:
            indexes_triangles.append([index_v1, index_v2, index_v3])

    for triangle_index in indexes_triangles:
        tr1_v1 = points1[triangle_index[0]]
        tr1_v2 = points1[triangle_index[1]]
        tr1_v3 = points1[triangle_index[2]]

        cropped_triangle, face_tri, _ = create_mask((tr1_v1, tr1_v2, tr1_v3), face)

        # body triangulation
        tr2_v1 = points2[triangle_index[0]]
        tr2_v2 = points2[triangle_index[1]]
        tr2_v3 = points2[triangle_index[2]]

        _, body_tri, rect = create_mask((tr2_v1, tr2_v2, tr2_v3), body)
        x, y, w, h = rect

        # warping triangles
        face_tri = np.float32(face_tri)
        body_tri = np.float32(body_tri)
        M = cv2.getAffineTransform(face_tri, body_tri)
        warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
        
        # creating new face
        body_new_face_rect_area = body_new_face[y: y + h, x: x + w]
        body_new_face_rect_area = cv2.add(body_new_face_rect_area, warped_triangle)
        body_new_face[y: y + h, x: x + w] = body_new_face_rect_area

    body_face_mask = np.zeros_like(body_gray)
    body_head_mask = cv2.fillConvexPoly(body_face_mask, convexhull2, 255)
    body_face_mask = cv2.bitwise_not(body_head_mask)

    body_faceless = cv2.bitwise_and(body, body, mask=body_face_mask)
    result = cv2.add(body_faceless, body_new_face)

    cv2.imshow('Result', result)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    json_0 = './images/obama.json'
    json_1 = './images/clinton.json'
    image_0 = './images/obama.png'
    image_1 = './images/clinton.png'
    warp(image_0, json_0, image_1, json_1)
