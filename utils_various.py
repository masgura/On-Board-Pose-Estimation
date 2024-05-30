import shutil
import os
import math
import numpy as np
from numpy import array as npa
import matplotlib as mpl



earthBGStart_ID = 7500

def get_coords_from_landmark_label(landmarks, point_label):
    is_my_label_here = [point['label'] == point_label for point in landmarks]
    my_label_idx = [idx for idx, myBool in enumerate(is_my_label_here) if myBool][0]
    return landmarks[my_label_idx]['r_B']

class Wireframe:
    """"
    Utility class that defines landmarks and coordinates
    of other points from the wireframe model.
    """

    # Define 11 landmark points (body coordinates)
    landmarks = [
    {'label': 'B1', 'r_B': [-0.37, 0.30, 0]},
    {'label': 'B2', 'r_B': [-0.37, -0.26, 0]},
    {'label': 'B3', 'r_B': [0.37, -0.26, 0]},
    {'label': 'B4', 'r_B': [0.37, 0.30, 0]},
    {'label': 'S1', 'r_B': [-0.37, 0.38, 0.32]},
    {'label': 'S2', 'r_B': [-0.37, -0.38, 0.32]},
    {'label': 'S3', 'r_B': [0.37, -0.38, 0.32]},
    {'label': 'S4', 'r_B': [0.37, 0.38, 0.32]},
    {'label': 'A1', 'r_B': [-0.54, 0.49, 0.255]},
    {'label': 'A2', 'r_B': [0.31, -0.56, 0.255]},
    {'label': 'A3', 'r_B': [0.54, 0.49, 0.255]}
        ]

    landmark_mat = np.column_stack( [point['r_B'] for point in landmarks] )

    # Top of the main body (not used as landmarks)
    topMainBody = [
    {'label': 'T1', 'r_B': [-0.37, 0.30, 0.305]},
    {'label': 'T2', 'r_B': [-0.37, -0.26, 0.305]},
    {'label': 'T3', 'r_B': [0.37, -0.26, 0.305]},
    {'label': 'T4', 'r_B': [0.37, 0.30, 0.305]}
        ]
    topMainBody_mat = np.column_stack( [point['r_B'] for point in topMainBody] )

    # Antenna clamps
    antClamps = [
    {'label': 'Ac1', 'r_B': [-0.23, 0.3, 0.255]},
    {'label': 'Ac2', 'r_B': [0.31, -0.26, 0.255]},
    {'label': 'Ac3', 'r_B': [0.23, 0.3, 0.255]}
        ]
    antClamps_mat = np.column_stack( [point['r_B'] for point in antClamps] )

    body_center = [0, 0, (0+get_coords_from_landmark_label(landmarks, 'S1')[2])/2]

    # We compute L_c as the diagonal length of the solar panel
    k1 = 1.05  # a constant empirically tuned, by testing on outliers
    charact_length = k1 * np.linalg.norm( npa(get_coords_from_landmark_label(landmarks, 'S1'))
                                     -
                                     npa(get_coords_from_landmark_label(landmarks, 'S3')) )


# reference points @ body frame (for drawing axes)
p_axes = np.array([[0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

BB_enlarge = .1  # i.e. 10% larger than minimum rectangle
max_ROI_size = 416  # [px]

class Camera:
    """" Utility class for accessing camera parameters. """

    fx = 0.0176  # focal length [m]
    fy = 0.0176  # focal length [m]
    nu = 1920  # no. horizontal pixels
    nv = 1200  # no. of vertical pixels
    ppx = 5.86e-6  # horizontal pixel pitch [m/px]
    ppy = ppx      # vertical pixel pitch [m/px]
    fpx = fx / ppx  # horizontal focal length [px]
    fpy = fy / ppy  # vertical focal length [px]
    k = [[fpx,   0, nu / 2], # camera intrinsic matrix
         [0,   fpy, nv / 2],
         [0,     0,      1]]
    K = npa(k)

    # angular size of the camera's FoV, considering the diagonal aperture [deg]
    FOV_diagonal_deg = 180*math.pi * 2 * math.atan( ppx * math.sqrt(nu**2 + nv**2) / (2*fx) )

def project3Dto2D(dcm_CB, t_CB, r_B_mat):
    """ Projecting points to image frame.
        q_CB:       quaternion representing rotation: camera_frame --> Tango princ. axes
        t_CB:       camera2body_translation
        r_B_mat:    body coordinates of SC points (stacked column by column)
    """

    points_body = np.concatenate( ( r_B_mat, np.ones((1,r_B_mat.shape[1])) ), axis=0 )

    # transformation to camera frame
    pose_mat = np.hstack( ( dcm_CB.T, np.expand_dims(t_CB, 1) ) )
    p_cam = np.dot(pose_mat, points_body)

    # getting homogeneous coordinates
    points_camera_frame = p_cam / p_cam[2]

    # projection to image plane
    points_image_plane = Camera.K.dot(points_camera_frame)

    x, y = (points_image_plane[0], points_image_plane[1])
    return x, y



def quat2dcm(q):

    """ Convert quaternion [q4 q1 q2 q3] to Direction Cosine Matrix. """

    # normalizing quaternion
    q = q/np.linalg.norm(q)

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    dcm = np.zeros((3, 3))

    dcm[0, 0] = 2 * q0**2 - 1 + 2 * q1**2
    dcm[1, 1] = 2 * q0**2 - 1 + 2 * q2**2
    dcm[2, 2] = 2 * q0**2 - 1 + 2 * q3**2

    dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2

    dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1

    dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1

    return dcm

def dcm2quat(dcm):
    """
    Based on the method described here:
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    """
    if dcm[2, 2] < 0:
        if dcm[0, 0] > dcm[1, 1]:
            t = 1 + dcm[0, 0] - dcm[1, 1] - dcm[2, 2]
            q = [dcm[1, 2] - dcm[2, 1], t, dcm[0, 1] + dcm[1, 0], dcm[2, 0] + dcm[0, 2]]
        else:
            t = 1 - dcm[0, 0] + dcm[1, 1] - dcm[2, 2]
            q = [dcm[2, 0] - dcm[0, 2], dcm[0, 1] + dcm[1, 0], t, dcm[1, 2] + dcm[2, 1]]
    else:
        if dcm[0, 0] < -dcm[1, 1]:
            t = 1 - dcm[0, 0] - dcm[1, 1] + dcm[2, 2]
            q = [dcm[0, 1] - dcm[1, 0], dcm[2, 0] + dcm[0, 2], dcm[1, 2] + dcm[2, 1], t]
        else:
            t = 1 + dcm[0, 0] + dcm[1, 1] + dcm[2, 2]
            q = [t, dcm[1, 2] - dcm[2, 1], dcm[2, 0] - dcm[0, 2], dcm[0, 1] - dcm[1, 0]]

    q = np.array(q)
    q *= 0.5 / math.sqrt(t)
    return q

def dcm2euler(dcm):
    """
    Converts Direction Cosine Matrix to corresponding Euler angles representation.

    th_x, th_y, th_z [deg] are computed as the rotation angles
    about the x,y,x axes, respectively, whose sign is given by the SCREW rule
    """
    # assert(isRotationMatrix(dcm))

    # N.B. I used math instead of numpy since it is a little faster

    sy = math.sqrt(dcm[0, 0] * dcm[0, 0] + dcm[1, 0] * dcm[1, 0])
    singular = sy < 1e-6

    if not singular:
       th_x = math.atan2( dcm[2, 1], dcm[2, 2])
       th_y = math.atan2(-dcm[2, 0], sy)
       th_z = math.atan2( dcm[1, 0], dcm[0, 0])

    else:
        th_x = math.atan2(-dcm[1, 2], dcm[1, 1])
        th_y = math.atan2(-dcm[2, 0], sy)
        th_z = 0

    return -npa([th_x, th_y, th_z]) * 180/math.pi  # [deg]



