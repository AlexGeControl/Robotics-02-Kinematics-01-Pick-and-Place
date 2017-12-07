#!/usr/bin/env python

from sympy import *
from time import time
from mpmath import radians
import tf

# Utilities:
def denavit_hartenburg_matrix(alpha, a, theta, d):
    """ Generate symbolic Denavit-Hartenberg matrix according to Craig J.J. convention

    Args:
    alpha: sympy symbol, Rotation along x_{n-1} from z_{i-1} to z_{i}
    a: sympy symbol, Translation along x_{n-1} from z_{i-1} to z_{i}
    d: sympy symbol, Translation along z_{i} from x_{i-1} to x_{i}
    theta: sympy symbol, Rotation along z_{i} from x_{i-1} to x_{i}

    Returns:
    sympy Matrix, The generated symbolic Denavit-Hartenberg matrix
    """
    # Generate composite matrix Rx_{i-1}(alpha)*Tx_{i-1}(a)*Rz_{i}(theta)*Tz_{i}(d)
    T = Matrix(
        [
            [            cos(theta),           -sin(theta),           0,             a],
            [ sin(theta)*cos(alpha), cos(theta)*cos(alpha), -sin(alpha), -sin(alpha)*d],
            [ sin(theta)*sin(alpha), cos(theta)*sin(alpha),  cos(alpha),  cos(alpha)*d],
            [                     0,                     0,           0,             1]
        ]
    )

    return T

def rotation_x_matrix(roll):
    """ Generate symbolic rotation matrix along x axis

    Args:
    roll: sympy symbol, Rotation along x axis

    Returns:
    sympy Matrix, the generated symbolic rotation matrix along x axis
    """
    T = Matrix(
        [
            [ 1,          0,          0, 0],
            [ 0, +cos(roll), -sin(roll), 0],
            [ 0, +sin(roll), +cos(roll), 0],
            [ 0,          0,          0, 1]
        ]
    )

    return T

def rotation_y_matrix(pitch):
    """ Generate symbolic rotation matrix along y axis

    Args:
    pitch: sympy symbol, Rotation along y axis

    Returns:
    sympy Matrix, the generated symbolic rotation matrix along y axis
    """
    T = Matrix(
        [
            [+cos(pitch), 0, +sin(pitch), 0],
            [          0, 1,           0, 0],
            [-sin(pitch), 0, +cos(pitch), 0],
            [          0, 0,           0, 1]
        ]
    )

    return T

def rotation_z_matrix(yaw):
    """ Generate symbolic rotation matrix along z axis

    Args:
    yaw: sympy symbol, Rotation along z axis

    Returns:
    sympy Matrix, the generated symbolic rotation matrix along z axis
    """
    T = Matrix(
        [
            [+cos(yaw), -sin(yaw), 0, 0],
            [+sin(yaw), +cos(yaw), 0, 0],
            [        0,         0, 1, 0],
            [        0,         0, 0, 1]
        ]
    )

    return T

def rotation_rpy_matrix(roll, pitch, yaw):
    """ Generate rotation matrix from Euler angles

    Args:
    roll: sympy symbol, Rotation along x axis
    pitch: sympy symbol, Rotation along y axis
    yaw: sympy symbol, Rotation along z axis

    Returns:
    sympy Matrix, the generated symbolic rotation matrix from Euler angles
    """
    # yaw:
    R_z = Matrix(
        [
            [+cos(yaw), -sin(yaw), 0],
            [+sin(yaw), +cos(yaw), 0],
            [        0,         0, 1]
        ]
    )
    # pitch:
    R_y = Matrix(
        [
            [+cos(pitch), 0, +sin(pitch)],
            [          0, 1,           0],
            [-sin(pitch), 0, +cos(pitch)]
        ]
    )
    # roll:
    R_x = Matrix(
        [
            [ 1,          0,          0],
            [ 0, +cos(roll), -sin(roll)],
            [ 0, +sin(roll), +cos(roll)]
        ]
    )

    # compose:
    R = R_z * R_y * R_x

    return R

class IKServer():
    """ Inverse kinematics solution server
    """
    def __init__(self):
        #
        # forward kinematics:
        #

        # a. DH params:
        # alpha:
        (alpha0,alpha1,alpha2,alpha3,alpha4,alpha5,alpha6) = symbols('alpha0:7')
        # a:
        (    a0,    a1,    a2,    a3,    a4,    a5,    a6) = symbols('a0:7')
        # theta:
        (theta1,theta2,theta3,theta4,theta5,theta6,theta7) = symbols('theta1:8')
        # d:
        (    d1,    d2,    d3,    d4,    d5,    d6,    d7) = symbols('d1:8')

        # b. config:
        #
        # Key params for config are determined as follows:
        # 1. d1: jo1_z + jo2_z =  0.33 +  0.42 = 0.75;
        # 2. a1:         jo2_x =  0.35;
        # 3. a2:         jo3_z =  1.25;
        # 4. a3:       - jo4_z = 0.054;
        # 5. d4: jo4_x + jo5_x =  0.96 +  0.54 = 1.50;
        # 6. d7: jo6_x + jog_x = 0.193 +  0.11 = 0.303
        #
        config = {
            alpha0:     0, a0:      0, d1:  0.75, theta1:         theta1,
            alpha1: -pi/2, a1: +0.350, d2:     0, theta2: -pi/2 + theta2,
            alpha2:     0, a2: +1.250, d3:     0, theta3:         theta3,
            alpha3: -pi/2, a3: -0.054, d4:  1.50, theta4:         theta4,
            alpha4: +pi/2, a4:      0, d5:     0, theta5:         theta5,
            alpha5: -pi/2, a5:      0, d6:     0, theta6:         theta6,
            alpha6:     0, a6:      0, d7: 0.303, theta7:              0,
        }

        # c. transformation:
        T_01 = simplify(
            denavit_hartenburg_matrix(alpha0, a0, theta1, d1).subs(config)
        )
        T_12 = simplify(
            denavit_hartenburg_matrix(alpha1, a1, theta2, d2).subs(config)
        )
        T_23 = simplify(
            denavit_hartenburg_matrix(alpha2, a2, theta3, d3).subs(config)
        )
        T_34 = simplify(
            denavit_hartenburg_matrix(alpha3, a3, theta4, d4).subs(config)
        )
        T_45 = simplify(
            denavit_hartenburg_matrix(alpha4, a4, theta5, d5).subs(config)
        )
        T_56 = simplify(
            denavit_hartenburg_matrix(alpha5, a5, theta6, d6).subs(config)
        )
        T_67 = simplify(
            denavit_hartenburg_matrix(alpha6, a6, theta7, d7).subs(config)
        )
        T_7G = simplify(
            rotation_z_matrix(pi)*rotation_y_matrix(-pi/2)
        )

        T_02 = simplify(
            T_01 * T_12
        )
        T_03 = simplify(
            T_02 * T_23
        )
        T_36 = simplify(
            T_34 * T_45 * T_56
        )
        T_6G = simplify(
            T_67 * T_7G
        ).evalf()
        T_0G = simplify(
            T_03 * T_36 * T_6G
        )

        # d. key elements inverse kinematics:
        # 1. wrist center:
        (px, py, pz) = symbols('px py pz')
        (roll, pitch, yaw) = symbols('roll pitch yaw')

        p_6G = T_6G[0:3,3]
        self.R_6G = T_6G[0:3,0:3]
        t = simplify(
            (self.R_6G.T)*p_6G
        )
        p = simplify(
            Matrix(
                [[px, py, pz]]
            ).T
        )
        self.R = simplify(
            rotation_rpy_matrix(
                roll,
                pitch,
                yaw
            )
        )

        self.wc = simplify(
            p - (self.R)*t
        )

        # 2. position
        (wcx, wcy, wcz) = symbols('wcx wcy wcz')
        wc = Matrix(3, 1, [wcx, wcy, wcz])

        self.A = simplify(
            a2.subs(config)
        )
        self.C = simplify(
            (sqrt(a3**2 + d4**2)).subs(config)
        )
        self.bias = simplify(
            atan2(-a3, d4).subs(config)
        ).evalf()

        self.o = simplify(
            T_02[0:3,3].subs(
                {
                    theta2: 0
                }
            )
        )
        c3 = cos(theta3 + self.bias)
        s3 = sin(theta3 + self.bias)
        ma = self.A - self.C*s3
        mc = self.C*c3
        self.MA = Matrix(
            [
                [ma, -mc],
                [mc,  ma]
            ]
        )
        self.Mb = Matrix(
            [
                [
                    wc[2,0]-d1,
                    sqrt(wc[0,0]**2 + wc[1,0]**2)-a1
                ]
            ]
        ).T.subs(config)

        # 3. orientation:
        self.R_03 = T_03[0:3,0:3]

        print(T_36[0,2], T_36[1,2], T_36[2,2], T_36[1,0], T_36[1,1])

        # 4. end-effector:
        self.ee = T_0G[0:3,3]
    #
    # inverse kinematics
    #
    def _solve_wc(self, pose):
        """ Solve wrist center position
        """
        wc = self.wc.evalf(
            subs = pose
        )
        R = (self.R * (self.R_6G.T)).evalf(
            subs = pose
        )
        return (wc, R)

    def _solve_position(self, pose):
        """ Solve theta1, theta2 & theta3
        """
        wc = self.wc.evalf(
            subs = pose
        )

        # 1. theta1
        theta1 = atan2(
            wc[1,0],
            wc[0,0]
        ).evalf()

        # 2. theta3:
        o = self.o.evalf(
            subs = dict(
                theta1 = theta1
            )
        )
        B_squared = (((wc-o).T)*(wc-o))[0,0]
        beta = acos(
            (self.A**2 + self.C**2 - B_squared)/(2*self.A*self.C)
        )
        theta3 = (
            pi/2 - beta - self.bias
        ).evalf()

        # 3. theta2:
        MA = self.MA.evalf(
            subs = dict(
                theta3 = theta3
            )
        )
        Mb = self.Mb.evalf(
            subs = dict(
                wcx = wc[0,0],
                wcy = wc[1,0],
                wcz = wc[2,0],
                theta1 = theta1
            )
        )
        Mx = MA.inv("LU")*Mb
        theta2 = atan2(Mx[1,0], Mx[0,0])

        R_03 = self.R_03.evalf(
            subs = dict(
                theta1 = theta1,
                theta2 = theta2,
                theta3 = theta3
            )
        )

        return (theta1, theta2, theta3, R_03)

    def _solve_orientation(self, R_36):
        """ Solve theta4, theta5, theta6
        """
        theta4 = atan2(
            R_36[2,2],
            -R_36[0,2]
        )
        theta5 = atan2(
            sqrt(R_36[0,2]**2 + R_36[2,2]**2),
            R_36[1,2]
        )
        theta6 = atan2(
            -R_36[1,1],
            R_36[1,0]
        )

        return (theta4, theta5, theta6)

    def solve(self, pose):
        """ Solve inverse kinematics
        """
        # 1. wrist center:
        (wc, R) = self._solve_wc(pose);

        # 2. position:
        (theta1, theta2, theta3, R_03) = self._solve_position(pose)

        (theta4, theta5, theta6) = self._solve_orientation(
            (R_03.T)*R
        )

        ee = self.ee.evalf(
            subs = dict(
                theta1 = theta1,
                theta2 = theta2,
                theta3 = theta3,
                theta4 = theta4,
                theta5 = theta5,
                theta6 = theta6,
            )
        )

        # format:
        wc = [wc[0,0], wc[1,0], wc[2,0]]
        ee = [ee[0,0], ee[1,0], ee[2,0]]

        return (wc, theta1, theta2, theta3, theta4, theta5, theta6, ee)

    def _handle_calculate_IK(self, req):
        """ Handle inverse kinematics solution request
        """
        # prompt:
        rospy.loginfo(
            "Received %s eef-poses from the plan",
            len(req.poses)
        )

        # pre-assumption check:
        if len(req.poses) < 1:
            print "No valid poses received"
            return -1
        else:
            # Initialize service response
            joint_trajectory_list = []
            for x in xrange(0, len(req.poses)):
                #
                # Extract end-effector position and orientation from request
                #
                # a. px,py,pz = end-effector position
                px = req.poses[x].position.x
                py = req.poses[x].position.y
                pz = req.poses[x].position.z

                # b. roll, pitch, yaw = end-effector orientation
                (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
                    [
                        req.poses[x].orientation.x,
                        req.poses[x].orientation.y,
                        req.poses[x].orientation.z,
                        req.poses[x].orientation.w
                    ]
                )

                # c. format as pose:
                pose = dict(
                    px = px,
                    py = py,
                    pz = pz,
                    roll = roll,
                    pitch = pitch,
                    yaw = yaw
                )

                # calculate joint angles using Geometric IK method
                joint_trajectory_point = JointTrajectoryPoint()

                (
                    wc,
                    theta1,
                    theta2,
                    theta3,
                    theta4,
                    theta5,
                    theta6,
                    ee
                ) = self.solve(pose)

                joint_trajectory_point.positions = [theta1, theta2, theta3, theta4, theta5, theta6]
                # Populate response for the IK request
                joint_trajectory_list.append(joint_trajectory_point)

            rospy.loginfo("length of Joint Trajectory List: %s" % len(joint_trajectory_list))

            return CalculateIKResponse(joint_trajectory_list)

'''
Format of test case is [ [[EE position],[EE orientation as quaternions]],[WC location],[joint angles]]
You can generate additional test cases by setting up your kuka project and running `$ roslaunch kuka_arm forward_kinematics.launch`
From here you can adjust the joint angles to find thetas, use the gripper to extract positions and orientation (in quaternion xyzw) and lastly use link 5
to find the position of the wrist center. These newly generated test cases can be added to the test_cases dictionary.
'''

test_cases = {1:[[[2.16135,-1.42635,1.55109],
                  [0.708611,0.186356,-0.157931,0.661967]],
                  [1.89451,-1.44302,1.69366],
                  [-0.65,0.45,-0.36,0.95,0.79,0.49]],
              2:[[[-0.56754,0.93663,3.0038],
                  [0.62073, 0.48318,0.38759,0.480629]],
                  [-0.638,0.64198,2.9988],
                  [-0.79,-0.11,-2.33,1.94,1.14,-3.68]],
              3:[[[-1.3863,0.02074,0.90986],
                  [0.01735,-0.2179,0.9025,0.371016]],
                  [-1.1669,-0.17989,0.85137],
                  [-2.99,-0.12,0.94,4.06,1.29,-4.12]],
              4:[],
              5:[]}


def test_code(test_case):
    ## Set up code
    ## Do not modify!
    x = 0
    class Position:
        def __init__(self,EE_pos):
            self.x = EE_pos[0]
            self.y = EE_pos[1]
            self.z = EE_pos[2]
    class Orientation:
        def __init__(self,EE_ori):
            self.x = EE_ori[0]
            self.y = EE_ori[1]
            self.z = EE_ori[2]
            self.w = EE_ori[3]

    position = Position(test_case[0][0])
    orientation = Orientation(test_case[0][1])

    class Combine:
        def __init__(self,position,orientation):
            self.position = position
            self.orientation = orientation

    comb = Combine(position,orientation)

    class Pose:
        def __init__(self,comb):
            self.poses = [comb]

    req = Pose(comb)

    ik_server = IKServer()

    start_time = time()

    ########################################################################################
    ##
    # a. px,py,pz = end-effector position
    px = req.poses[x].position.x
    py = req.poses[x].position.y
    pz = req.poses[x].position.z

    # b. roll, pitch, yaw = end-effector orientation
    (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
        [
            req.poses[x].orientation.x,
            req.poses[x].orientation.y,
            req.poses[x].orientation.z,
            req.poses[x].orientation.w
        ]
    )

    # c. format as pose:
    pose = dict(
           px = px,
           py = py,
           pz = pz,
         roll = roll,
        pitch = pitch,
          yaw = yaw
    )

    ## Insert IK code here!
    (
        wc,
        theta1,
        theta2,
        theta3,
        theta4,
        theta5,
        theta6,
        ee
    ) = ik_server.solve(pose)
    ##
    ########################################################################################

    ########################################################################################
    ## For additional debugging add your forward kinematics here. Use your previously calculated thetas
    ## as the input and output the position of your end effector as your_ee = [x,y,z]
    ## End your code input for forward kinematics here!
    ########################################################################################

    ## For error analysis please set the following variables of your WC location and EE location in the format of [x,y,z]
    your_wc = wc # <--- Load your calculated WC values in this array
    your_ee = ee # <--- Load your calculated end effector value from your forward kinematics
    ########################################################################################

    ## Error analysis
    print ("\nTotal run time to calculate joint angles from pose is %04.4f seconds" % (time()-start_time))

    # Find WC error
    if not(sum(your_wc)==3):
        wc_x_e = abs(your_wc[0]-test_case[1][0])
        wc_y_e = abs(your_wc[1]-test_case[1][1])
        wc_z_e = abs(your_wc[2]-test_case[1][2])
        wc_offset = sqrt(wc_x_e**2 + wc_y_e**2 + wc_z_e**2)
        print ("\nWrist error for x position is: %04.8f" % wc_x_e)
        print ("Wrist error for y position is: %04.8f" % wc_y_e)
        print ("Wrist error for z position is: %04.8f" % wc_z_e)
        print ("Overall wrist offset is: %04.8f units" % wc_offset)

    # Find theta errors
    t_1_e = abs(theta1-test_case[2][0])
    t_2_e = abs(theta2-test_case[2][1])
    t_3_e = abs(theta3-test_case[2][2])
    t_4_e = abs(theta4-test_case[2][3])
    t_5_e = abs(theta5-test_case[2][4])
    t_6_e = abs(theta6-test_case[2][5])
    print ("\nTheta 1 error is: %04.8f" % t_1_e)
    print ("Theta 2 error is: %04.8f" % t_2_e)
    print ("Theta 3 error is: %04.8f" % t_3_e)
    print ("Theta 4 error is: %04.8f" % t_4_e)
    print ("Theta 5 error is: %04.8f" % t_5_e)
    print ("Theta 6 error is: %04.8f" % t_6_e)
    print ("\n**These theta errors may not be a correct representation of your code, due to the fact \
           \nthat the arm can have muliple positions. It is best to add your forward kinmeatics to \
           \nconfirm whether your code is working or not**")
    print (" ")

    # Find FK EE error
    if not(sum(your_ee)==3):
        ee_x_e = abs(your_ee[0]-test_case[0][0][0])
        ee_y_e = abs(your_ee[1]-test_case[0][0][1])
        ee_z_e = abs(your_ee[2]-test_case[0][0][2])
        ee_offset = sqrt(ee_x_e**2 + ee_y_e**2 + ee_z_e**2)
        print ("\nEnd effector error for x position is: %04.8f" % ee_x_e)
        print ("End effector error for y position is: %04.8f" % ee_y_e)
        print ("End effector error for z position is: %04.8f" % ee_z_e)
        print ("Overall end effector offset is: %04.8f units \n" % ee_offset)

if __name__ == "__main__":
    # Change test case number for different scenarios
    test_case_number = 3

    test_code(test_cases[test_case_number])
