#!/usr/bin/env python

# Copyright (C) 2017 Alex Ge, alexgecontrol@qq.com.
#
# Solution for Udacity Robotics Nanodegree: Pick and Place
#
# All Rights Reserved.

# import modules
import rospy
import tf
from kuka_arm.srv import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose
from mpmath import *
from sympy import *

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
        rospy.init_node('IK_server')

        #
        # forward kinematics:
        #
        start_time = rospy.Time.now().to_sec()

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

        # 4. end-effector:
        self.ee = T_0G[0:3,3]

        time_elapsed = rospy.Time.now().to_sec() - start_time
        self.srv = rospy.Service(
            'calculate_ik',
            CalculateIK,
            self._handle_calculate_IK
        )
        print "[IK Server]: Ready for request -- {:.3f}".format(
            time_elapsed
        )

        rospy.spin()
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

        # format:
        '''
        wc = [wc[0,0], wc[1,0], wc[2,0]]
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
        ee = [ee[0,0], ee[1,0], ee[2,0]]
        '''
        wc = [1, 1, 1]
        ee = [1, 1, 1]

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

if __name__ == "__main__":
    try:
        IKServer()
    except rospy.ROSInterruptException:
        pass
