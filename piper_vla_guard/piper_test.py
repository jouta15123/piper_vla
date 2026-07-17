#!/usr/bin/env python3
# -*-coding:utf8-*-
import time
import math
from piper_sdk import Piper

if __name__ == "__main__":
    def get_pos():
        '''Get the current joint radians of the robotic arm and the gripper opening distance'''
        joint_state = piper.get_joint_states()[0]
        if have_gripper:
            return joint_state + (piper.get_gripper_states()[0][0], )
        return joint_state 
    
    def stop():
        '''Stop the robotic arm; this function must be called first when exiting the teaching mode for the first time to control the robotic arm in CAN mode'''
        interface.EmergencyStop(0x01)
        time.sleep(1.0)
        limit_angle = [0.1745, 0.7854, 0.2094]  # The robotic arm can be restored only when the radians of joints 2, 3, and 5 are within the limit range to prevent damage caused by falling from a large radian
        pos = get_pos()
        while not (abs(pos[1]) < limit_angle[0] and abs(pos[2]) < limit_angle[0] and pos[4] < limit_angle[1] and pos[4] > limit_angle[2]):
            time.sleep(0.01)
            pos = get_pos()
        # Restore the robotic arm
        piper.disable_arm()
        time.sleep(1.0)
    
    def enable():
        '''Enable the robotic arm and gripper'''
        while not piper.enable_arm():
            time.sleep(0.01)
        if have_gripper:
            time.sleep(0.01)
            piper.enable_gripper()
        interface.ModeCtrl(0x01, 0x01, move_spd_rate_ctrl, 0x00)
        print("INFO: Enable successful")


    move_spd_rate_ctrl = 40

    # ===== 円運動設定 =====
    # radius = 0.12
    radius = 0.15
    period = 2

    # ===== グリッパ設定 =====
    grip_min = 0.01      # 閉
    grip_max = 0.1   # 開（機種に合わせて調整）
    grip_period = 2.0 # 開閉周期

    dt = 0.01
    run_time = 10
    have_gripper = True
    timeout = 10.0

    piper = Piper("can0")
    interface = piper.init()
    piper.connect()
    time.sleep(0.2)

    while not piper.enable_arm():
        time.sleep(0.01)

    piper.enable_gripper()

    if interface.GetArmStatus().arm_status.ctrl_mode != 1:
        stop()  # This function must be called first when exiting the teaching mode for the first time to switch to CAN mode
    over_time = time.time() + timeout
    while interface.GetArmStatus().arm_status.ctrl_mode != 1:
        if over_time < time.time():
            print("ERROR: Failed to switch to CAN mode, please check if the teaching mode is exited")
            exit()
        interface.ModeCtrl(0x01, 0x01, move_spd_rate_ctrl, 0x00)
        time.sleep(0.01)
    enable()

    while not piper.enable_arm():
        time.sleep(0.01)

    piper.enable_gripper()

    print("Start circular motion with gripper...")

    base = list(piper.get_joint_states()[0])

    omega = 2 * math.pi / period
    grip_omega = 2 * math.pi / grip_period

    start = time.time()

    while time.time() - start < run_time:

        t = time.time() - start

        joints = base.copy()

        # ===== 円運動 =====
        joints[0] = base[0] + radius * math.cos(omega * t)
        joints[1] = base[1] + radius * math.sin(omega * t)
        # joints[2] = base[2] + radius * math.cos(omega * t)
        # joints[3] = base[3] + radius * math.sin(omega * t)
        # joints[4] = base[4] + radius * math.cos(omega * t)
        # joints[5] = base[5] + radius * math.sin(omega * t)


        piper.move_j(joints, move_spd_rate_ctrl)

        # ===== グリッパー開閉（sin波）=====
        g = (math.sin(grip_omega * t) + 1) / 2
        grip_pos = grip_min + g * (grip_max - grip_min)

        piper.move_gripper(grip_pos, 1)

        print("===================")
        # print("Joint: ", joints)
        # print("Gripper: ", grip_pos)
        print("Joint_Pos;",piper.get_joint_states()[0])###Jointの現在状態
        print("gripper:", piper.get_gripper_states()[0][0])
        print("End_Pos_Eular;",piper.get_end_pose_euler()[0])
        # print("x;",piper.get_end_pose_euler()[0][0])
        # print("y;",piper.get_end_pose_euler()[0][1])
        # print("z;",piper.get_end_pose_euler()[0][2])
        # print("rx;",piper.get_end_pose_euler()[0][3])
        # print("ry;",piper.get_end_pose_euler()[0][4])
        # print("rz;",piper.get_end_pose_euler()[0][5])
        print("===================\n\n")
        
        time.sleep(dt)

    print("Finished")
