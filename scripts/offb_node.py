#! /usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped , TwistStamped
from mavros_msgs.msg import State 
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
from ros2yolo import orientation
from offboard_py.msg import Orientation 
import numpy as np
import time

current_state = State()
foward_state=0
moving_state =0
last_x=0
last_y=0
last_z=2
global key
key=0
def state_cb(msg):
    global current_state
    current_state = msg

def position_cb(msg):
    global c_x,c_y,c_z
    current_position = msg
    c_x = current_position.pose.position.x
    c_y = current_position.pose.position.y
    c_z = current_position.pose.position.z


def moving_cb(msg):
    global norm_vector,foward_state,moving_state
    foward_state=msg.foward_state
    norm_vector = np.array(msg.norm_vector)
    moving_state=(foward_state!=0 or (not np.array_equal(norm_vector, [0,0])))
    # print("foward_state",foward_state,"norm_vector:",norm_vector,"moving_state",moving_state)


def pose_keep(x,y,z):

    # last_c_y=0
    # last_c_z=2
  
    
    pose.pose.position.x = last_x
    pose.pose.position.y = last_y
    pose.pose.position.z = last_z
    local_pos_pub.publish(pose)

    

if __name__ == "__main__":

    rospy.init_node("ofrfb_node_py")

    state_sub = rospy.Subscriber("mavros/state", State, callback = state_cb)
    c_position_sub=rospy.Subscriber("/mavros/local_position/pose", PoseStamped, position_cb)

    moving_sub=rospy.Subscriber('orientation_topic', Orientation, moving_cb)



    local_pos_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)
    local_velocity_pub = rospy.Publisher("mavros/setpoint_velocity/cmd_vel",  TwistStamped, queue_size=10)
    
    # local_velocity_pub = rospy.Publisher("mavros/setpoint_velocity/cmd_vel_unstamped",  Twist, queue_size=1)
    rospy.wait_for_service("/mavros/cmd/arming")
    arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)

    rospy.wait_for_service("/mavros/set_mode")
    set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)


    # Setpoint publishing MUST be faster than 2Hz
    rate = rospy.Rate(10)

    # Wait for Flight Controller connection
    while(not rospy.is_shutdown() and not current_state.connected):
        rate.sleep()


    pose = PoseStamped()
    
    pose.pose.position.x = 0
    pose.pose.position.y = 0
    pose.pose.position.z = 2

    vel= TwistStamped()
    Max_vel=1
    # vel.linear.z=0
    # vel.linear.y=0.1


    for i in range(50):
        if(rospy.is_shutdown()):
            break
        # local_velocity_pub.publish(vel)
        local_pos_pub.publish(pose)
        rate.sleep()

    offb_set_mode = SetModeRequest()
    offb_set_mode.custom_mode = 'OFFBOARD'

    arm_cmd = CommandBoolRequest()
    arm_cmd.value = True

    last_req = rospy.Time.now()

    # Send a few setpoints before starting

    # while not rospy.is_shutdown():
    #     local_pos_pub.publish(pose)
    #     if current_state.mode == "OFFBOARD" and current_state.armed:
    #         rospy.loginfo("OFFBOARD enabled,Vehicle armed")
    #         break  # 成功切换到OFFBOARD模式并且无人机已解锁
    #     rate.sleep()

    # while not rospy.is_shutdown():
    #     local_velocity_pub.publish(vel)
    #     rate.sleep()

    while(not rospy.is_shutdown()):
        if(current_state.mode != "OFFBOARD" and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
            if(set_mode_client.call(offb_set_mode).mode_sent == True):
                rospy.loginfo("OFFBOARD enabled")

            last_req = rospy.Time.now()
        else:
            if(not current_state.armed and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
                if(arming_client.call(arm_cmd).success == True):
                    rospy.loginfo("Vehicle armed")

                last_req = rospy.Time.now()
        # local_velocity_pub.publish(vel)
        local_pos_pub.publish(pose)
        # print (c_x,c_y,c_z)
        

        if(abs(pose.pose.position.x - c_x) < 0.1 and 
           abs(pose.pose.position.y - c_y) < 0.1 and 
           abs(pose.pose.position.z - c_z )< 0.1):
            # print("off boarded")
            break
        # rate.sleep()
             
        
        
    while(not rospy.is_shutdown()):
        
        

        if foward_state==4:    #avoidence 
            vel.twist.linear.x=-4*Max_vel
            local_velocity_pub.publish(vel)
            time.sleep(0.4)
            last_x=c_x
            last_y=c_y
            last_z=c_z
            pose_keep(last_x,last_y,last_z)
            print("avoided")
        if moving_state:
            vel.twist.angular.z = 0
            [vector_y,vector_z]=norm_vector
            vector_y=vector_y*(-1.0)
            vector_X=(2.0-abs(vector_y)-abs(vector_z))/2.0

            if foward_state==0:  #moving zy
                print(0)
                vel.twist.linear.x=0
                vel.twist.linear.y=vector_y*Max_vel
                vel.twist.linear.z=vector_z*Max_vel
                local_velocity_pub.publish(vel)
                last_x=c_x
                last_y=c_y
                last_z=c_z

                
            if foward_state==1 :#moving x
                print(1)
                vel.twist.linear.x=vector_X*Max_vel
                vel.twist.linear.y=vector_y*Max_vel
                vel.twist.linear.z=vector_z*Max_vel
                local_velocity_pub.publish(vel)
                last_x=c_x
                last_y=c_y
                last_z=c_z
            
            if foward_state==2:
                print(2)
                vel.twist.linear.y=vector_y*Max_vel
                vel.twist.linear.z=vector_z*Max_vel
                local_velocity_pub.publish(vel)      
                last_x=c_x
                last_y=c_y
                last_z=c_z
                key=1

            if foward_state ==3 and key==1:
                print(3)
                key=0
                foward_state=0
                vel.twist.linear.x=0.5
                local_velocity_pub.publish(vel)
                time.sleep(1)
                last_x=c_x
                last_y=c_y
                last_z=c_z
                
                pose_keep(last_x,last_y,last_z)
            # if foward_state==3 and key>10:#moving x
            #     foward_state=0
            #     print(333333333)
            #     key=0   
            #     vel.twist.linear.x=vector_X*Max_vel*2
            #     vel.twist.linear.y=vector_y*Max_vel
            #     vel.twist.linear.z=vector_z*Max_vel
                
            #     local_velocity_pub.publish(vel)
            #     time.sleep(0.2)
            #     last_x=c_x
            #     last_y=c_y
            #     last_z=c_z
            #     vel.twist.linear.x=0
            #     vel.twist.linear.y=0
            #     vel.twist.linear.z=0
            #     local_velocity_pub.publish(vel)
            #     last_x=c_x
            #     last_y=c_y
            #     last_z=c_z
            #     pose_keep(last_x,last_y,last_z)
                

            # if foward_state==2 and key==1 :
            #     vel.twist.linear.x=2*Max_vel
            #     vel.twist.linear.y=0
            #     vel.twist.linear.z=0
            #     local_velocity_pub.publish(vel)
            #     time.sleep(0.3)
            #     vel.twist.linear.x=0
            #     vel.twist.linear.y=0
            #     vel.twist.linear.z=0
            #     local_velocity_pub.publish(vel)
            #     last_x=c_x
            #     last_y=c_y
            #     last_z=c_z
            #     pose_keep(last_x,last_y,last_z)
            #     key=0
            #     print("passing")
        else:
            vel.twist.linear.x=0
            vel.twist.linear.y=0
            vel.twist.linear.z=0
            local_velocity_pub.publish(vel) 
         
            pose_keep(last_x,last_y,last_z)
      
            
