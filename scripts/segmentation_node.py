#!/usr/bin/env python3

from helper import *
import numpy as np
import rospy
import cv_bridge
from sensor_msgs.msg import Image
from std_msgs.msg import Duration
from keras_segmentation.models.unet import resnet50_unet

def on_image(msg):
    on_image.last_image = msg

on_image.last_image = None

if __name__ == "__main__":
    rospy.init_node('segmentation_node')

    TOPIC_IMAGE = rospy.get_param('/topic_image', 'image_raw')
    TOPIC_SEMANTIC = rospy.get_param('/topic_semantic', 'semantic')
    TOPIC_SEMANTIC_COLOR = rospy.get_param('/topic_semantic_color', 'semantic_color')
    RATE = rospy.get_param('~rate', 30.0)

    sub_image = rospy.Subscriber(TOPIC_IMAGE, Image, on_image)
    pub_semantic = rospy.Publisher(TOPIC_SEMANTIC, Image, queue_size = 1)
    pub_semantic_color = rospy.Publisher(TOPIC_SEMANTIC_COLOR, Image, queue_size = 1)
    #pub_dur = rospy.Publisher("/duration", Duration, queue_size=10)

    rate = rospy.Rate(RATE)
    model = resnet50_unet(n_classes=23 ,  input_height=416, input_width=608)
    model.load_weights('checkpoint.h5')


    while not rospy.is_shutdown():
        #rate.sleep()

        if on_image.last_image is None:
            continue

        header = on_image.last_image.header
        t_begin = rospy.Time.now()
        semantic = seg_predict(model,cv_bridge.imgmsg_to_cv2(on_image.last_image,flip_channels=True))
        t_end = rospy.Time.now()
        rospy.loginfo(t_end - t_begin)
        #pub_dur/publish(t_end - t_begin) 

        if pub_semantic.get_num_connections() > 0:
            m = cv_bridge.cv2_to_imgmsg(semantic.astype(np.uint8), encoding = 'mono8')
            m.header.stamp.secs = header.stamp.secs
            m.header.stamp.nsecs = header.stamp.nsecs
            pub_semantic.publish(m)

        if pub_semantic_color.get_num_connections() > 0:
            m = cv_bridge.cv2_to_imgmsg(mask_to_color(semantic,color_map), encoding = 'rgb8')
            m.header.stamp.secs = header.stamp.secs
            m.header.stamp.nsecs = header.stamp.nsecs
            pub_semantic_color.publish(m)


