#!/usr/bin/env python3
import rospy
from gazebo_msgs.msg import LinkStates
from geometry_msgs.msg import TransformStamped
import tf2_ros

STAND_LINK = "camera_stand::stand_link"

def main():
    rospy.init_node("camera_stand_tf_publisher")

    # Broadcasters
    br = tf2_ros.TransformBroadcaster()
    static_br = tf2_ros.StaticTransformBroadcaster()

    # Static TF: stand_link -> rgbd_camera_link
    # This MUST match SDF sensor pose:
    # <pose>0.25 0 0.30 0 0 0</pose>
    # Pose order: x y z roll pitch yaw
    import tf.transformations as tft
    x, y, z = 0.25, 0.0, 0.30
    roll, pitch, yaw = 0.0, 0.0, 0.0
    q = tft.quaternion_from_euler(roll, pitch, yaw)

    st = TransformStamped()
    st.header.stamp = rospy.Time.now()
    st.header.frame_id = "stand_link"
    st.child_frame_id = "rgbd_camera_link"
    st.transform.translation.x = x
    st.transform.translation.y = y
    st.transform.translation.z = z
    st.transform.rotation.x = q[0]
    st.transform.rotation.y = q[1]
    st.transform.rotation.z = q[2]
    st.transform.rotation.w = q[3]
    static_br.sendTransform(st)

    # Dynamic TF: world -> stand_link from Gazebo
    state = {"idx": None, "pose": None, "rot": None}

    def cb(msg: LinkStates):
        if state["idx"] is None:
            try:
                state["idx"] = msg.name.index(STAND_LINK)
            except ValueError:
                rospy.logwarn_throttle(5.0, f"Link '{STAND_LINK}' not found in /gazebo/link_states")
                return

        i = state["idx"]
        state["pose"] = msg.pose[i].position
        state["rot"]  = msg.pose[i].orientation

    def pub_timer(event):
        if state["pose"] is None:
            return
        p = state["pose"]
        o = state["rot"]

        t = TransformStamped()
        t.header.stamp = rospy.Time.now()  # sim time
        t.header.frame_id = "world"
        t.child_frame_id = "stand_link"
        t.transform.translation.x = p.x
        t.transform.translation.y = p.y
        t.transform.translation.z = p.z
        t.transform.rotation = o
        br.sendTransform(t)

    rospy.Subscriber("/gazebo/link_states", LinkStates, cb, queue_size=1)
    rospy.Timer(rospy.Duration(1.0/30.0), pub_timer)
    rospy.spin()

if __name__ == "__main__":
    main()