import pyrealsense2 as rs
import numpy as np
import open3d
import cv2

def initial_configuration():
    hole_filling0 = rs.hole_filling_filter(0) #fill from left hole filling 
    align = rs.align(rs.stream.color) #aligning color to depth 
    pipeline = rs.pipeline()
    config = rs.config() 

    # Configure the pipeline to stream the depth stream and color stream 
    config.enable_stream(rs.stream.depth,rs.format.z16, 30)
    config.enable_stream(rs.stream.color,rs.format.rgb8, 30)

    # Start streaming from file
    profile = pipeline.start(config)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    # Pinhole_camera_intrinsic = open3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
    geometrie_added = False
    
    # Create a window to visualize the video
    #vis = open3d.visualization.VisualizerWithKeyCallback()
    #vis.create_window("Pointcloud")
    #pointcloud = open3d.geometry.PointCloud()

    # Getting the depth sensor's depth scale 
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    

    #convert from meters to unit of depth 
    # up_clipping_distance = up_clipping_distance_in_meters / depth_scale 
    # low_clipping_distance = lower_clipping_distance_in_meters / depth_scale

    return profile, pipeline,hole_filling0,align,intr, geometrie_added, depth_scale

def get_frame(pipeline,hole_filling0,align):       
    # Get frameset 
    frames = pipeline.wait_for_frames()
    frames = hole_filling0.process(frames).as_frameset()
    
    #Align the depth frame to color frame
    aligned_frames0 = align.process(frames)

    # Get color frame and depth frame
    color_frame = aligned_frames0.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    # Cropping the image
    # color_image = color_image[:,400:1280]
    #color_image = rescaleFrame(color_image,0.5)
    depth_frame0 = aligned_frames0.get_depth_frame()
    depth_image0 = np.asanyarray(depth_frame0.get_data())
    # Cropping the image
    # depth_image0 = depth_image0[:,400:1280]
    #depth_image0 = rescaleFrame(depth_image0,0.5)

    color_image1 = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR) #convert from RGB to BGR (color model used by cv2)
    #cv2.imshow('Color Stream', color_image1)
    
    return  depth_image0,depth_frame0,color_image1