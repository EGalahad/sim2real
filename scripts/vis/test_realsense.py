import pyrealsense2 as rs
import numpy as np
import cv2
import time
from datetime import datetime

# Initialize RealSense pipeline
pipe = rs.pipeline()
cfg = rs.config()

cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

pipe.start(cfg)

# Recording parameters
min_depth, max_depth = 0.1, 4.0
recording_interval = 0.02  # 50 FPS (every 0.02 seconds)
recorded_frames = []  # Store all frames in memory
start_time = time.time()
last_record_time = 0

# Generate output filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"realsense_recording_{timestamp}.mp4"

print(f"Recording started. Press 'q' to quit and save as {output_filename}")
print("Recording every 0.02s (50 FPS)")

try:
    while True:
        current_time = time.time()
        
        # Use try_wait_for_frames for non-blocking frame retrieval
        success, frames = pipe.try_wait_for_frames(timeout_ms=100)
        
        if not success:
            continue
            
        # Get depth and color frames
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue
            
        # Record frame every 0.02s
        if current_time - last_record_time >= recording_interval:
            # Process depth image for display
            depth_image = np.asanyarray(depth_frame.get_data()) / 1000.0  # Convert to meters
            depth_image = (depth_image - min_depth) / (max_depth - min_depth)
            depth_image = (np.clip(depth_image, 0, 1) * 255).astype(np.uint8)
            depth_cm = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
            
            # Process color image
            color_image = np.asanyarray(color_frame.get_data())
            # Convert RGB to BGR for OpenCV
            color_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            
            # Display frames
            cv2.imshow('depth', depth_cm)
            cv2.imshow('color', color_bgr)
            
            # Create side-by-side frame for recording
            # Resize depth colormap to match color frame if needed
            if depth_cm.shape[:2] != color_bgr.shape[:2]:
                depth_cm_resized = cv2.resize(depth_cm, (color_bgr.shape[1], color_bgr.shape[0]))
            else:
                depth_cm_resized = depth_cm
            
            # Combine depth and color side by side
            combined_frame = np.hstack([color_bgr, depth_cm_resized])
            
            # Store frame in memory for later writing
            recorded_frames.append(combined_frame.copy())
            last_record_time = current_time
            
            # Print progress every 100 frames
            if len(recorded_frames) % 100 == 0:
                elapsed = current_time - start_time
                print(f"Recorded {len(recorded_frames)} frames in {elapsed:.1f}s")
        
        # Check for quit key
        if cv2.waitKey(1) == ord('q'):
            break
            
except KeyboardInterrupt:
    print("\nRecording interrupted by user")
    
finally:
    # Clean up
    pipe.stop()
    cv2.destroyAllWindows()
    
    # Write all recorded frames to MP4 file
    if len(recorded_frames) > 0:
        print(f"\nWriting {len(recorded_frames)} frames to {output_filename}...")
        
        # Get frame dimensions from first frame
        height, width = recorded_frames[0].shape[:2]

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_filename, fourcc, 50.0, (width, height))
        for i, frame in enumerate(recorded_frames):
            video_writer.write(frame)
        video_writer.release()
        
        total_time = time.time() - start_time
        print(f"\nRecording saved: {output_filename}")
        print(f"Total frames recorded: {len(recorded_frames)}")
        print(f"Recording duration: {total_time:.2f}s")
        print(f"Average FPS: {len(recorded_frames)/total_time:.1f}")
        print(f"Video dimensions: {width}x{height}")
    else:
        print("No frames were recorded")