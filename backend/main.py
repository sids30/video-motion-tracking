from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import tempfile
import json
import base64
import io
import os
import traceback
import uuid
from PIL import Image
from collections import OrderedDict

app = FastAPI()

# Add CORS middleware to allow requests from Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# A simple cache with a maximum size for storing tracking sessions
class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

# Initialize the cache for tracking sessions
TRACKING_CACHE = LRUCache(10)  

@app.get("/")
def read_root():
    return {"message": "Video Tracking API is running"}

@app.post("/track_point")
async def track_point(video: UploadFile = File(...), data: str = Form(...)):
    """
    Track a user-selected point in a video using the Lucas-Kanade method.
    """
    try:
        # Parse the request data
        request_data = json.loads(data)
        start_time = request_data["start_time"]
        end_time = request_data["end_time"]
        sample_rate = request_data["sample_rate"]
        
        # Fix for the selected point format issue
        try:
            # Check the format of selected_point
            selected_point_data = request_data["selected_point"]
            
            if isinstance(selected_point_data, list) and len(selected_point_data) >= 2:
                selected_point = (int(selected_point_data[0]), int(selected_point_data[1]))

            elif isinstance(selected_point_data, dict) or (isinstance(selected_point_data, list) and hasattr(selected_point_data, '__getitem__') and 0 in selected_point_data and 1 in selected_point_data):
                selected_point = (int(selected_point_data[0]), int(selected_point_data[1]))

            elif isinstance(selected_point_data, tuple) and len(selected_point_data) >= 2:
                selected_point = selected_point_data

            else:
                # Try to extract values from what we received
                x = y = 0
                for k, v in selected_point_data.items() if hasattr(selected_point_data, 'items') else enumerate(selected_point_data):
                    if k in [0, '0', 'x', 'X']:
                        x = int(v)
                    elif k in [1, '1', 'y', 'Y']:
                        y = int(v)
                selected_point = (x, y)
                
            print(f"Parsed selected_point as: {selected_point}")
        except Exception as e:
            print(f"Error parsing selected_point: {e}")
            print(f"Original data: {selected_point_data}")
            # Default to center of frame if we can't parse the point
            selected_point = (frame_width // 2, frame_height // 2)
        
        window_size = request_data["window_size"]
        preprocessing = request_data["preprocessing"]
        
        # Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(await video.read())
            video_path = tmp_file.name
        
        # Check if the video file exists and is valid
        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            raise HTTPException(status_code=400, detail="Invalid video file")
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Failed to open video file")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate frame indices based on start and end times
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        # Set the video position to the start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Initialize variables for tracking
        prev_gray = None
        prev_points = np.array([selected_point], dtype=np.float32)
        initial_point = selected_point
        
        # Initialize results
        tracked_frames = []
        point_positions = []
        speeds = []
        
        # Initialize params for lucas kanade optical flow
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Initialize variables for stabilization if enabled
        if preprocessing["stabilization"]:
            # Initialize ORB detector for stabilization
            orb = cv2.ORB_create(500)
            prev_kp = None
            prev_des = None
        
        # Process frames
        frame_idx = start_frame
        processed_frame_count = 0
        
        # Debug information
        debug_info = {
            "video_path": video_path,
            "fps": fps,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "selected_point": selected_point,
            "sample_rate": sample_rate
        }
        
        # For frame processing tracking
        frames_read = 0
        frames_processed = 0
        
        while frame_idx < end_frame:
            # Read the next frame
            ret, frame = cap.read()
            if not ret:
                break
                
            frames_read += 1
            
            # Skip frames based on sample rate
            if (frame_idx - start_frame) % sample_rate != 0:
                frame_idx += 1
                continue
                
            frames_processed += 1
            
            # Apply preprocessing if enabled
            processed_frame = frame.copy()
            
            # Image stabilization
            if preprocessing["stabilization"] and processed_frame_count > 0:
                # Convert to grayscale for feature detection
                current_gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
                
                # Detect keypoints and compute descriptors
                kp, des = orb.detectAndCompute(current_gray, None)
                
                if prev_kp is not None and prev_des is not None and len(kp) > 0 and len(prev_kp) > 0:
                    # Match features
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = bf.match(prev_des, des)
                    
                    # Sort matches by distance
                    matches = sorted(matches, key=lambda x: x.distance)
                    
                    # Keep only good matches
                    good_matches = matches[:min(50, len(matches))]
                    
                    if len(good_matches) >= 10:  # Need enough matches for transformation
                        # Extract location of matched keypoints
                        src_pts = np.float32([prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        
                        # Find homography
                        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        
                        if H is not None:
                            # Warp the current frame to align with the previous frame
                            h, w = processed_frame.shape[:2]
                            processed_frame = cv2.warpPerspective(processed_frame, H, (w, h))
                
                # Update previous keypoints and descriptors
                prev_kp = kp
                prev_des = des
            
            # Histogram equalization
            if preprocessing["histogram_eq"]:
                # Convert to YUV color space
                yuv = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2YUV)
                # Apply histogram equalization to the Y channel
                yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                # Convert back to BGR
                processed_frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            
            # Gaussian blur
            if preprocessing["blur"]:
                kernel_size = preprocessing["blur_strength"]
                if kernel_size % 2 == 0:  # Ensure kernel size is odd
                    kernel_size += 1
                processed_frame = cv2.GaussianBlur(processed_frame, (kernel_size, kernel_size), 0)
            
            # Edge detection
            if preprocessing["edge_detection"]:
                # Convert to grayscale
                gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
                # Apply Canny edge detection
                edges = cv2.Canny(gray, 50, 150)
                # Convert back to BGR for display
                processed_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            # Convert frame to grayscale for optical flow
            gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
            
            if prev_gray is None:
                # First frame
                prev_gray = gray
                # Add the initial position
                point_positions.append(selected_point)
            else:
                # Calculate optical flow
                next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, prev_points, None, **lk_params
                )
                
                # Check if the point was successfully tracked
                if status[0][0] == 1:
                    # Update the tracking point - Fixed handling of next_points
                    try:
                        # This is where the error occurred before - properly handle numpy arrays
                        next_x = int(next_points[0][0][0]) if isinstance(next_points[0][0], np.ndarray) else int(next_points[0][0])
                        next_y = int(next_points[0][0][1]) if isinstance(next_points[0][0], np.ndarray) else int(next_points[0][1])
                        current_point = (next_x, next_y)
                        prev_points = next_points
                    except Exception as e:
                        print(f"Error processing next_points: {e}")
                        print(f"next_points shape: {next_points.shape}, type: {type(next_points)}")
                        print(f"next_points contents: {next_points}")
                        # If we can't process the next point, use the previous one
                        if point_positions:
                            current_point = point_positions[-1]
                        else:
                            current_point = selected_point
                    
                    # Add the new position
                    point_positions.append(current_point)
                    
                    # Calculate speed in pixels per second
                    if len(point_positions) > 1:
                        prev_point = point_positions[-2]
                        current_point = point_positions[-1]
                        
                        # Distance between points
                        distance = np.sqrt(
                            (current_point[0] - prev_point[0])**2 + 
                            (current_point[1] - prev_point[1])**2
                        )
                        
                        # Convert to pixels per second
                        time_between_frames = sample_rate / fps
                        speed = distance / time_between_frames
                        speeds.append(speed)
                    else:
                        speeds.append(0)
                    
                    # Draw rectangle around the tracked point
                    half_window = window_size // 2
                    top_left = (current_point[0] - half_window, current_point[1] - half_window)
                    bottom_right = (current_point[0] + half_window, current_point[1] + half_window)
                    cv2.rectangle(processed_frame, top_left, bottom_right, (0, 255, 0), 2)
                    
                    # Draw a red dot at the center of the tracked point
                    cv2.circle(processed_frame, current_point, 5, (0, 0, 255), -1)
                    
                    # Add frame number and speed
                    frame_text = f"Frame: {processed_frame_count}"
                    speed_text = f"Speed: {speeds[-1]:.2f} px/sec"
                    cv2.putText(processed_frame, frame_text, (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(processed_frame, speed_text, (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                else:
                    # If tracking fails, display a message
                    cv2.putText(processed_frame, "Tracking failed", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # Append the last known position and a speed of 0
                    if point_positions:
                        point_positions.append(point_positions[-1])
                    else:
                        point_positions.append(selected_point)
                    speeds.append(0)
            
            prev_gray = gray
            
            # Convert the processed frame to base64 for sending to frontend
            # Use lower JPEG quality for smaller file size
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70] 
            success, buffer = cv2.imencode('.jpg', processed_frame, encode_params)
            if not success:
                raise Exception("Failed to encode frame as JPEG")
                
            img_str = base64.b64encode(buffer).decode('utf-8')
            tracked_frames.append(img_str)
            
            # Increment counters
            frame_idx += 1
            processed_frame_count += 1
            
        # Additional debug information about frame processing
        debug_info["frames_read"] = frames_read
        debug_info["frames_processed"] = frames_processed
        
        # Close the video file
        cap.release()
        
        # Make sure we have at least one frame
        if not tracked_frames:
            raise HTTPException(
                status_code=400, 
                detail="No frames were processed. Check your video file and time range."
            )
        
        # Remove the temporary file
        if os.path.exists(video_path):
            os.unlink(video_path)
        
        # Generate a unique ID for this tracking session
        tracking_id = str(uuid.uuid4())
        
        # Store frames in the cache for chunked access
        TRACKING_CACHE.put(tracking_id, tracked_frames)
        
        # Return the results with frames for backward compatibility
        return {
            "tracking_id": tracking_id,
            "frames": tracked_frames, 
            "positions": point_positions,
            "speeds": speeds,
            "total_frames": processed_frame_count
        }
    
    except Exception as e:
        # Print the full traceback for debugging
        traceback_str = traceback.format_exc()
        print(f"Error in track_point: {traceback_str}")
        
        # Clean up in case of error
        if 'video_path' in locals() and os.path.exists(video_path):
            os.unlink(video_path)
            
        # Return detailed error information
        return {
            "error": str(e),
            "traceback": traceback_str,
            "request_data": request_data if 'request_data' in locals() else "Not parsed yet",
            "debug_info": debug_info if 'debug_info' in locals() else {}
        }

@app.post("/track_point_chunked")
async def track_point_chunked(video: UploadFile = File(...), data: str = Form(...)):
    """
    Track a user-selected point in a video but return results without frames
    for better handling of large videos.
    """
    try:
        response = await track_point(video, data)
        
        # If there was an error, return it
        if "error" in response:
            return response

        return {
            "tracking_id": response["tracking_id"],
            "positions": response["positions"],
            "speeds": response["speeds"],
            "total_frames": response["total_frames"]
        }
    
    except Exception as e:
        return {"error": str(e)}

@app.get("/get_frame/{tracking_id}/{frame_index}")
async def get_frame(tracking_id: str, frame_index: int):
    """
    Get a specific frame from a tracking session.
    """
    frames = TRACKING_CACHE.get(tracking_id)
    if not frames:
        raise HTTPException(status_code=404, detail="Tracking session not found")
    
    try:
        frame_index = int(frame_index)
        if frame_index < 0 or frame_index >= len(frames):
            raise HTTPException(status_code=404, detail=f"Frame index out of range (0-{len(frames)-1})")
        
        return {"frame": frames[frame_index]}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid frame index")

@app.get("/get_frames_batch/{tracking_id}/{start_index}/{end_index}")
async def get_frames_batch(tracking_id: str, start_index: int, end_index: int):
    """
    Get a batch of frames from a tracking session.
    """
    frames = TRACKING_CACHE.get(tracking_id)
    if not frames:
        raise HTTPException(status_code=404, detail="Tracking session not found")
    
    try:
        start_index = int(start_index)
        end_index = int(end_index)
        
        if start_index < 0 or end_index >= len(frames) or start_index > end_index:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid range. Valid range is 0-{len(frames)-1}"
            )
        
        # Return a batch of frames
        return {"frames": frames[start_index:end_index+1]}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid index values")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)