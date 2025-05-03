import streamlit as st
import requests
import base64
import cv2
import numpy as np
import io
import json
import tempfile
import asyncio
import time
from PIL import Image
import matplotlib.pyplot as plt

# Define the backend API URL
BACKEND_URL = "https://backend-z5ayxyaklq-uc.a.run.app"

# Initialize session state to store results
if 'tracking_results' not in st.session_state:
    st.session_state.tracking_results = None
if 'tracking_frames' not in st.session_state:
    st.session_state.tracking_frames = None
if 'tracking_id' not in st.session_state:
    st.session_state.tracking_id = None

def main():
    st.title("Video Feature Tracking App")
    st.sidebar.header("Controls")

    # File uploader for MP4 video
    uploaded_file = st.sidebar.file_uploader("Upload a video", type=["mp4"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            video_path = tmp_file.name

        # Display the uploaded video
        st.video(uploaded_file)

        # Get video info (frame count, FPS, duration)
        video_info = get_video_info(video_path)
        if not video_info:
            st.error("Failed to extract video information. Please try another video.")
            return

        # Display video information
        st.sidebar.subheader("Video Information")
        st.sidebar.text(f"Duration: {video_info['duration']:.2f} seconds")
        st.sidebar.text(f"Frame count: {video_info['frame_count']}")
        st.sidebar.text(f"FPS: {video_info['fps']:.2f}")

        # Time range selection
        st.sidebar.subheader("Time Range")
        start_time = st.sidebar.slider("Start Time (seconds)", 
                                      min_value=0.0, 
                                      max_value=max(0.1, video_info['duration'] - 1.0), 
                                      value=0.0)
        
        end_time = st.sidebar.slider("End Time (seconds)", 
                                    min_value=min(start_time + 1.0, video_info['duration']), 
                                    max_value=video_info['duration'], 
                                    value=min(start_time + 5.0, video_info['duration']))
        
        # Frame sampling rate
        sample_rate = st.sidebar.slider("Sample every Nth frame", 
                                      min_value=1, 
                                      max_value=10, 
                                      value=3)

        # Preprocessing options
        st.sidebar.subheader("Preprocessing Options")
        use_stabilization = st.sidebar.checkbox("Image Stabilization", value=False)
        use_histogram_eq = st.sidebar.checkbox("Histogram Equalization", value=False)
        use_blur = st.sidebar.checkbox("Gaussian Blur", value=False)
        blur_strength = st.sidebar.slider("Blur Strength", min_value=1, max_value=15, value=5, step=2) if use_blur else 5
        use_edge_detection = st.sidebar.checkbox("Edge Detection", value=False)

        # Get a frame for point selection
        start_frame_idx = int(start_time * video_info['fps'])
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            st.error("Failed to read the first frame. Please try another video.")
            return

        # Convert to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # User selects point
        st.subheader("Select a Point to Track")
        st.text("Click on the image to select a point to track")
        
        # Use Streamlit columns to display frame and results side by side
        col1, col2 = st.columns(2)
        
        with col1:
            # Display the frame for point selection
            frame_h, frame_w = frame_rgb.shape[:2]
            selected_point = None
            
            # Display the image
            st.image(frame_rgb, caption="Reference frame")
            
            # Add clearer instructions for point selection
            st.info("⚠️ Use the X and Y coordinate inputs below to select a point to track.")
            
            # Use number inputs for coordinates
            selected_x = st.number_input("X coordinate (horizontal position)", 
                                        min_value=0, 
                                        max_value=frame_w-1, 
                                        value=frame_w//2,
                                        help="Horizontal position of the point to track (0 is left, higher values move right)")
            
            selected_y = st.number_input("Y coordinate (vertical position)", 
                                        min_value=0, 
                                        max_value=frame_h-1, 
                                        value=frame_h//2,
                                        help="Vertical position of the point to track (0 is top, higher values move down)")
            
            selected_point = (selected_x, selected_y)
            
            # Draw the selected point on the image
            display_frame = frame_rgb.copy()
     
            cv2.circle(display_frame, selected_point, 10, (255, 0, 0), -1)  
            cv2.circle(display_frame, selected_point, 12, (0, 255, 0), 2)   
            st.image(display_frame, caption="Selected point (blue dot)")
        
        with col2:
            st.subheader("Tracking Results")
            if selected_point:
                st.text("Point will be tracked when processing starts")
                
        # Window size for tracking
        window_size = st.sidebar.slider("Tracking Window Size", min_value=10, max_value=100, value=50)
        
        # Choose processing method
        use_chunked = st.sidebar.checkbox("Use Chunked Processing (recommended for large videos)", value=True)
        
        # Tracking buttons with unique keys
        start_button = st.sidebar.button("Start Tracking", key="start_tracking")
        reset_button = st.sidebar.button("Reset Tracking", key="reset_tracking") if 'tracking_results' in st.session_state and st.session_state.tracking_results is not None else False
        
        # Reset button
        if reset_button:
            # Clear previous results
            st.session_state.tracking_results = None
            st.session_state.tracking_frames = None
            st.session_state.tracking_id = None
            st.experimental_rerun()
        
        # Start button
        if start_button:
            if selected_point and not st.session_state.tracking_results:
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                tracking_video_placeholder = st.empty()
                speed_chart_placeholder = st.empty()
                
                # Prepare the request data
                request_data = {
                    "start_time": start_time,
                    "end_time": end_time,
                    "sample_rate": sample_rate,
                    "selected_point": [selected_x, selected_y],  # Send as a simple array, not a tuple
                    "window_size": window_size,
                    "preprocessing": {
                        "stabilization": use_stabilization,
                        "histogram_eq": use_histogram_eq,
                        "blur": use_blur,
                        "blur_strength": blur_strength,
                        "edge_detection": use_edge_detection
                    }
                }
                
                # Upload the video file to the backend
                files = {'video': open(video_path, 'rb')}
                
                # Send the request to the backend
                status_text.text("Sending video to backend for processing...")
                
                try:
                    if use_chunked:
                        # Use chunked processing endpoint
                        response = requests.post(
                            f"{BACKEND_URL}/track_point_chunked",
                            files=files,
                            data={"data": json.dumps(request_data)}
                        )
                    else:
                        # Use standard processing endpoint
                        response = requests.post(
                            f"{BACKEND_URL}/track_point",
                            files=files,
                            data={"data": json.dumps(request_data)}
                        )
                    
                    if response.status_code == 200:
                        results = response.json()
                        st.session_state.sample_rate = sample_rate  # Store sample rate for timestamp calculation
                        
                        # Display processing complete message
                        status_text.text("Processing complete!")
                        
                        # Debug: Display the received response structure
                        st.write("Response structure:", results.keys())
                        
                        # Check for errors
                        if "error" in results:
                            st.error(f"Error from backend: {results['error']}")
                            if 'traceback' in results:
                                with st.expander("Error details"):
                                    st.text(results['traceback'])
                            return
                        
                        # Store the tracking ID if using chunked processing
                        if use_chunked and "tracking_id" in results:
                            st.session_state.tracking_id = results["tracking_id"]
                            
                            # Fetch frames in batches
                            status_text.text("Downloading frames...")
                            tracking_frames = []
                            
                            total_frames = results["total_frames"]
                            batch_size = 5  # Number of frames to fetch at once
                            
                            for start_idx in range(0, total_frames, batch_size):
                                progress_bar.progress(start_idx / total_frames)
                                
                                end_idx = min(start_idx + batch_size - 1, total_frames - 1)
                                try:
                                    batch_response = requests.get(
                                        f"{BACKEND_URL}/get_frames_batch/{results['tracking_id']}/{start_idx}/{end_idx}"
                                    )
                                    
                                    if batch_response.status_code == 200:
                                        batch_data = batch_response.json()
                                        
                                        for frame_data in batch_data["frames"]:
                                            try:
                                                img_bytes = base64.b64decode(frame_data)
                                                img = Image.open(io.BytesIO(img_bytes))
                                                tracking_frames.append(np.array(img))
                                            except Exception as e:
                                                st.error(f"Error decoding frame: {str(e)}")
                                    else:
                                        st.warning(f"Failed to fetch frames batch {start_idx}-{end_idx}")
                                except Exception as e:
                                    st.error(f"Error fetching frames batch: {str(e)}")
                            
                            progress_bar.progress(1.0)
                            
                            # Store frames and results in session state
                            st.session_state.tracking_frames = tracking_frames
                            st.session_state.tracking_results = results
                            
                        elif "frames" in results:
                            # Standard processing - decode frames directly
                            tracking_frames = []
                            for frame_data in results["frames"]:
                                # Decode base64 image
                                try:
                                    img_bytes = base64.b64decode(frame_data)
                                    img = Image.open(io.BytesIO(img_bytes))
                                    tracking_frames.append(np.array(img))
                                except Exception as e:
                                    st.error(f"Error decoding frame: {str(e)}")
                                    continue
                            
                            if not tracking_frames:
                                st.error("Failed to decode any tracking frames.")
                                return
                                
                            # Store frames and results in session state
                            st.session_state.tracking_frames = tracking_frames
                            st.session_state.tracking_results = results
                        else:
                            st.error("No frame data received from backend")
                            return
                        
                        # Trigger a rerun to show the frames with the slider
                        st.experimental_rerun()
                    else:
                        st.error(f"Backend processing failed with status code {response.status_code}: {response.text}")
                        # Try to parse response if it's JSON
                        try:
                            st.json(response.json())
                        except:
                            st.text(response.text)
                
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")
                    
            else:
                if not selected_point:
                    st.error("Please select a point to track first")
        
        # If we have tracking results, display them
        if st.session_state.tracking_results is not None and st.session_state.tracking_frames is not None:
            results = st.session_state.tracking_results
            tracking_frames = st.session_state.tracking_frames
            
            st.subheader("Tracking Results")
            st.write(f"Displaying {len(tracking_frames)} tracked frames:")
            
            # Create a slider for navigating frames
            frame_index = st.slider("Frame", 0, len(tracking_frames) - 1, 0)
            
            # Calculate the actual video timestamp for the current frame
            if "sample_rate" in st.session_state:
                sample_rate = st.session_state.sample_rate
                fps = video_info["fps"]
                actual_frame_number = frame_index * sample_rate
                timestamp_seconds = actual_frame_number / fps
                minutes = int(timestamp_seconds // 60)
                seconds = timestamp_seconds % 60
                timestamp_str = f"{minutes:02d}:{seconds:05.2f}"
                
                # Create a visual timeline to show position in the video
                total_duration = video_info["duration"]
                progress_percent = min(100, (timestamp_seconds / total_duration) * 100)
                
                st.write(f"Timeline position: {progress_percent:.1f}% of video duration")
                st.progress(progress_percent / 100)
            else:
                timestamp_str = "Unknown"
            
            # Display the selected frame with enhanced caption
            st.image(
                tracking_frames[frame_index], 
                caption=f"Frame {frame_index + 1}/{len(tracking_frames)} | Video time: {timestamp_str} | Original frame: {frame_index * sample_rate}",
                use_column_width=True
            )
            
            # Display speed data as a chart
            if "speeds" in results and results["speeds"]:
                fig, ax = plt.subplots(figsize=(10, 4))
                frame_numbers = list(range(len(results["speeds"])))
                ax.plot(frame_numbers, results["speeds"])
                ax.set_xlabel("Frame Number")
                ax.set_ylabel("Speed (pixels/second)")
                ax.set_title("Object Speed Over Time")
                ax.grid(True)
                st.pyplot(fig)
                
                # Display speed statistics
                avg_speed = sum(results["speeds"]) / len(results["speeds"]) if results["speeds"] else 0
                max_speed = max(results["speeds"]) if results["speeds"] else 0
                st.text(f"Average Speed: {avg_speed:.2f} pixels/second")
                st.text(f"Maximum Speed: {max_speed:.2f} pixels/second")
            else:
                st.warning("No speed data was returned from the backend.")

def get_video_info(video_path):
    """Get basic information about the video file"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps
        
        cap.release()
        
        return {
            "frame_count": frame_count,
            "fps": fps,
            "duration": duration
        }
    except Exception as e:
        st.error(f"Error getting video info: {str(e)}")
        return None

if __name__ == "__main__":
    main()