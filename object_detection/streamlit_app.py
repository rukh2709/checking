import streamlit as st
import cv2
import os
from ultralytics import YOLO
import logging
import tempfile
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_video_in_realtime(video_path, model):
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Video properties: {width}x{height} at {fps} fps, {total_frames} frames")
    
    frame_count = 0
    video_placeholder = st.empty()  # Placeholder for displaying the video

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info(f"Finished processing at frame {frame_count}/{total_frames}")
                break
            
            frame_count += 1
            if frame_count % 100 == 0:
                logger.info(f"Processing frame {frame_count}/{total_frames}")

            original_frame = frame.copy()
            
            # Dynamically read the selected classes in each frame
            selected_classes = [
                class_name for class_name, selected in checkbox_state.items() if selected
            ]
            
            # Run YOLO model on the frame
            results = model(frame)
            
            # Draw bounding boxes for detected objects
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    if class_name in selected_classes:
                        x1, y1, x2, y2 = box.xyxy[0].astype(int)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        label = f"{class_name}"
                        label_font_scale = 3
                        label_thickness = 7
                        (label_width, label_height), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, label_thickness
                        )
                        label_bg_x1, label_bg_y1 = x1, y1 - label_height - 10
                        label_bg_x2, label_bg_y2 = x1 + label_width, y1
                        cv2.rectangle(frame, (label_bg_x1, label_bg_y1), 
                                      (label_bg_x2, label_bg_y2), (0, 0, 0), -1)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                    label_font_scale, (255, 255, 255), label_thickness)
            
            combined_frame = cv2.hconcat([original_frame, frame])
            combined_frame_rgb = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(combined_frame_rgb, channels="RGB", use_column_width=True)

            time.sleep(0.1)  # Control processing speed
    except Exception as e:
        logger.error(f"Error during video processing: {str(e)}")
    finally:
        cap.release()
        logger.info("Video processing completed")

def main():
    st.title("YOLOv11 Object Detection")
    
    # Load the model
    model = YOLO("model/yolo11s.pt")
    
    # Get class names from the model
    global checkbox_state
    checkbox_state = {class_name: False for class_name in model.names.values()}
    
    # Sidebar for class selection checkboxes
    st.sidebar.header("Select Classes to Detect")
    for class_name in checkbox_state.keys():
        checkbox_state[class_name] = st.sidebar.checkbox(class_name, value=False)
    
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Save uploaded file to a temporary local file
        temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())
        
        st.video(temp_video_path)
        st.success("Video uploaded successfully. You can now process it.")
        
        if st.button("Process Video in Real-Time"):
            with st.spinner("Processing video..."):
                try:
                    process_video_in_realtime(temp_video_path, model)
                    st.success("Video processed successfully.")
                except Exception as e:
                    st.error(f"An error occurred during video processing: {str(e)}")
                    logger.exception("Error in video processing")

if __name__ == "__main__":
    main()
