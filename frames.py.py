import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import winsound
import time

frame_dir = "ai-cctv-main/Avenue Dataset/testing_frames"
frames = sorted(os.listdir(frame_dir))

bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=False
)

prev_gray = None
frame_count = 0

# RULES

def classify_motion(avg_motion):
    if avg_motion < 0.5:
        return "NORMAL"
    elif avg_motion < 1.5:
        return "UNUSUAL"
    else:
        return "CHAOTIC"

def classify_density(density):
    if density < 0.05:
        return "LOW"
    elif density < 0.35:
        return "MEDIUM"
    else:
        return "HIGH"

def classify_risk(motion_state, density_state):
    if density_state == "HIGH" or motion_state == "CHAOTIC":
        return "HIGH"
    if motion_state == "UNUSUAL" or density_state == "MEDIUM":
        return "MEDIUM"
    return "LOW"

def generate_alert(risk_level):
    if risk_level == "LOW":
        return "Normal crowd behavior detected."
    elif risk_level == "MEDIUM":
        return "Unusual crowd movement detected. Monitoring advised."
    else:
        return "High-risk crowd behavior detected. Immediate attention required!"
# Main Loop
previous_risk = "LOW"

for frame_name in frames:
    frame_count += 1

    frame_path = os.path.join(frame_dir, frame_name)
    frame = cv2.imread(frame_path)
    if frame is None:
        continue


    fg_mask = bg_subtractor.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    foreground_pixels = cv2.countNonZero(fg_mask)
    total_pixels = fg_mask.shape[0] * fg_mask.shape[1]
    density_ratio = foreground_pixels / total_pixels
    density_state = classify_density(density_ratio)
    
    # MOTION 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # warm-up period
    if frame_count < 30:
        prev_gray = gray
        continue

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray,
        None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    avg_motion = np.mean(magnitude)
    motion_state = classify_motion(avg_motion)
    
    risk_level_1 = min(int(avg_motion * 10), 100)  
    
    confidence = avg_motion / density_ratio
    
    # RISK
    risk_level = classify_risk(motion_state, density_state)
    alert_msg = generate_alert(risk_level)
    current_time = time.asctime()

    display_frame = frame.copy()
    
    # SOUND ALERT ON RISK CHANGE
    if risk_level != previous_risk:
        if risk_level == "MEDIUM":
            winsound.Beep(800, 300)
        elif risk_level == "HIGH":
            winsound.Beep(1000, 700)

    previous_risk = risk_level
    
    # color selection
    if risk_level == "LOW":
        color = (0, 255, 0)
    elif risk_level == "MEDIUM":
        color = (0, 255, 255)
    else:
        color = (0, 0, 255)

    # Display text on frame
    y_offset = 40
    line_height = 40
    
    cv2.putText(display_frame, f"RISK: {risk_level}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    y_offset += line_height
    
    cv2.putText(display_frame, f"Motion: {motion_state}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset += line_height
    
    cv2.putText(display_frame, f"Density: {density_state}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset += line_height
    
    cv2.putText(display_frame, alert_msg, (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    y_offset += line_height
    
    cv2.putText(display_frame, f"Time: {current_time}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    y_offset += line_height
    
    cv2.putText(display_frame, f"Confidence: {confidence:.2f}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    y_offset += line_height
    
    cv2.putText(display_frame, f"Risk Level: {risk_level_1}%", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (640, 480))

    # DENSITY
    plt.imshow(display_rgb)
    plt.axis("off")
    plt.pause(0.01)
    plt.clf()

    prev_gray = gray 

print("Processing complete!")
