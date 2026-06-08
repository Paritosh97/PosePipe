import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


app = FastAPI()

class DetectionRequest(BaseModel):
    file_path: str
    use_pose: bool = False
    use_hand: bool = False
    use_face: bool = False

@app.post("/process_video")
async def process_video(req: DetectionRequest):
    if not os.path.exists(req.file_path):
        raise HTTPException(status_code=404, detail="File not found")

    # 1. Initialize requested detectors
    detectors = {}
    if req.use_pose:
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path="/home/paritosh97/Desktop/blender/addons/PosePipe/pose_landmarker_heavy.task"),
            running_mode=mp_vision.RunningMode.VIDEO)
        detectors['pose'] = mp_vision.PoseLandmarker.create_from_options(options)

    if req.use_hand:
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path="/home/paritosh97/Desktop/blender/addons/PosePipe/hand_landmarker.task"),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2)
        detectors['hand'] = mp_vision.HandLandmarker.create_from_options(options)

    if req.use_face:
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path="/home/paritosh97/Desktop/blender/addons/PosePipe/face_landmarker.task"),
            running_mode=mp_vision.RunningMode.VIDEO,
            output_face_blendshapes=False, # Set True if you need blendshapes
            num_faces=1)
        detectors['face'] = mp_vision.FaceLandmarker.create_from_options(options)

    # 2. Process video
    cap = cv2.VideoCapture(req.file_path)
    results_sequence = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        frame_data = {"frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES))}
        
        # Pose
        if 'pose' in detectors:
            res = detectors['pose'].detect_for_video(mp_image, timestamp_ms)
            if res.pose_landmarks:
                frame_data['pose'] = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in res.pose_landmarks[0]]

        # Hands
        if 'hand' in detectors:
            res = detectors['hand'].detect_for_video(mp_image, timestamp_ms)
            if res.hand_landmarks:
                frame_data['hand'] = [[{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand] for hand in res.hand_landmarks]

        # Face
        if 'face' in detectors:
            res = detectors['face'].detect_for_video(mp_image, timestamp_ms)
            if res.face_landmarks:
                frame_data['face'] = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in res.face_landmarks[0]]

        results_sequence.append(frame_data)

    # Cleanup
    for d in detectors.values(): d.close()
    cap.release()
    
    return {"status": "success", "results": results_sequence}

if __name__ == "__main__":
    uvicorn.run("mp_server:app", host="0.0.0.0", port=8000, reload=True)