import mediapipe as mp

class MediaPipe:

    def __init__(self, settings=None):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        
        self.settings = settings
        self.results = None

        self.holistic = self.mp_holistic.Holistic(
            model_complexity=settings.model_complexity,
            smooth_landmarks=settings.smooth_landmarks,
            min_detection_confidence=settings.detection_confidence,
            min_tracking_confidence=settings.tracking_confidence,
            enable_segmentation=settings.enable_segmentation,
            smooth_segmentation=settings.smooth_segmentation
        )

    def processImage(self, image):
        image.flags.writeable = True  # Ensure the image is writable
        out = self.holistic.process(image)
        
        self.results = out

        if self.settings.face_tracking: 
            self.mp_drawing.draw_landmarks(
                image, out.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION,
                self.mp_drawing.DrawingSpec(
                    color=(128,0,128), thickness=1, circle_radius=1
                ),
                self.mp_drawing.DrawingSpec(
                    color=(255,0,255), thickness=1, circle_radius=1
                )
            )
        
        if self.settings.hand_tracking:
            self.mp_drawing.draw_landmarks(
                image, out.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(
                    color=(128,0,0), thickness=1, circle_radius=3
                ),
                self.mp_drawing.DrawingSpec(
                    color=(255,0,0), thickness=3, circle_radius=1
                )
            )

            self.mp_drawing.draw_landmarks(
                image, out.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(
                    color=(0,0,128), thickness=1, circle_radius=3
                ),
                self.mp_drawing.DrawingSpec(
                    color=(0,0,255), thickness=3, circle_radius=1
                )
            )
        
        if self.settings.body_tracking:
            self.mp_drawing.draw_landmarks(
                image, out.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(
                    color=(0,128,0), thickness=1, circle_radius=2
                ),
                self.mp_drawing.DrawingSpec(
                    color=(0,255,0), thickness=2, circle_radius=1
                )
            )
        
        return out
