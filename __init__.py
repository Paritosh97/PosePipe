"""
__init__.py
Desc: UI Addon
"""

bl_info = {
    "name": "PosePipe",
    "author": "ZonkoSoft, SpectralVectors, TwoOneOne",
    "version": (0, 8, 4),
    "blender": (2, 80, 0),
    "location": "3D View > Sidebar > PosePipe",
    "description": "Motion capture using your web camera or stream camera!",
    "category": "3D View",
    "wiki_url": "https://github.com/SpectralVectors/PosePipe/wiki",
    "tracker_url": "https://github.com/SpectralVectors/PosePipe/issues"
}

import pip
import pkg_resources
import bpy
import os
from bpy.types import Panel, Operator, PropertyGroup, FloatProperty, PointerProperty
from bpy.utils import register_class, unregister_class
from bpy_extras.io_utils import ImportHelper
from mathutils import Vector, Matrix
import time
import logging
import traceback
import textwrap

from PosePipe.core.Setups import *

def ShowMessageBox(text="Empty message", title="Message Box", icon='INFO'): 
    #Show popup window with message
    def draw(self, context):
        #single line
        #self.layout.label(text=text)

        #multiline wrap
        chars = int(200 / 7)   # 7 pix on 1 character | 200 width of dialog
        wrapper = textwrap.TextWrapper(width=chars)
        text_lines = wrapper.wrap(text=text)
        for text_line in text_lines:
            self.layout.label(text=text_line)
        
    bpy.context.window_manager.popup_menu(draw, title=title, icon=icon)

def batch_convert(file_dir):

    # loop through all videos in the directory
    for file in os.listdir(file_dir):
        if file.endswith(".mp4"):
            file_path = os.path.join(file_dir, file)
            run_full(file_path)

            # put skeleton on the generated mediapipe
            bpy.ops.pose.skeleton_builder()

            # bake the animation of the skeleton with visual transforms without clearing constraints
            bpy.ops.pose.bake(frame_start=0, frame_end=bpy.context.scene.frame_end, only_selected=False, visual_keying=True, clear_constraints=False, clear_parents=True, use_current_action=False, bake_types={'POSE'})

            # bake again with constraints cleared
            bpy.ops.pose.bake(frame_start=0, frame_end=bpy.context.scene.frame_end, only_selected=False, visual_keying=True, clear_constraints=True, clear_parents=True, use_current_action=False, bake_types={'POSE'})

            # export bvh
            bpy.ops.export_anim.bvh(filepath=os.path.join(file_dir, file.replace(".mp4", ".bvh")), check_existing=False, filter_glob="*.bvh", global_scale=1.0, frame_start=0, frame_end=bpy.context.scene.frame_end, rotate_mode='NATIVE', root_transform_only=False, bone_transform_only=False, use_custom_normals=False, use_keys=False, use_current_action=False, use_anim=False, use_selection=False, use_all_actions=False, axis_forward='Y', axis_up='Z')

            # delete all objects
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete(use_global=False)

def run_full(file_path):
    from PosePipe.engine.MediaPipe import MediaPipe

    import numpy as np
    import cv2
    
    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    
    settings = bpy.context.scene.settings
        
    try:
        bpy.ops.object.mode_set(mode='OBJECT')
    except:
        pass
    
    bpy.context.view_layer.objects.active = None

    try:
        bpy.ops.object.mode_set(mode='EDIT')
    except:
        pass

    if settings.body_tracking:
        body = body_setup()
    if settings.hand_tracking:
        hand_left, hand_right = hands_setup()
    if settings.face_tracking: 
        face = face_setup()

    try:
        if file_path == "None": 
            cap = cv2.VideoCapture(int(settings.camera_number))
                    
        if file_path != "None" and file_path != "Stream":
            cap = cv2.VideoCapture(file_path)
        elif file_path == "Stream":
            if "http" in str(settings.stream_url_string) or "rtsp:" in str(settings.stream_url_string):
                cap = cv2.VideoCapture()
                cap.open(settings.stream_url_string)
            else:
                ShowMessageBox(title="Error", icon='ERROR',text="Please enter url to connect. Ex.: http://<ip>/stream or rtsp://<ip>/")
                return
            
            if cap is None or not cap.isOpened():
                raise ConnectionError
            
    except Exception:
        ShowMessageBox(title="Error", icon='ERROR', text="Error on connect to resource.")
        return
    
    except ConnectionError:
        ShowMessageBox(title="Error", icon='ERROR', text="Camera or Stream cannot open.")
        return

    holistic = MediaPipe(settings=settings)

    n = int(1)
    previousTime = 0
    
    while True:
        if n > 9000: break

        success, image = cap.read()

        if not success:
            ShowMessageBox(title="Error", icon='ERROR', text="No camera present or empty stream.")
            break

        key = cv2.waitKey(33)

        if key == ord('q') or key == 27:
            break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        if file_path == "None" or settings.is_selfie == True:
            image = cv2.flip(image, 1)

        results = holistic.processImage(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        currentTime = time.time()
        capture_fps = int(1 / (currentTime - previousTime))
        previousTime = currentTime

        settings.capture_fps = capture_fps

        if settings.enable_segmentation == True:
            stack = np.stack((results.segmentation_mask,) * 3, axis=-1)
            if stack is not None:
                condition = stack > 0.1
                bg_image = np.zeros(image.shape, dtype=np.uint8)
                bg_image[:] = (192, 192, 192)
                image = np.where(condition, image, bg_image)

        cv2.putText(img=image, 
                    text='press long <ESC> or <Q> key to exit', 
                    org=(10,10), 
                    fontFace=cv2.FONT_HERSHEY_PLAIN, 
                    fontScale=1, 
                    color=(255,255,255), 
                    thickness=1)
        cv2.putText(img=image, 
                    text='FPS: ' + str(int(capture_fps)), 
                    org=(10,50), 
                    fontFace=cv2.FONT_HERSHEY_PLAIN, 
                    fontScale=2, 
                    color=(255,255,255), 
                    thickness=2)
        
        if int(settings.preview_size_enum) == 800:
            image = cv2.resize(image, (800, 600))

        if int(settings.preview_size_enum) < 10 and int(settings.preview_size_enum) > 1:
            h = int((image.shape[0]/int(settings.preview_size_enum)))
            w = int((image.shape[1]/int(settings.preview_size_enum)))
            image = cv2.resize(image, (w, h))
                
        cv2.imshow(f'MediaPipe Holistic {image.shape[1]}x{image.shape[0]}', image)

        if settings.body_tracking:
            if holistic.results.pose_landmarks:
                bns = [b for b in results.pose_landmarks.landmark]
                scale = 2
                bones = sorted(body.children, key=lambda b: b.name)

                for k in range(33):
                    try:
                        bones[k].location.y = bns[k].z / 4
                        bones[k].location.x = (0.5-bns[k].x)
                        bones[k].location.z = (0.2-bns[k].y) + 2
                        bones[k].keyframe_insert(data_path="location", frame=n)
                    except:
                        pass
                    
        if settings.hand_tracking:
            if holistic.results.left_hand_landmarks:
                bns = [b for b in holistic.results.left_hand_landmarks.landmark]
                scale = 2
                bones = sorted(hand_left.children, key=lambda b: b.name)
                for k in range(21):
                    try:
                        bones[k].location.y = bns[k].z
                        bones[k].location.x = (0.5-bns[k].x)
                        bones[k].location.z = (0.5-bns[k].y)/2 + 1.6
                        bones[k].keyframe_insert(data_path="location", frame=n)
                    except:
                        pass

            if holistic.results.right_hand_landmarks:
                bns = [b for b in holistic.results.right_hand_landmarks.landmark]
                scale = 2
                bones = sorted(hand_right.children, key=lambda b: b.name)
                for k in range(21):
                    try:
                        bones[k].location.y = bns[k].z
                        bones[k].location.x = (0.5-bns[k].x)
                        bones[k].location.z = (0.5-bns[k].y)/2 + 1.6
                        bones[k].keyframe_insert(data_path="location", frame=n)
                    except:
                        pass

        if settings.face_tracking:
            if holistic.results.face_landmarks:
                bns = [b for b in holistic.results.face_landmarks.landmark]
                scale = 2
                bones = sorted(face.children, key=lambda b: b.name)
                for k in range(468):
                    try:
                        bones[k].location.y = bns[k].z
                        bones[k].location.x = (0.5-bns[k].x)
                        bones[k].location.z = (0.2-bns[k].y) + 2
                        bones[k].keyframe_insert(data_path="location", frame=n)
                    except:
                        pass
        
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        bpy.context.scene.frame_set(n)
        n = n + 1

        bpy.context.scene.frame_end = n

    cap.release()
    cv2.destroyAllWindows()

    if settings.face_tracking:
        bpy.context.view_layer.objects.active = bpy.data.objects['Face']
        bpy.ops.object.constraint_add(type='COPY_LOCATION')
        do_assign(bpy.data.objects, "Face", "Copy Location", bpy.data.objects, "Pose")
        bpy.data.objects['Face'].constraints["Copy Location"].use_y = False
        bpy.ops.object.constraint_add(type='COPY_LOCATION')
        do_assign(bpy.data.objects, "Face", "Copy Location.001", bpy.data.objects, "00 nose")
        bpy.data.objects['Face'].constraints["Copy Location.001"].use_x = False
        bpy.data.objects['Face'].constraints["Copy Location.001"].use_z = False

    if settings.hand_tracking:
        bpy.context.view_layer.objects.active = bpy.data.objects['Hand Right']
        bpy.ops.object.constraint_add(type='COPY_LOCATION')
        do_assign(bpy.data.objects, "Hand Right", "Copy Location", bpy.data.objects, "Pose")
        bpy.data.objects['Hand Right'].constraints["Copy Location"].use_y = False
        bpy.ops.object.constraint_add(type='COPY_LOCATION')
        do_assign(bpy.data.objects, "Hand Right", "Copy Location.001", bpy.data.objects, "16 right wrist")
        bpy.data.objects['Hand Right'].constraints["Copy Location.001"].use_x = False
        bpy.data.objects['Hand Right'].constraints["Copy Location.001"].use_z = False 
        
        bpy.context.view_layer.objects.active = bpy.data.objects['Hand Left']
        bpy.ops.object.constraint_add(type='COPY_LOCATION')
        do_assign(bpy.data.objects, "Hand Left", "Copy Location", bpy.data.objects, "Pose")
        bpy.data.objects['Hand Left'].constraints["Copy Location"].use_y = False
        bpy.ops.object.constraint_add(type='COPY_LOCATION')
        do_assign(bpy.data.objects, "Hand Left", "Copy Location.001", bpy.data.objects, "15 left wrist")
        bpy.data.objects['Hand Left'].constraints["Copy Location.001"].use_x = False
        bpy.data.objects['Hand Left'].constraints["Copy Location.001"].use_z = False
        
    try:
        bpy.ops.object.mode_set(mode='OBJECT')
    except:
        pass


class RetimeAnimation(bpy.types.Operator):
    """Builds an armature to use with the mocap data"""
    bl_idname = "posepipe.retime_animation"
    bl_label = "Retime Animation"

    def execute(self, context):

        # Retime animation
        #bpy.data.objects['Pose'].select_set(True)
        scene_objects = [n for n in bpy.context.scene.objects.keys()]
        
        if "Body" in scene_objects:
            for c in bpy.context.scene.objects["Body"].children:
                bpy.data.objects[c.name].select_set(True)
        if "Hand Left" in scene_objects:
            for c in bpy.context.scene.objects["Hand Left"].children:
                bpy.data.objects[c.name].select_set(True)
        if "Hand Right" in scene_objects:
            for c in bpy.context.scene.objects["Hand Right"].children:
                bpy.data.objects[c.name].select_set(True)
        if "Face" in scene_objects:
            for c in bpy.context.scene.objects["Face"].children:
                bpy.data.objects[c.name].select_set(True)

        bpy.data.scenes['Scene'].frame_current = 0
        frame_rate = bpy.data.scenes['Scene'].render.fps
        timescale = frame_rate / bpy.context.scene.settings.capture_fps
        #bpy.context.area.type =  bpy.data.screens['Layout'].areas[2].type
        context.area.type = 'DOPESHEET_EDITOR'
        context.area.spaces[0].mode = 'TIMELINE'
        bpy.ops.transform.transform(mode='TIME_SCALE', value=(timescale, 0, 0, 0))
        #bpy.context.area.type = bpy.data.screens['Layout'].areas[-1].type
        context.area.type = 'VIEW_3D'
        return{'FINISHED'}

'''
def draw_file_opener(self, context):
    layout = self.layout
    scn = context.scene
    col = layout.column()
    row = col.row(align=True)
    row.prop(scn.settings, 'file_path', text='directory:')
    row.operator("something.identifier_selector", icon="FILE_FOLDER", text="")
'''

class RunFileSelector(Operator, ImportHelper):
    bl_idname = "something.identifier_selector"
    bl_label = "Select Video File"
    filename_ext = ""

    def execute(self, context):
        file_dir = self.properties.filepath
        run_full(file_dir)
        return{'FINISHED'}
    
class BatchConvert(Operator, ImportHelper):
    bl_idname = "object.batch_convert"
    bl_label = "Select Video Folder"
    filename_ext = ""

    def execute(self, context):
        file_dir = self.properties.filepath
        batch_convert(file_dir)
        return{'FINISHED'}

class RunOperator(Operator):
    bl_idname = "object.run_body_operator"
    bl_label = "Run Body Operator"

    def execute(self, context):
        run_full("None")
        return {'FINISHED'}

class RunOperatorStream(Operator):
    bl_idname = "object.connect_camera_stream"
    bl_label = "Connect to camera stream"

    def execute(self, context):
        run_full("Stream")
        return {'FINISHED'}

class Settings(PropertyGroup):
    # Capture only body pose if True, otherwise capture hands, face and body

    preview_size_options = [
        #value        #Description  #
        ("0", "Default", ""),
        ("800", "800x600", ""),
        ("2", "-2x", ""),
        ("3", "-3x", ""),
        ("4", "-4x", ""),
    ]

    stream_url_string: bpy.props.StringProperty(
        name="Url",
        description="Write url like http://192.168.0.100/stream",
        default="",
    )

    is_selfie: bpy.props.BoolProperty(default=False)

    face_tracking: bpy.props.BoolProperty(default=False)
    hand_tracking: bpy.props.BoolProperty(default=False)
    body_tracking: bpy.props.BoolProperty(default=True)

    preview_size_enum: bpy.props.EnumProperty(
        name="Size", 
        items=preview_size_options,
        description="Size of preview window",
        default="0",
    )
    
    camera_number: bpy.props.IntProperty(
        default=0, 
        soft_min=0, 
        soft_max=10, 
        description="If you have more than one camera, you can choose here. 0 should work for most users."
    )
    
    tracking_confidence: bpy.props.FloatProperty(
        default=0.5,
        soft_min=0.1,
        soft_max=1,
        description="Minimum level of data necessary to track, higher numbers = higher latency."
    )
    
    detection_confidence: bpy.props.FloatProperty(
        default=0.5,
        soft_min=0.1,
        soft_max=1,
        description="Minimum level of data necessary to detect, higher numbers = higher latency."
    )
    
    smooth_landmarks: bpy.props.BoolProperty(
        default=True,
        description="If True, applies a smoothing pass to the tracked data."
    )
    
    enable_segmentation: bpy.props.BoolProperty(
        default=False,
        description="Addition to the pose landmarks the solution also generates the segmentation mask."
    )

    smooth_segmentation: bpy.props.BoolProperty(
        default=True,
        description="Solution filters segmentation masks across different input images to reduce jitter."
    )
    
    model_complexity: bpy.props.IntProperty(
        default=1,
        soft_min=0,
        soft_max=2,
        description='Complexity of the tracking model, higher numbers = higher latency'
    )

    capture_fps: bpy.props.IntProperty(
        default=0,
        description='Framerate of the motion capture'
    )
    
class SkeletonBuilder(bpy.types.Operator):
    """Builds an armature to use with the mocap data"""
    bl_idname = "pose.skeleton_builder"
    bl_label = "Skeleton Builder"

    def execute(self, context):
        settings = bpy.context.scene.settings

        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except:
            pass

        bpy.context.scene.frame_set(0)

        bpy.context.scene.view_layers.update()

        bpy.ops.object.armature_add(radius=0.1)

        arm_obj = bpy.context.object
        arm_obj.name = "BAZeel.body.rig"

        bpy.data.armatures['Armature'].name = "BAZeel.body.rig"
        Body_Skeleton = bpy.data.armatures["BAZeel.body.rig"]
        Body_Skeleton.display_type = 'STICK'

        armature = {
            "mixamorig:Hips": None,
            "mixamorig:Spine.001": "mixamorig:Hips",
            "mixamorig:Spine1.001": "mixamorig:Spine.001",
            "mixamorig:Spine2.001": "mixamorig:Spine1.001",
            "mixamorig:Neck.001": "mixamorig:Spine2.001",
            "mixamorig:Head.001": "mixamorig:Neck.001",
            "mixamorig:LeftUpLeg": "mixamorig:Hips",
            "mixamorig:LeftLeg": "mixamorig:LeftUpLeg",
            "mixamorig:LeftFoot": "mixamorig:LeftLeg",
            "mixamorig:RightUpLeg": "mixamorig:Hips",
            "mixamorig:RightLeg": "mixamorig:RightUpLeg",
            "mixamorig:RightFoot": "mixamorig:RightLeg",
            "mixamorig:LeftShoulder.001": "mixamorig:Spine2.001",
            "mixamorig:LeftArm.001": "mixamorig:LeftShoulder.001",
            "mixamorig:LeftForeArm.001": "mixamorig:LeftArm.001",
            "mixamorig:RightShoulder.001": "mixamorig:Spine2.001",
            "mixamorig:RightArm.001": "mixamorig:RightShoulder.001",
            "mixamorig:RightForeArm.001": "mixamorig:RightArm.001",
            "mixamorig:LeftHand.001": "mixamorig:LeftForeArm.001",
            "mixamorig:LeftHandThumb1.001": "mixamorig:LeftHand.001",
            "mixamorig:LeftHandThumb2.001": "mixamorig:LeftHandThumb1.001",
            "mixamorig:LeftHandThumb3.001": "mixamorig:LeftHandThumb2.001",
            "mixamorig:LeftHandIndex1.001": "mixamorig:LeftHand.001",
            "mixamorig:LeftHandIndex2.001": "mixamorig:LeftHandIndex1.001",
            "mixamorig:LeftHandIndex3.001": "mixamorig:LeftHandIndex2.001",
            "mixamorig:LeftHandMiddle1.001": "mixamorig:LeftHand.001",
            "mixamorig:LeftHandMiddle2.001": "mixamorig:LeftHandMiddle1.001",
            "mixamorig:LeftHandMiddle3.001": "mixamorig:LeftHandMiddle2.001",
            "mixamorig:LeftHandRing1.001": "mixamorig:LeftHand.001",
            "mixamorig:LeftHandRing2.001": "mixamorig:LeftHandRing1.001",
            "mixamorig:LeftHandRing3.001": "mixamorig:LeftHandRing2.001",
            "mixamorig:LeftHandPinky1.001": "mixamorig:LeftHand.001",
            "mixamorig:LeftHandPinky2.001": "mixamorig:LeftHandPinky1.001",
            "mixamorig:LeftHandPinky3.001": "mixamorig:LeftHandPinky2.001",
            "mixamorig:RightHand.001": "mixamorig:RightForeArm.001",
            "mixamorig:RightHandThumb1.001": "mixamorig:RightHand.001",
            "mixamorig:RightHandThumb2.001": "mixamorig:RightHandThumb1.001",
            "mixamorig:RightHandThumb3.001": "mixamorig:RightHandThumb2.001",
            "mixamorig:RightHandIndex1.001": "mixamorig:RightHand.001",
            "mixamorig:RightHandIndex2.001": "mixamorig:RightHandIndex1.001",
            "mixamorig:RightHandIndex3.001": "mixamorig:RightHandIndex2.001",
            "mixamorig:RightHandMiddle1.001": "mixamorig:RightHand.001",
            "mixamorig:RightHandMiddle2.001": "mixamorig:RightHandMiddle1.001",
            "mixamorig:RightHandMiddle3.001": "mixamorig:RightHandMiddle2.001",
            "mixamorig:RightHandRing1.001": "mixamorig:RightHand.001",
            "mixamorig:RightHandRing2.001": "mixamorig:RightHandRing1.001",
            "mixamorig:RightHandRing3.001": "mixamorig:RightHandRing2.001",
            "mixamorig:RightHandPinky1.001": "mixamorig:RightHand.001",
            "mixamorig:RightHandPinky2.001": "mixamorig:RightHandPinky1.001",
            "mixamorig:RightHandPinky3.001": "mixamorig:RightHandPinky2.001"
        }

        bone_location_references = {
            "mixamorig:Hips": ["23 left hip", "24 right hip"],
            "mixamorig:LeftShoulder.001": ["11 left shoulder"],
            "mixamorig:LeftArm.001": ["11 left shoulder"],
            "mixamorig:LeftForeArm.001": ["13 left elbow"],
            "mixamorig:RightShoulder.001": ["12 right shoulder"],
            "mixamorig:RightArm.001": ["12 right shoulder"],
            "mixamorig:RightForeArm.001": ["14 right elbow"],
            "mixamorig:LeftUpLeg": ["23 left hip"],
            "mixamorig:LeftLeg": ["25 left knee"],
            "mixamorig:LeftFoot": ["27 left ankle"],
            "mixamorig:RightUpLeg": ["24 right hip"],
            "mixamorig:RightLeg": ["26 right knee"],
            "mixamorig:RightFoot": ["28 right ankle"],
            "mixamorig:Neck.001": ["11 left shoulder", "12 right shoulder"],
            "mixamorig:Head.001": ["09 mouth (left)", "10 mouth (right)"],
            "mixamorig:RightHand.001": ["00Hand Right", "09Hand Right"],
            "mixamorig:RightHandThumb1.001": ["01Hand Right", "02Hand Right"],
            "mixamorig:RightHandThumb2.001": ["02Hand Right", "03Hand Right"],
            "mixamorig:RightHandThumb3.001": ["03Hand Right", "04Hand Right"],
            "mixamorig:RightHandIndex1.001": ["05Hand Right", "06Hand Right"],
            "mixamorig:RightHandIndex2.001": ["06Hand Right", "07Hand Right"],
            "mixamorig:RightHandIndex3.001": ["07Hand Right", "08Hand Right"],
            "mixamorig:RightHandMiddle1.001": ["09Hand Right", "10Hand Right"],
            "mixamorig:RightHandMiddle2.001": ["10Hand Right", "11Hand Right"],
            "mixamorig:RightHandMiddle3.001": ["11Hand Right", "12Hand Right"],
            "mixamorig:RightHandRing1.001": ["13Hand Right", "14Hand Right"],
            "mixamorig:RightHandRing2.001": ["14Hand Right", "15Hand Right"],
            "mixamorig:RightHandRing3.001": ["15Hand Right", "16Hand Right"],
            "mixamorig:RightHandPinky1.001": ["17Hand Right", "18Hand Right"],
            "mixamorig:RightHandPinky2.001": ["18Hand Right", "19Hand Right"],
            "mixamorig:RightHandPinky3.001": ["19Hand Right", "20Hand Right"],
            "mixamorig:LeftHand.001": ["00Hand Left", "09Hand Left"],
            "mixamorig:LeftHandThumb1.001": ["01Hand Left", "02Hand Left"],
            "mixamorig:LeftHandThumb2.001": ["02Hand Left", "03Hand Left"],
            "mixamorig:LeftHandThumb3.001": ["03Hand Left", "04Hand Left"],
            "mixamorig:LeftHandIndex1.001": ["05Hand Left", "06Hand Left"],
            "mixamorig:LeftHandIndex2.001": ["06Hand Left", "07Hand Left"],
            "mixamorig:LeftHandIndex3.001": ["07Hand Left", "08Hand Left"],
            "mixamorig:LeftHandMiddle1.001": ["09Hand Left", "10Hand Left"],
            "mixamorig:LeftHandMiddle2.001": ["10Hand Left", "11Hand Left"],
            "mixamorig:LeftHandMiddle3.001": ["11Hand Left", "12Hand Left"],
            "mixamorig:LeftHandRing1.001": ["13Hand Left", "14Hand Left"],
            "mixamorig:LeftHandRing2.001": ["14Hand Left", "15Hand Left"],
            "mixamorig:LeftHandRing3.001": ["15Hand Left", "16Hand Left"],
            "mixamorig:LeftHandPinky1.001": ["17Hand Left", "18Hand Left"],
            "mixamorig:LeftHandPinky2.001": ["18Hand Left", "19Hand Left"],
            "mixamorig:LeftHandPinky3.001": ["19Hand Left", "20Hand Left"]
        }

        bone_ik_references = {
            "mixamorig:Spine2.001": ["mixamorig:Neck.001", 3]
        }

        bone_length_references = {
            "mixamorig:LeftShoulder.001": ["12 right shoulder", 0.6],
            "mixamorig:LeftArm.001": ["13 left elbow", 1.0],
            "mixamorig:LeftForeArm.001": ["00Hand Left", 1.0],
            "mixamorig:RightShoulder.001": ["11 left shoulder", 0.6],
            "mixamorig:RightArm.001": ["14 right elbow", 1.0],
            "mixamorig:RightForeArm.001": ["00Hand Right", 1.0],
            "mixamorig:LeftUpLeg": ["25 left knee", 1.0],
            "mixamorig:LeftLeg": ["27 left ankle", 1.0],
            "mixamorig:LeftFoot": ["31 left foot index", 1.0],
            "mixamorig:RightUpLeg": ["26 right knee", 1.0],
            "mixamorig:RightLeg": ["28 right ankle", 1.0],
            "mixamorig:RightFoot": ["32 right foot index", 1.0],
            "mixamorig:RightHand.001": ["09Hand Right", 1.0],
            "mixamorig:RightHandThumb1.001": ["02Hand Right", 1.0],
            "mixamorig:RightHandThumb2.001": ["03Hand Right", 1.0],
            "mixamorig:RightHandThumb3.001": ["04Hand Right", 1.0],
            "mixamorig:RightHandIndex1.001": ["06Hand Right", 1.0],
            "mixamorig:RightHandIndex2.001": ["07Hand Right", 1.0],
            "mixamorig:RightHandIndex3.001": ["08Hand Right", 1.0],
            "mixamorig:RightHandMiddle1.001": ["10Hand Right", 1.0],
            "mixamorig:RightHandMiddle2.001": ["11Hand Right", 1.0],
            "mixamorig:RightHandMiddle3.001": ["12Hand Right", 1.0],
            "mixamorig:RightHandRing1.001": ["14Hand Right", 1.0],
            "mixamorig:RightHandRing2.001": ["15Hand Right", 1.0],
            "mixamorig:RightHandRing3.001": ["16Hand Right", 1.0],
            "mixamorig:RightHandPinky1.001": ["18Hand Right", 1.0],
            "mixamorig:RightHandPinky2.001": ["19Hand Right", 1.0],
            "mixamorig:RightHandPinky3.001": ["20Hand Right", 1.0],
            "mixamorig:LeftHand.001": ["09Hand Left", 1.0],
            "mixamorig:LeftHandThumb1.001": ["02Hand Left", 1.0],
            "mixamorig:LeftHandThumb2.001": ["03Hand Left", 1.0],
            "mixamorig:LeftHandThumb3.001": ["04Hand Left", 1.0],
            "mixamorig:LeftHandIndex1.001": ["06Hand Left", 1.0],
            "mixamorig:LeftHandIndex2.001": ["07Hand Left", 1.0],
            "mixamorig:LeftHandIndex3.001": ["08Hand Left", 1.0],
            "mixamorig:LeftHandMiddle1.001": ["10Hand Left", 1.0],
            "mixamorig:LeftHandMiddle2.001": ["11Hand Left", 1.0],
            "mixamorig:LeftHandMiddle3.001": ["12Hand Left", 1.0],
            "mixamorig:LeftHandRing1.001": ["14Hand Left", 1.0],
            "mixamorig:LeftHandRing2.001": ["15Hand Left", 1.0],
            "mixamorig:LeftHandRing3.001": ["16Hand Left", 1.0],
            "mixamorig:LeftHandPinky1.001": ["18Hand Left", 1.0],
            "mixamorig:LeftHandPinky2.001": ["19Hand Left", 1.0],
            "mixamorig:LeftHandPinky3.001": ["20Hand Left", 1.0],
        }

        def create_bone(name, tail_z, parent_name=None):
            bpy.ops.armature.bone_primitive_add(name=name)
            bone = bpy.context.object.data.edit_bones[name]
            bone.tail[2] = tail_z
            if parent_name:
                bone.parent = bpy.context.object.data.edit_bones[parent_name]
            return bone

        def position_bone_edit(name, targets):
            bone = arm_obj.data.edit_bones[name]
            head_location_global = sum([bpy.data.objects[target].matrix_world.translation for target in targets], Vector()) / len(targets)
            bone.head = head_location_global
            bone.tail = head_location_global + Vector((0, 0, 0.1))
            if name in bone_length_references:
                # interpolate to find location
                target, length = bone_length_references[name]
                bone.tail = (1 - length) * head_location_global + length * bpy.data.objects[target].matrix_world.translation

        try:
        bpy.data.armatures["Body_Skeleton"].bones["Bone"].name = list(armature.keys())[0] 

        bpy.ops.object.mode_set(mode='EDIT')

        for bone_name, parent_name in armature.items():
            if parent_name:
                create_bone(bone_name, 0.1, parent_name)

        for bone_name, targets in bone_location_references.items():
            position_bone_edit(bone_name, targets)

        # for spine chain
        for bone_name, (end_effector, chain_length) in bone_ik_references.items():
            bone = bpy.context.object.data.edit_bones.get(bone_name)
            chain = []
            for i in range(chain_length):
                chain.append(bone)
                bone = bone.parent

            for bone in reversed(chain):
                bone.head = bone.parent.tail
                bone.tail = bone.head + Vector((0, 0, 0.1))

        bpy.context.scene.view_layers.update()

        bpy.ops.object.mode_set(mode='POSE')

        # Set the rest pose at frame 0
        bpy.ops.pose.armature_apply()
        bpy.ops.pose.armature_apply(selected=False)

        #for bone_name, (end_effector, chain_length) in bone_ik_references.items():
            #bone = arm_obj.pose.bones.get(bone_name)
            #constraint = bone.constraints.new("IK")
            #constraint.target = arm_obj
            #constraint.subtarget = end_effector
            #constraint.chain_count = chain_length

        # loop through frames and set keyframes
        #for frame in range(0, bpy.context.scene.frame_end):
            #bpy.context.scene.frame_set(frame)

            #for bone_name, targets in bone_location_references.items():
                #position_bone_pose(bone_name, targets)


        # Hide the trackers
        hide_trackers = ['Body', 'Hand Left', 'Hand Right', 'Face',
                         '17 left pinky', '18 right pinky', '19 left index',
                         '20 right index', '21 left thumb', '22 right thumb']

        for tracker in hide_trackers:
            try:
                bpy.data.objects[tracker].hide_set(True)
            except Exception as exception:
                logging.error(traceback.format_exc())

        face_trackers = ['01 left eye (inner)', '02 left eye', '03 left eye (outer)',
                         '04 right eye (inner)', '05 right eye', '06 right eye (outer)',
                         '09 mouth (left)', '10 mouth (right)']

        if settings.face_tracking:
            for tracker in face_trackers:
                try:
                    bpy.data.objects[tracker].hide_set(True)
                except Exception as exception:
                    logging.error(traceback.format_exc())

        return {'FINISHED'}

class AttachSkeleton(bpy.types.Operator):
    """Attach skeleton to mediapipe"""
    bl_idname = "posepipe.attach_skeleton"
    bl_label = "Place Bones on objects"

    def execute(self, context):
        body_names = [
            "00 nose", "01 left eye (inner)", "02 left eye", "03 left eye (outer)", "04 right eye (inner)", "05 right eye", 
            "06 right eye (outer)", "07 left ear", "08 right ear", "09 mouth (left)", "10 mouth (right)", "11 left shoulder", 
            "12 right shoulder", "13 left elbow", "14 right elbow", "15 left wrist", "16 right wrist", "17 left pinky", 
            "18 right pinky", "19 left index", "20 right index", "21 left thumb", "22 right thumb", "23 left hip", "24 right hip", 
            "25 left knee", "26 right knee", "27 left ankle", "28 right ankle", "29 left heel", "30 right heel", "31 left foot index", 
            "32 right foot index"
        ]

        bpy.context.scene.frame_set(0)

        bpy.context.scene.view_layers.update()

        arm_obj = bpy.context.object

        armature = {
            "mixamorig:Hips": None,
            "mixamorig:Spine.001": "mixamorig:Hips",
            "mixamorig:Spine1.001": "mixamorig:Spine.001",
            "mixamorig:Spine2.001": "mixamorig:Spine1.001",
            "mixamorig:Neck.001": "mixamorig:Spine2.001",
            "mixamorig:Head.001": "mixamorig:Neck.001",
            "mixamorig:LeftUpLeg": "mixamorig:Hips",
            "mixamorig:LeftLeg": "mixamorig:LeftUpLeg",
            "mixamorig:LeftFoot": "mixamorig:LeftLeg",
            "mixamorig:RightUpLeg": "mixamorig:Hips",
            "mixamorig:RightLeg": "mixamorig:RightUpLeg",
            "mixamorig:RightFoot": "mixamorig:RightLeg",
            "mixamorig:LeftShoulder.001": "mixamorig:Spine2.001",
            "mixamorig:LeftArm.001": "mixamorig:LeftShoulder.001",
            "mixamorig:LeftForeArm.001": "mixamorig:LeftArm.001",
            "mixamorig:RightShoulder.001": "mixamorig:Spine2.001",
            "mixamorig:RightArm.001": "mixamorig:RightShoulder.001",
            "mixamorig:RightForeArm.001": "mixamorig:RightArm.001",
            "mixamorig:LeftHand.001": "mixamorig:LeftForeArm.001",
            "mixamorig:LeftHandThumb1.001": "mixamorig:LeftHand.001",
            "mixamorig:LeftHandThumb2.001": "mixamorig:LeftHandThumb1.001",
            "mixamorig:LeftHandThumb3.001": "mixamorig:LeftHandThumb2.001",
            "mixamorig:LeftHandIndex1.001": "mixamorig:LeftHand.001",
            "mixamorig:LeftHandIndex2.001": "mixamorig:LeftHandIndex1.001",
            "mixamorig:LeftHandIndex3.001": "mixamorig:LeftHandIndex2.001",
            "mixamorig:LeftHandMiddle1.001": "mixamorig:LeftHand.001",
            "mixamorig:LeftHandMiddle2.001": "mixamorig:LeftHandMiddle1.001",
            "mixamorig:LeftHandMiddle3.001": "mixamorig:LeftHandMiddle2.001",
            "mixamorig:LeftHandRing1.001": "mixamorig:LeftHand.001",
            "mixamorig:LeftHandRing2.001": "mixamorig:LeftHandRing1.001",
            "mixamorig:LeftHandRing3.001": "mixamorig:LeftHandRing2.001",
            "mixamorig:LeftHandPinky1.001": "mixamorig:LeftHand.001",
            "mixamorig:LeftHandPinky2.001": "mixamorig:LeftHandPinky1.001",
            "mixamorig:LeftHandPinky3.001": "mixamorig:LeftHandPinky2.001",
            "mixamorig:RightHand.001": "mixamorig:RightForeArm.001",
            "mixamorig:RightHandThumb1.001": "mixamorig:RightHand.001",
            "mixamorig:RightHandThumb2.001": "mixamorig:RightHandThumb1.001",
            "mixamorig:RightHandThumb3.001": "mixamorig:RightHandThumb2.001",
            "mixamorig:RightHandIndex1.001": "mixamorig:RightHand.001",
            "mixamorig:RightHandIndex2.001": "mixamorig:RightHandIndex1.001",
            "mixamorig:RightHandIndex3.001": "mixamorig:RightHandIndex2.001",
            "mixamorig:RightHandMiddle1.001": "mixamorig:RightHand.001",
            "mixamorig:RightHandMiddle2.001": "mixamorig:RightHandMiddle1.001",
            "mixamorig:RightHandMiddle3.001": "mixamorig:RightHandMiddle2.001",
            "mixamorig:RightHandRing1.001": "mixamorig:RightHand.001",
            "mixamorig:RightHandRing2.001": "mixamorig:RightHandRing1.001",
            "mixamorig:RightHandRing3.001": "mixamorig:RightHandRing2.001",
            "mixamorig:RightHandPinky1.001": "mixamorig:RightHand.001",
            "mixamorig:RightHandPinky2.001": "mixamorig:RightHandPinky1.001",
            "mixamorig:RightHandPinky3.001": "mixamorig:RightHandPinky2.001"
        }

        bone_location_references = {
            "mixamorig:Hips": ["23 left hip", "24 right hip"],
            "mixamorig:LeftShoulder.001": ["11 left shoulder"],
            "mixamorig:LeftArm.001": ["11 left shoulder"],
            "mixamorig:LeftForeArm.001": ["13 left elbow"],
            "mixamorig:RightShoulder.001": ["12 right shoulder"],
            "mixamorig:RightArm.001": ["12 right shoulder"],
            "mixamorig:RightForeArm.001": ["14 right elbow"],
            "mixamorig:LeftUpLeg": ["23 left hip"],
            "mixamorig:LeftLeg": ["25 left knee"],
            "mixamorig:LeftFoot": ["27 left ankle"],
            "mixamorig:RightUpLeg": ["24 right hip"],
            "mixamorig:RightLeg": ["26 right knee"],
            "mixamorig:RightFoot": ["28 right ankle"],
            "mixamorig:Neck.001": ["11 left shoulder", "12 right shoulder"],
            "mixamorig:Head.001": ["09 mouth (left)", "10 mouth (right)"],
            "mixamorig:RightHand.001": ["00Hand Right", "09Hand Right"],
            "mixamorig:RightHandThumb1.001": ["01Hand Right", "02Hand Right"],
            "mixamorig:RightHandThumb2.001": ["02Hand Right", "03Hand Right"],
            "mixamorig:RightHandThumb3.001": ["03Hand Right", "04Hand Right"],
            "mixamorig:RightHandIndex1.001": ["05Hand Right", "06Hand Right"],
            "mixamorig:RightHandIndex2.001": ["06Hand Right", "07Hand Right"],
            "mixamorig:RightHandIndex3.001": ["07Hand Right", "08Hand Right"],
            "mixamorig:RightHandMiddle1.001": ["09Hand Right", "10Hand Right"],
            "mixamorig:RightHandMiddle2.001": ["10Hand Right", "11Hand Right"],
            "mixamorig:RightHandMiddle3.001": ["11Hand Right", "12Hand Right"],
            "mixamorig:RightHandRing1.001": ["13Hand Right", "14Hand Right"],
            "mixamorig:RightHandRing2.001": ["14Hand Right", "15Hand Right"],
            "mixamorig:RightHandRing3.001": ["15Hand Right", "16Hand Right"],
            "mixamorig:RightHandPinky1.001": ["17Hand Right", "18Hand Right"],
            "mixamorig:RightHandPinky2.001": ["18Hand Right", "19Hand Right"],
            "mixamorig:RightHandPinky3.001": ["19Hand Right", "20Hand Right"],
            "mixamorig:LeftHand.001": ["00Hand Left", "09Hand Left"],
            "mixamorig:LeftHandThumb1.001": ["01Hand Left", "02Hand Left"],
            "mixamorig:LeftHandThumb2.001": ["02Hand Left", "03Hand Left"],
            "mixamorig:LeftHandThumb3.001": ["03Hand Left", "04Hand Left"],
            "mixamorig:LeftHandIndex1.001": ["05Hand Left", "06Hand Left"],
            "mixamorig:LeftHandIndex2.001": ["06Hand Left", "07Hand Left"],
            "mixamorig:LeftHandIndex3.001": ["07Hand Left", "08Hand Left"],
            "mixamorig:LeftHandMiddle1.001": ["09Hand Left", "10Hand Left"],
            "mixamorig:LeftHandMiddle2.001": ["10Hand Left", "11Hand Left"],
            "mixamorig:LeftHandMiddle3.001": ["11Hand Left", "12Hand Left"],
            "mixamorig:LeftHandRing1.001": ["13Hand Left", "14Hand Left"],
            "mixamorig:LeftHandRing2.001": ["14Hand Left", "15Hand Left"],
            "mixamorig:LeftHandRing3.001": ["15Hand Left", "16Hand Left"],
            "mixamorig:LeftHandPinky1.001": ["17Hand Left", "18Hand Left"],
            "mixamorig:LeftHandPinky2.001": ["18Hand Left", "19Hand Left"],
            "mixamorig:LeftHandPinky3.001": ["19Hand Left", "20Hand Left"]
        }

        bone_ik_references = {
            "mixamorig:Spine2.001": ["mixamorig:Neck.001", 3]
        }

        bone_length_references = {
            "mixamorig:LeftShoulder.001": ["12 right shoulder", 0.6],
            "mixamorig:LeftArm.001": ["13 left elbow", 1.0],
            "mixamorig:LeftForeArm.001": ["00Hand Left", 1.0],
            "mixamorig:RightShoulder.001": ["11 left shoulder", 0.6],
            "mixamorig:RightArm.001": ["14 right elbow", 1.0],
            "mixamorig:RightForeArm.001": ["00Hand Right", 1.0],
            "mixamorig:LeftUpLeg": ["25 left knee", 1.0],
            "mixamorig:LeftLeg": ["27 left ankle", 1.0],
            "mixamorig:LeftFoot": ["31 left foot index", 1.0],
            "mixamorig:RightUpLeg": ["26 right knee", 1.0],
            "mixamorig:RightLeg": ["28 right ankle", 1.0],
            "mixamorig:RightFoot": ["32 right foot index", 1.0],
            "mixamorig:RightHand.001": ["09Hand Right", 1.0],
            "mixamorig:RightHandThumb1.001": ["02Hand Right", 1.0],
            "mixamorig:RightHandThumb2.001": ["03Hand Right", 1.0],
            "mixamorig:RightHandThumb3.001": ["04Hand Right", 1.0],
            "mixamorig:RightHandIndex1.001": ["06Hand Right", 1.0],
            "mixamorig:RightHandIndex2.001": ["07Hand Right", 1.0],
            "mixamorig:RightHandIndex3.001": ["08Hand Right", 1.0],
            "mixamorig:RightHandMiddle1.001": ["10Hand Right", 1.0],
            "mixamorig:RightHandMiddle2.001": ["11Hand Right", 1.0],
            "mixamorig:RightHandMiddle3.001": ["12Hand Right", 1.0],
            "mixamorig:RightHandRing1.001": ["14Hand Right", 1.0],
            "mixamorig:RightHandRing2.001": ["15Hand Right", 1.0],
            "mixamorig:RightHandRing3.001": ["16Hand Right", 1.0],
            "mixamorig:RightHandPinky1.001": ["18Hand Right", 1.0],
            "mixamorig:RightHandPinky2.001": ["19Hand Right", 1.0],
            "mixamorig:RightHandPinky3.001": ["20Hand Right", 1.0],
            "mixamorig:LeftHand.001": ["09Hand Left", 1.0],
            "mixamorig:LeftHandThumb1.001": ["02Hand Left", 1.0],
            "mixamorig:LeftHandThumb2.001": ["03Hand Left", 1.0],
            "mixamorig:LeftHandThumb3.001": ["04Hand Left", 1.0],
            "mixamorig:LeftHandIndex1.001": ["06Hand Left", 1.0],
            "mixamorig:LeftHandIndex2.001": ["07Hand Left", 1.0],
            "mixamorig:LeftHandIndex3.001": ["08Hand Left", 1.0],
            "mixamorig:LeftHandMiddle1.001": ["10Hand Left", 1.0],
            "mixamorig:LeftHandMiddle2.001": ["11Hand Left", 1.0],
            "mixamorig:LeftHandMiddle3.001": ["12Hand Left", 1.0],
            "mixamorig:LeftHandRing1.001": ["14Hand Left", 1.0],
            "mixamorig:LeftHandRing2.001": ["15Hand Left", 1.0],
            "mixamorig:LeftHandRing3.001": ["16Hand Left", 1.0],
            "mixamorig:LeftHandPinky1.001": ["18Hand Left", 1.0],
            "mixamorig:LeftHandPinky2.001": ["19Hand Left", 1.0],
            "mixamorig:LeftHandPinky3.001": ["20Hand Left", 1.0],
        }

        bpy.ops.object.mode_set(mode='POSE')

        def position_bone(name, targets):
            bone = arm_obj.data.edit_bones[name]
            head_location_global = sum([bpy.data.objects[target].matrix_world.translation for target in targets], Vector()) / len(targets)
            bone.head = head_location_global
            bone.tail = head_location_global + Vector((0, 0, 0.1))
            if name in bone_length_references:
                # interpolate to find location
                target, length = bone_length_references[name]
                bone.tail = (1 - length) * head_location_global + length * bpy.data.objects[target].matrix_world.translation

        for bone_name, targets in bone_location_references.items():
            position_bone(bone_name, targets)

        # for spine chain
        for bone_name, (end_effector, chain_length) in bone_ik_references.items():
            bone = bpy.context.object.data.edit_bones.get(bone_name)
            chain = []
            for i in range(chain_length):
                chain.append(bone)
                bone = bone.parent

            for bone in reversed(chain):
                bone.head = bone.parent.tail
                bone.tail = bone.head + Vector((0, 0, 0.1))

        bpy.context.scene.view_layers.update()


        return {'FINISHED'}

    
class PosePipePanel(Panel):
    bl_label = "PosePipe - Camera MoCap"
    bl_category = "PosePipe"
    bl_idname = "VIEW3D_PT_Pose"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'

    def draw(self, context):

        settings = context.scene.settings

        layout = self.layout
        
        box = layout.box()
        column_flow = box.column_flow()
        column = column_flow.column(align=True)
        column.label(text="Camera Settings:", icon='VIEW_CAMERA')
        split = column.split(factor=0.6)
        split.prop(settings, 'camera_number', text='Camera: ')
        split.label(text="to Exit", icon='EVENT_ESC')
        column.operator(RunOperator.bl_idname, text="Start Camera", icon='CAMERA_DATA')
        
        box = layout.box()
        column_flow = box.column_flow()
        column = column_flow.column(align=True)
        column.label(text="Stream:", icon='WORLD')
        column.prop(settings, "stream_url_string")
        column.operator(RunOperatorStream.bl_idname, text="Start Stream", icon='LIBRARY_DATA_DIRECT')
                       
        box = layout.box()
        column_flow = box.column_flow()
        column = column_flow.column(align=True)
        column.label(text="Process from file:", icon='FILE_MOVIE')
        column.operator(RunFileSelector.bl_idname, text="Load Video File", icon='FILE_BLANK')

        box = layout.box()
        column_flow = box.column_flow()
        column = column_flow.column(align=True)
        column.label(text="Batch Convert:", icon='FILE_FOLDER')
        column.operator(BatchConvert.bl_idname, text="Convert videos into bvh", icon='FILE_FOLDER')

        box = layout.box()
        column_flow = box.column_flow()
        column = column_flow.column(align=True)
        column.label(text="Preview window size:", icon='CON_SIZELIKE')
        column.prop(settings, 'preview_size_enum')


        box = layout.box()
        column_flow = box.column_flow()
        column = column_flow.column(align=True)
        column.label(text="Capture Mode:", icon='MOD_ARMATURE')
        column.prop(settings, 'body_tracking', text='Body', icon='ARMATURE_DATA')
        column.prop(settings, 'hand_tracking', text='Hands', icon='VIEW_PAN')
        column.prop(settings, 'face_tracking', text='Face', icon='MONKEY')
        column.label(text='Capture Settings:', icon='PREFERENCES')
        
        column.prop(settings, 'is_selfie', text='Is Felfie? (flip Hor.)', icon='MOD_MIRROR')
        column.prop(settings, 'smooth_landmarks', text='Jitter Smoothing', icon='MOD_SMOOTH')
        column.prop(settings, 'enable_segmentation', text='Enable Mask', icon='MOD_MASK')
        column.prop(settings, 'smooth_segmentation', text='Smooth Mask', icon='SMOOTHCURVE')
        
        column.prop(settings, 'model_complexity', text='Model Complexity:')
        column.prop(settings, 'detection_confidence', text='Detect Confidence:')
        column.prop(settings, 'tracking_confidence', text='Track Confidence:')
        
        box = layout.box()
        column_flow = box.column_flow()
        column = column_flow.column(align=True)
        column.label(text="Edit Capture Data:", icon='MODIFIER_ON')
        column.operator(RetimeAnimation.bl_idname, text="Retime Animation", icon='MOD_TIME')

        box = layout.box()
        column_flow = box.column_flow()
        column = column_flow.column(align=True)
        column.label(text="Armature:", icon='BONE_DATA')
        column.operator(SkeletonBuilder.bl_idname, text="Generate Bones", icon='ARMATURE_DATA')
        column.operator(AttachSkeleton.bl_idname, text="Attach Skeleton to mediapipe", icon='OBJECT_DATA')


# ----------------------------------------

class Install():
    def __init__(self):
        pipInstalledModules = [p.project_name for p in pkg_resources.working_set]
        
        for dep in depList.keys():
            for item in pipInstalledModules:
                if str(dep) in str(item):
                    depList[dep] = True
                    
    def check(self):
        valid = True
        for key, value in depList.items():
            if value == False:
                valid = False
                
        return valid

class PreUsagePanel(Panel):
    bl_label = "PosePipe - Camera MoCap"
    bl_category = "PosePipe"
    bl_idname = "VIEW3D_PT_Pose"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'

    def draw(self, context):

        settings = context.scene.settings

        layout = self.layout
        
        #checks of libraries
        box = layout.box()
        column_flow = box.column_flow()
        column = column_flow.column(align=True)
        column.label(text="Dependencies check:", icon='MEMORY')
                
        for key, value in depList.items():
            column.label(text=key, icon='CHECKBOX_HLT' if value else 'CHECKBOX_DEHLT')

        column.operator(RunInstallDependences.bl_idname, icon='PLUGIN')        

class RunInstallDependences(Operator):
    bl_idname = "pip.dep"
    bl_label = "Install dependencies"
    bl_info = "This button run installer for needed dependencies to run this plugin."

    def execute(self, context):
        self.report({'INFO'}, f"Run pip for install dependencies")
        
        for key, value in depList.items():
            if value == False:
                pip.main(['install', str(key)])
                depList[key] = True

        valid = Install().check()
        self.report({'INFO'}, f"All installed")

        if valid: 
            for c in _classes: 
                register_class(c)

        return {'FINISHED'}

# ----------------------------------------

dependencesController = None
depList = {
    "opencv-python":False,
    "mediapipe-silicon":False,
    "protobuf":False,
    "numpy":False,
    "ultralytics":False, #yolov8
}       

_classesPre = [
    PreUsagePanel,
    RunInstallDependences,
]

_classes = [
    PosePipePanel,
    RunOperator,
    RunOperatorStream,
    RunFileSelector,
    SkeletonBuilder,
    RetimeAnimation,
    AttachSkeleton,
    BatchConvert,
]

def register():
    register_class(Settings)
    
    dependencesController = Install()
    
    if dependencesController.check():
        for c in _classes: 
            register_class(c)
    else:
        for c in _classesPre: 
            register_class(c)
    
    bpy.types.Scene.settings = bpy.props.PointerProperty(type=Settings)
        
def unregister():
    for c in _classes: 
        unregister_class(c)
    del bpy.types.Scene.settings

if __name__ == "__main__":
    register()