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
from bpy.types import Panel, Operator, PropertyGroup, FloatProperty, PointerProperty
from bpy.utils import register_class, unregister_class
from bpy_extras.io_utils import ImportHelper
import time
import logging
import traceback
import textwrap
import numpy as np
import cv2

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

def run_full(file_path):
    from PosePipe.engine.MediaPipe import MediaPipe
    
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
    

class SkeletonBuilder(Operator):
    """Builds an armature to use with the mocap data"""
    bl_idname = "pose.skeleton_builder"
    bl_label = "Skeleton Builder"

    def execute(self, context):

        settings = bpy.context.scene.settings
        
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except:
            pass

        bpy.ops.object.armature_add(radius=0.1)

        PosePipe_BodyBones = bpy.context.object
        PosePipe_BodyBones.name = "PosePipe_BodyBones"

        bpy.data.armatures['Armature'].name = "Body_Skeleton"
        Body_Skeleton = bpy.data.armatures["Body_Skeleton"]
        Body_Skeleton.display_type = 'STICK'

        try:
            bpy.data.armatures["Body_Skeleton"].bones["Bone"].name = "root"
        except:
            pass

        bpy.ops.object.editmode_toggle()

        root = None
        try:
            root = bpy.context.active_object.data.edit_bones["root"]
        except:
            root = None
        if (root == None):
            return {'FINISHED'}

        bpy.ops.armature.bone_primitive_add(name="pelvis")
        pelvis = bpy.context.active_object.data.edit_bones["pelvis"]
        bpy.context.active_object.data.edit_bones["pelvis"].tail[2] = 0.1
        pelvis.parent = root

        bpy.ops.armature.bone_primitive_add(name="spine01")
        spine01 = bpy.context.active_object.data.edit_bones["spine01"]
        bpy.context.active_object.data.edit_bones["spine01"].tail[2] = 0.1
        spine01.parent = pelvis

        bpy.ops.armature.bone_primitive_add(name="spine02")
        spine02 = bpy.context.active_object.data.edit_bones["spine02"]
        bpy.context.active_object.data.edit_bones["spine02"].tail[2] = 0.1
        spine02.parent = spine01

        bpy.ops.armature.bone_primitive_add(name="spine03")
        spine03 = bpy.context.active_object.data.edit_bones["spine03"]
        bpy.context.active_object.data.edit_bones["spine03"].tail[2] = 0.1
        spine03.parent = spine02

        bpy.ops.armature.bone_primitive_add(name="neck_01")
        neck_01 = bpy.context.active_object.data.edit_bones["neck_01"]
        bpy.context.active_object.data.edit_bones["neck_01"].tail[2] = 0.1
        neck_01.parent = spine03

        bpy.ops.armature.bone_primitive_add(name="head")
        head = bpy.context.active_object.data.edit_bones["head"]
        bpy.context.active_object.data.edit_bones["head"].tail[2] = 0.1
        head.parent = neck_01

        bpy.ops.armature.bone_primitive_add(name="thigh_l")
        thigh_l = bpy.context.active_object.data.edit_bones["thigh_l"]
        bpy.context.active_object.data.edit_bones["thigh_l"].tail[2] = 0.1
        thigh_l.parent = pelvis

        bpy.ops.armature.bone_primitive_add(name="calf_l")
        calf_l = bpy.context.active_object.data.edit_bones["calf_l"]
        bpy.context.active_object.data.edit_bones["calf_l"].tail[2] = 0.1
        calf_l.parent = thigh_l

        bpy.ops.armature.bone_primitive_add(name="foot_l")
        foot_l = bpy.context.active_object.data.edit_bones["foot_l"]
        bpy.context.active_object.data.edit_bones["foot_l"].tail[2] = 0.1
        foot_l.parent = calf_l

        bpy.ops.armature.bone_primitive_add(name="thigh_r")
        thigh_r = bpy.context.active_object.data.edit_bones["thigh_r"]
        bpy.context.active_object.data.edit_bones["thigh_r"].tail[2] = 0.1
        thigh_r.parent = pelvis

        bpy.ops.armature.bone_primitive_add(name="calf_r")
        calf_r = bpy.context.active_object.data.edit_bones["calf_r"]
        bpy.context.active_object.data.edit_bones["calf_r"].tail[2] = 0.1
        calf_r.parent = thigh_r

        bpy.ops.armature.bone_primitive_add(name="foot_r")
        foot_r = bpy.context.active_object.data.edit_bones["foot_r"]
        bpy.context.active_object.data.edit_bones["foot_r"].tail[2] = 0.1
        foot_r.parent = calf_r

        bpy.ops.armature.bone_primitive_add(name="clavicle_l")
        clavicle_l = bpy.context.active_object.data.edit_bones["clavicle_l"]
        bpy.context.active_object.data.edit_bones["clavicle_l"].tail[2] = 0.1
        clavicle_l.parent = spine03

        bpy.ops.armature.bone_primitive_add(name="upperarm_l")
        upperarm_l = bpy.context.active_object.data.edit_bones["upperarm_l"]
        bpy.context.active_object.data.edit_bones["upperarm_l"].tail[2] = 0.1
        upperarm_l.parent = clavicle_l

        bpy.ops.armature.bone_primitive_add(name="lowerarm_l")
        lowerarm_l = bpy.context.active_object.data.edit_bones["lowerarm_l"]
        bpy.context.active_object.data.edit_bones["lowerarm_l"].tail[2] = 0.1
        lowerarm_l.parent = upperarm_l

        bpy.ops.armature.bone_primitive_add(name="clavicle_r")
        clavicle_r = bpy.context.active_object.data.edit_bones["clavicle_r"]
        bpy.context.active_object.data.edit_bones["clavicle_r"].tail[2] = 0.1
        clavicle_r.parent = spine03

        bpy.ops.armature.bone_primitive_add(name="upperarm_r")
        upperarm_r = bpy.context.active_object.data.edit_bones["upperarm_r"]
        bpy.context.active_object.data.edit_bones["upperarm_r"].tail[2] = 0.1
        upperarm_r.parent = clavicle_r

        bpy.ops.armature.bone_primitive_add(name="lowerarm_r")
        lowerarm_r = bpy.context.active_object.data.edit_bones["lowerarm_r"]
        bpy.context.active_object.data.edit_bones["lowerarm_r"].tail[2] = 0.1
        lowerarm_r.parent = upperarm_r
        
        if settings.hand_tracking:
            bpy.ops.armature.bone_primitive_add(name="hand_l")
            hand_l = bpy.context.active_object.data.edit_bones["hand_l"]
            bpy.context.active_object.data.edit_bones["hand_l"].tail[2] = 0.1
            hand_l.parent = lowerarm_l

            bpy.ops.armature.bone_primitive_add(name="thumb_01_l")
            thumb_01_l = bpy.context.active_object.data.edit_bones["thumb_01_l"]
            bpy.context.active_object.data.edit_bones["thumb_01_l"].tail[2] = 0.1
            thumb_01_l.parent = hand_l

            bpy.ops.armature.bone_primitive_add(name="thumb_02_l")
            thumb_02_l = bpy.context.active_object.data.edit_bones["thumb_02_l"]
            bpy.context.active_object.data.edit_bones["thumb_02_l"].tail[2] = 0.1
            thumb_02_l.parent = thumb_01_l

            bpy.ops.armature.bone_primitive_add(name="thumb_03_l")
            thumb_03_l = bpy.context.active_object.data.edit_bones["thumb_03_l"]
            bpy.context.active_object.data.edit_bones["thumb_03_l"].tail[2] = 0.1
            thumb_03_l.parent = thumb_02_l

            bpy.ops.armature.bone_primitive_add(name="index_01_l")
            index_01_l = bpy.context.active_object.data.edit_bones["index_01_l"]
            bpy.context.active_object.data.edit_bones["index_01_l"].tail[2] = 0.1
            index_01_l.parent = hand_l

            bpy.ops.armature.bone_primitive_add(name="index_02_l")
            index_02_l = bpy.context.active_object.data.edit_bones["index_02_l"]
            bpy.context.active_object.data.edit_bones["index_02_l"].tail[2] = 0.1
            index_02_l.parent = index_01_l

            bpy.ops.armature.bone_primitive_add(name="index_03_l")
            index_03_l = bpy.context.active_object.data.edit_bones["index_03_l"]
            bpy.context.active_object.data.edit_bones["index_03_l"].tail[2] = 0.1
            index_03_l.parent = index_02_l

            bpy.ops.armature.bone_primitive_add(name="middle_01_l")
            middle_01_l = bpy.context.active_object.data.edit_bones["middle_01_l"]
            bpy.context.active_object.data.edit_bones["middle_01_l"].tail[2] = 0.1
            middle_01_l.parent = hand_l

            bpy.ops.armature.bone_primitive_add(name="middle_02_l")
            middle_02_l = bpy.context.active_object.data.edit_bones["middle_02_l"]
            bpy.context.active_object.data.edit_bones["middle_02_l"].tail[2] = 0.1
            middle_02_l.parent = middle_01_l

            bpy.ops.armature.bone_primitive_add(name="middle_03_l")
            middle_03_l = bpy.context.active_object.data.edit_bones["middle_03_l"]
            bpy.context.active_object.data.edit_bones["middle_03_l"].tail[2] = 0.1
            middle_03_l.parent = middle_02_l

            bpy.ops.armature.bone_primitive_add(name="ring_01_l")
            ring_01_l = bpy.context.active_object.data.edit_bones["ring_01_l"]
            bpy.context.active_object.data.edit_bones["ring_01_l"].tail[2] = 0.1
            ring_01_l.parent = hand_l

            bpy.ops.armature.bone_primitive_add(name="ring_02_l")
            ring_02_l = bpy.context.active_object.data.edit_bones["ring_02_l"]
            bpy.context.active_object.data.edit_bones["ring_02_l"].tail[2] = 0.1
            ring_02_l.parent = ring_01_l

            bpy.ops.armature.bone_primitive_add(name="ring_03_l")
            ring_03_l = bpy.context.active_object.data.edit_bones["ring_03_l"]
            bpy.context.active_object.data.edit_bones["ring_03_l"].tail[2] = 0.1
            ring_03_l.parent = ring_02_l

            bpy.ops.armature.bone_primitive_add(name="pinky_01_l")
            pinky_01_l = bpy.context.active_object.data.edit_bones["pinky_01_l"]
            bpy.context.active_object.data.edit_bones["pinky_01_l"].tail[2] = 0.1
            pinky_01_l.parent = hand_l

            bpy.ops.armature.bone_primitive_add(name="pinky_02_l")
            pinky_02_l = bpy.context.active_object.data.edit_bones["pinky_02_l"]
            bpy.context.active_object.data.edit_bones["pinky_02_l"].tail[2] = 0.1
            pinky_02_l.parent = pinky_01_l

            bpy.ops.armature.bone_primitive_add(name="pinky_03_l")
            pinky_03_l = bpy.context.active_object.data.edit_bones["pinky_03_l"]
            bpy.context.active_object.data.edit_bones["pinky_03_l"].tail[2] = 0.1
            pinky_03_l.parent = pinky_02_l

            bpy.ops.armature.bone_primitive_add(name="hand_r")
            hand_r = bpy.context.active_object.data.edit_bones["hand_r"]
            bpy.context.active_object.data.edit_bones["hand_r"].tail[2] = 0.1
            hand_r.parent = lowerarm_r

            bpy.ops.armature.bone_primitive_add(name="thumb_01_r")
            thumb_01_r = bpy.context.active_object.data.edit_bones["thumb_01_r"]
            bpy.context.active_object.data.edit_bones["thumb_01_r"].tail[2] = 0.1
            thumb_01_r.parent = hand_r

            bpy.ops.armature.bone_primitive_add(name="thumb_02_r")
            thumb_02_r = bpy.context.active_object.data.edit_bones["thumb_02_r"]
            bpy.context.active_object.data.edit_bones["thumb_02_r"].tail[2] = 0.1
            thumb_02_r.parent = thumb_01_r

            bpy.ops.armature.bone_primitive_add(name="thumb_03_r")
            thumb_03_r = bpy.context.active_object.data.edit_bones["thumb_03_r"]
            bpy.context.active_object.data.edit_bones["thumb_03_r"].tail[2] = 0.1
            thumb_03_r.parent = thumb_02_r

            bpy.ops.armature.bone_primitive_add(name="index_01_r")
            index_01_r = bpy.context.active_object.data.edit_bones["index_01_r"]
            bpy.context.active_object.data.edit_bones["index_01_r"].tail[2] = 0.1
            index_01_r.parent = hand_r

            bpy.ops.armature.bone_primitive_add(name="index_02_r")
            index_02_r = bpy.context.active_object.data.edit_bones["index_02_r"]
            bpy.context.active_object.data.edit_bones["index_02_r"].tail[2] = 0.1
            index_02_r.parent = index_01_r

            bpy.ops.armature.bone_primitive_add(name="index_03_r")
            index_03_r = bpy.context.active_object.data.edit_bones["index_03_r"]
            bpy.context.active_object.data.edit_bones["index_03_r"].tail[2] = 0.1
            index_03_r.parent = index_02_r

            bpy.ops.armature.bone_primitive_add(name="middle_01_r")
            middle_01_r = bpy.context.active_object.data.edit_bones["middle_01_r"]
            bpy.context.active_object.data.edit_bones["middle_01_r"].tail[2] = 0.1
            middle_01_r.parent = hand_r

            bpy.ops.armature.bone_primitive_add(name="middle_02_r")
            middle_02_r = bpy.context.active_object.data.edit_bones["middle_02_r"]
            bpy.context.active_object.data.edit_bones["middle_02_r"].tail[2] = 0.1
            middle_02_r.parent = middle_01_r

            bpy.ops.armature.bone_primitive_add(name="middle_03_r")
            middle_03_r = bpy.context.active_object.data.edit_bones["middle_03_r"]
            bpy.context.active_object.data.edit_bones["middle_03_r"].tail[2] = 0.1
            middle_03_r.parent = middle_02_r

            bpy.ops.armature.bone_primitive_add(name="ring_01_r")
            ring_01_r = bpy.context.active_object.data.edit_bones["ring_01_r"]
            bpy.context.active_object.data.edit_bones["ring_01_r"].tail[2] = 0.1
            ring_01_r.parent = hand_r

            bpy.ops.armature.bone_primitive_add(name="ring_02_r")
            ring_02_r = bpy.context.active_object.data.edit_bones["ring_02_r"]
            bpy.context.active_object.data.edit_bones["ring_02_r"].tail[2] = 0.1
            ring_02_r.parent = ring_01_r

            bpy.ops.armature.bone_primitive_add(name="ring_03_r")
            ring_03_r = bpy.context.active_object.data.edit_bones["ring_03_r"]
            bpy.context.active_object.data.edit_bones["ring_03_r"].tail[2] = 0.1
            ring_03_r.parent = ring_02_r

            bpy.ops.armature.bone_primitive_add(name="pinky_01_r")
            pinky_01_r = bpy.context.active_object.data.edit_bones["pinky_01_r"]
            bpy.context.active_object.data.edit_bones["pinky_01_r"].tail[2] = 0.1
            pinky_01_r.parent = hand_r

            bpy.ops.armature.bone_primitive_add(name="pinky_02_r")
            pinky_02_r = bpy.context.active_object.data.edit_bones["pinky_02_r"]
            bpy.context.active_object.data.edit_bones["pinky_02_r"].tail[2] = 0.1
            pinky_02_r.parent = pinky_01_r

            bpy.ops.armature.bone_primitive_add(name="pinky_03_r")
            pinky_03_r = bpy.context.active_object.data.edit_bones["pinky_03_r"]
            bpy.context.active_object.data.edit_bones["pinky_03_r"].tail[2] = 0.1
            pinky_03_r.parent = pinky_02_r

        bpy.ops.object.posemode_toggle()

        bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['pelvis'].bone
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        do_assign(bpy.context.object.pose.bones, "pelvis", "Copy Location", bpy.data.objects, "23 left hip")
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        do_assign(bpy.context.object.pose.bones, "pelvis", "Copy Location.001", bpy.data.objects, "24 right hip")
        bpy.context.object.pose.bones["pelvis"].constraints["Copy Location.001"].influence = 0.5

        bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['spine01'].bone
        #bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        PosePipe_BodyBones.pose.bones['spine01'].location[1] = 0.1
        PosePipe_BodyBones.pose.bones['spine02'].location[1] = 0.1
        PosePipe_BodyBones.pose.bones['spine03'].location[1] = 0.1
        PosePipe_BodyBones.pose.bones['neck_01'].location[1] = 0.1
        PosePipe_BodyBones.pose.bones['head'].location[1] = 0.1

        bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['spine03'].bone
        bpy.ops.pose.constraint_add(type='IK')
        do_assign(bpy.context.object.pose.bones, "spine03", "IK", PosePipe_BodyBones)
        bpy.context.object.pose.bones["spine03"].constraints["IK"].subtarget = 'neck_01'
        bpy.context.object.pose.bones["spine03"].constraints["IK"].chain_count = 3
        bpy.context.object.pose.bones["spine03"].constraints["IK"].pole_target = PosePipe_BodyBones
        bpy.context.object.pose.bones["spine03"].constraints["IK"].pole_subtarget = 'neck_01'

        bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['clavicle_l'].bone
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        do_assign(bpy.context.object.pose.bones, "clavicle_l", "Copy Location", bpy.data.objects, "11 left shoulder")
        bpy.ops.pose.constraint_add(type="STRETCH_TO")
        do_assign(bpy.context.object.pose.bones, "clavicle_l", "Stretch To", bpy.data.objects, "12 right shoulder")
        bpy.context.object.pose.bones["clavicle_l"].constraints['Stretch To'].rest_length = 0.4
        bpy.context.object.pose.bones["clavicle_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
        bpy.context.object.pose.bones["clavicle_l"].constraints['Stretch To'].keep_axis = 'PLANE_Z'

        bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['upperarm_l'].bone
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        do_assign(bpy.context.object.pose.bones, "upperarm_l", "Copy Location", bpy.data.objects, "11 left shoulder")
        bpy.ops.pose.constraint_add(type="STRETCH_TO")
        do_assign(bpy.context.object.pose.bones, "upperarm_l", "Stretch To", bpy.data.objects, "13 left elbow")
        bpy.context.object.pose.bones["upperarm_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
        bpy.context.object.pose.bones["upperarm_l"].constraints['Stretch To'].rest_length = 0.1

        bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['lowerarm_l'].bone
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        do_assign(bpy.context.object.pose.bones, "lowerarm_l", "Copy Location", bpy.data.objects, "13 left elbow")
        bpy.ops.pose.constraint_add(type="STRETCH_TO")
        if settings.body_tracking and settings.hand_tracking:
            do_assign(bpy.context.object.pose.bones, "lowerarm_l", "Stretch To", bpy.data.objects, "00Hand Left")
        else:
            do_assign(bpy.context.object.pose.bones, "lowerarm_l", "Stretch To", bpy.data.objects, "15 left wrist")
        bpy.context.object.pose.bones["lowerarm_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
        bpy.context.object.pose.bones["lowerarm_l"].constraints['Stretch To'].rest_length = 0.1

        bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['clavicle_r'].bone
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        do_assign(bpy.context.object.pose.bones, "clavicle_r", "Copy Location", bpy.data.objects, "12 right shoulder")
        bpy.ops.pose.constraint_add(type="STRETCH_TO")
        do_assign(bpy.context.object.pose.bones, "clavicle_r", "Stretch To", bpy.data.objects, "11 left shoulder")
        bpy.context.object.pose.bones["clavicle_r"].constraints['Stretch To'].rest_length = 0.4
        bpy.context.object.pose.bones["clavicle_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
        bpy.context.object.pose.bones["clavicle_r"].constraints['Stretch To'].keep_axis = 'PLANE_Z'

        bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['upperarm_r'].bone
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        do_assign(bpy.context.object.pose.bones, "upperarm_r", "Copy Location", bpy.data.objects, "12 right shoulder")
        bpy.ops.pose.constraint_add(type="STRETCH_TO")
        do_assign(bpy.context.object.pose.bones, "upperarm_r", "Stretch To", bpy.data.objects, "14 right elbow")
        bpy.context.object.pose.bones["upperarm_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
        bpy.context.object.pose.bones["upperarm_r"].constraints['Stretch To'].rest_length = 0.1

        bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['lowerarm_r'].bone
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        do_assign(bpy.context.object.pose.bones, "lowerarm_r", "Copy Location", bpy.data.objects, "14 right elbow")
        bpy.ops.pose.constraint_add(type="STRETCH_TO")
        if settings.body_tracking and settings.hand_tracking:
            do_assign(bpy.context.object.pose.bones, "lowerarm_r", "Stretch To", bpy.data.objects, "00Hand Right")
        else:
            do_assign(bpy.context.object.pose.bones, "lowerarm_r", "Stretch To", bpy.data.objects, "16 right wrist")
        bpy.context.object.pose.bones["lowerarm_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
        bpy.context.object.pose.bones["lowerarm_r"].constraints['Stretch To'].rest_length = 0.1

        bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['thigh_l'].bone
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        do_assign(bpy.context.object.pose.bones, "thigh_l", "Copy Location", bpy.data.objects, "23 left hip")
        bpy.ops.pose.constraint_add(type="STRETCH_TO")
        do_assign(bpy.context.object.pose.bones, "thigh_l", "Stretch To", bpy.data.objects, "25 left knee")
        bpy.context.object.pose.bones["thigh_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
        bpy.context.object.pose.bones["thigh_l"].constraints['Stretch To'].rest_length = 0.1

        bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['calf_l'].bone
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        do_assign(bpy.context.object.pose.bones, "calf_l", "Copy Location", bpy.data.objects, "25 left knee")
        bpy.ops.pose.constraint_add(type="STRETCH_TO")
        do_assign(bpy.context.object.pose.bones, "calf_l", "Stretch To", bpy.data.objects, "27 left ankle")
        bpy.context.object.pose.bones["calf_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
        bpy.context.object.pose.bones["calf_l"].constraints['Stretch To'].rest_length = 0.1

        bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['foot_l'].bone
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        do_assign(bpy.context.object.pose.bones, "foot_l", "Copy Location", bpy.data.objects, "27 left ankle")
        bpy.ops.pose.constraint_add(type="STRETCH_TO")
        do_assign(bpy.context.object.pose.bones, "foot_l", "Stretch To", bpy.data.objects, "31 left foot index")
        bpy.context.object.pose.bones["foot_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
        bpy.context.object.pose.bones["foot_l"].constraints['Stretch To'].rest_length = 0.1

        bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['thigh_r'].bone
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        do_assign(bpy.context.object.pose.bones, "thigh_r", "Copy Location", bpy.data.objects, "24 right hip")
        bpy.ops.pose.constraint_add(type="STRETCH_TO")
        do_assign(bpy.context.object.pose.bones, "thigh_r", "Stretch To", bpy.data.objects, "26 right knee")
        bpy.context.object.pose.bones["thigh_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
        bpy.context.object.pose.bones["thigh_r"].constraints['Stretch To'].rest_length = 0.1

        bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['calf_r'].bone
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        do_assign(bpy.context.object.pose.bones, "calf_r", "Copy Location", bpy.data.objects, "26 right knee")
        bpy.ops.pose.constraint_add(type="STRETCH_TO")
        do_assign(bpy.context.object.pose.bones, "calf_r", "Stretch To", bpy.data.objects, "28 right ankle")
        bpy.context.object.pose.bones["calf_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
        bpy.context.object.pose.bones["calf_r"].constraints['Stretch To'].rest_length = 0.1

        bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['foot_r'].bone
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        do_assign(bpy.context.object.pose.bones, "foot_r", "Copy Location", bpy.data.objects, "28 right ankle")
        bpy.ops.pose.constraint_add(type="STRETCH_TO")
        do_assign(bpy.context.object.pose.bones, "foot_r", "Stretch To", bpy.data.objects, "32 right foot index")
        bpy.context.object.pose.bones["foot_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
        bpy.context.object.pose.bones["foot_r"].constraints['Stretch To'].rest_length = 0.1

        bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['neck_01'].bone
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        do_assign(bpy.context.object.pose.bones, "neck_01", "Copy Location", bpy.data.objects, "11 left shoulder")
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        do_assign(bpy.context.object.pose.bones, "neck_01", "Copy Location.001", bpy.data.objects, "12 right shoulder")
        bpy.context.object.pose.bones["neck_01"].constraints["Copy Location.001"].influence = 0.5

        bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['head'].bone
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        do_assign(bpy.context.object.pose.bones, "head", "Copy Location", bpy.data.objects, "09 mouth (left)")
        bpy.context.object.pose.bones["head"].constraints['Copy Location'].use_y = False
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        do_assign(bpy.context.object.pose.bones, "head", "Copy Location.001", bpy.data.objects, "10 mouth (right)")
        bpy.context.object.pose.bones["head"].constraints["Copy Location.001"].influence = 0.5
        bpy.context.object.pose.bones["head"].constraints["Copy Location.001"].use_y = False
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        do_assign(bpy.context.object.pose.bones, "head", "Copy Location.002", bpy.data.objects, "08 right ear")
        bpy.context.object.pose.bones["head"].constraints["Copy Location.002"].use_x = False
        bpy.context.object.pose.bones["head"].constraints["Copy Location.002"].use_z = False

        if settings.hand_tracking:
            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['hand_r'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            do_assign(bpy.context.object.pose.bones, "hand_r", "Copy Location", bpy.data.objects, "00Hand Right")
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            do_assign(bpy.context.object.pose.bones, "hand_r", "Stretch To", bpy.data.objects, "09Hand Right")
            bpy.context.object.pose.bones["hand_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["hand_r"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['thumb_01_r'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            do_assign(bpy.context.object.pose.bones, "thumb_01_r", "Copy Location", bpy.data.objects, "01Hand Right")
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            do_assign(bpy.context.object.pose.bones, "thumb_01_r", "Stretch To", bpy.data.objects, "02Hand Right")
            bpy.context.object.pose.bones["thumb_01_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["thumb_01_r"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['thumb_02_r'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            do_assign(bpy.context.object.pose.bones, "thumb_02_r", "Copy Location", bpy.data.objects, "02Hand Right")
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            do_assign(bpy.context.object.pose.bones, "thumb_02_r", "Stretch To", bpy.data.objects, "03Hand Right")
            bpy.context.object.pose.bones["thumb_02_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["thumb_02_r"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['thumb_03_r'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            do_assign(bpy.context.object.pose.bones, "thumb_03_r", "Copy Location", bpy.data.objects, "03Hand Right")
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            do_assign(bpy.context.object.pose.bones, "thumb_03_r", "Stretch To", bpy.data.objects, "04Hand Right")
            bpy.context.object.pose.bones["thumb_03_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["thumb_03_r"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['index_01_r'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            do_assign(bpy.context.object.pose.bones, "index_01_r", "Copy Location", bpy.data.objects, "05Hand Right")
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            do_assign(bpy.context.object.pose.bones, "index_01_r", "Stretch To", bpy.data.objects, "06Hand Right")
            bpy.context.object.pose.bones["index_01_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["index_01_r"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['index_02_r'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            do_assign(bpy.context.object.pose.bones, "index_02_r", "Copy Location", bpy.data.objects, "06Hand Right")
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            do_assign(bpy.context.object.pose.bones, "index_02_r", "Stretch To", bpy.data.objects, "07Hand Right")
            bpy.context.object.pose.bones["index_02_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["index_02_r"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['index_03_r'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            do_assign(bpy.context.object.pose.bones, "index_03_r", "Copy Location", bpy.data.objects, "07Hand Right")
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            do_assign(bpy.context.object.pose.bones, "index_03_r", "Stretch To", bpy.data.objects, "08Hand Right")
            bpy.context.object.pose.bones["index_03_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["index_03_r"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['middle_01_r'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            do_assign(bpy.context.object.pose.bones, "middle_01_r", "Copy Location", bpy.data.objects, "09Hand Right")
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            do_assign(bpy.context.object.pose.bones, "middle_01_r", "Stretch To", bpy.data.objects, "10Hand Right")
            bpy.context.object.pose.bones["middle_01_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["middle_01_r"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['middle_02_r'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            do_assign(bpy.context.object.pose.bones, "middle_02_r", "Copy Location", bpy.data.objects, "10Hand Right")
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            do_assign(bpy.context.object.pose.bones, "middle_02_r", "Stretch To", bpy.data.objects, "11Hand Right")
            bpy.context.object.pose.bones["middle_02_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["middle_02_r"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['middle_03_r'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            do_assign(bpy.context.object.pose.bones, "middle_03_r", "Copy Location", bpy.data.objects, "11Hand Right")
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            do_assign(bpy.context.object.pose.bones, "middle_03_r", "Stretch To", bpy.data.objects, "12Hand Right")
            bpy.context.object.pose.bones["middle_03_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["middle_03_r"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['ring_01_r'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            do_assign(bpy.context.object.pose.bones, "ring_01_r", "Copy Location", bpy.data.objects, "13Hand Right")
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            do_assign(bpy.context.object.pose.bones, "ring_01_r", "Stretch To", bpy.data.objects, "14Hand Right")
            bpy.context.object.pose.bones["ring_01_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["ring_01_r"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['ring_02_r'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            do_assign(bpy.context.object.pose.bones, "ring_02_r", "Copy Location", bpy.data.objects, "14Hand Right")
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            do_assign(bpy.context.object.pose.bones, "ring_02_r", "Stretch To", bpy.data.objects, "15Hand Right")
            bpy.context.object.pose.bones["ring_02_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["ring_02_r"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['ring_03_r'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            do_assign(bpy.context.object.pose.bones, "ring_03_r", "Copy Location", bpy.data.objects, "15Hand Right")
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            do_assign(bpy.context.object.pose.bones, "ring_03_r", "Stretch To", bpy.data.objects, "16Hand Right")
            bpy.context.object.pose.bones["ring_03_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["ring_03_r"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['pinky_01_r'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            do_assign(bpy.context.object.pose.bones, "pinky_01_r", "Copy Location", bpy.data.objects, "17Hand Right")
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            do_assign(bpy.context.object.pose.bones, "pinky_01_r", "Stretch To", bpy.data.objects, "18Hand Right")
            bpy.context.object.pose.bones["pinky_01_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["pinky_01_r"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['pinky_02_r'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            do_assign(bpy.context.object.pose.bones, "pinky_02_r", "Copy Location", bpy.data.objects, "18Hand Right")
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            do_assign(bpy.context.object.pose.bones, "pinky_02_r", "Stretch To", bpy.data.objects, "19Hand Right")
            bpy.context.object.pose.bones["pinky_02_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["pinky_02_r"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['pinky_03_r'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            do_assign(bpy.context.object.pose.bones, "pinky_03_r", "Copy Location", bpy.data.objects, "19Hand Right")
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            do_assign(bpy.context.object.pose.bones, "pinky_03_r", "Stretch To", bpy.data.objects, "20Hand Right")
            bpy.context.object.pose.bones["pinky_03_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["pinky_03_r"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['hand_l'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            do_assign(bpy.context.object.pose.bones, "hand_l", "Copy Location", bpy.data.objects, "00Hand Left")
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            do_assign(bpy.context.object.pose.bones, "hand_l", "Stretch To", bpy.data.objects, "09Hand Left")
            bpy.context.object.pose.bones["hand_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["hand_l"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['thumb_01_l'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            do_assign(bpy.context.object.pose.bones, "thumb_01_l", "Copy Location", bpy.data.objects, "01Hand Left")
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            do_assign(bpy.context.object.pose.bones, "thumb_01_l", "Stretch To", bpy.data.objects, "02Hand Left")
            bpy.context.object.pose.bones["thumb_01_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["thumb_01_l"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['thumb_02_l'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            do_assign(bpy.context.object.pose.bones, "thumb_02_l", "Copy Location", bpy.data.objects, "02Hand Left")
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            do_assign(bpy.context.object.pose.bones, "thumb_02_l", "Stretch To", bpy.data.objects, "03Hand Left")
            bpy.context.object.pose.bones["thumb_02_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["thumb_02_l"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['thumb_03_l'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            do_assign(bpy.context.object.pose.bones, "thumb_03_l", "Copy Location", bpy.data.objects, "03Hand Left")
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            do_assign(bpy.context.object.pose.bones, "thumb_03_l", "Stretch To", bpy.data.objects, "04Hand Left")
            bpy.context.object.pose.bones["thumb_03_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["thumb_03_l"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['index_01_l'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            do_assign(bpy.context.object.pose.bones, "index_01_l", "Copy Location", bpy.data.objects, "05Hand Left")
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            do_assign(bpy.context.object.pose.bones, "index_01_l", "Stretch To", bpy.data.objects, "06Hand Left")
            bpy.context.object.pose.bones["index_01_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["index_01_l"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['index_02_l'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            do_assign(bpy.context.object.pose.bones, "index_02_l", "Copy Location", bpy.data.objects, "06Hand Left")
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            do_assign(bpy.context.object.pose.bones, "index_02_l", "Stretch To", bpy.data.objects, "07Hand Left")
            bpy.context.object.pose.bones["index_02_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["index_02_l"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['index_03_l'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            do_assign(bpy.context.object.pose.bones, "index_03_l", "Copy Location", bpy.data.objects, "07Hand Left")
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            do_assign(bpy.context.object.pose.bones, "index_03_l", "Stretch To", bpy.data.objects, "08Hand Left")
            bpy.context.object.pose.bones["index_03_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["index_03_l"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['middle_01_l'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            do_assign(bpy.context.object.pose.bones, "middle_01_l", "Copy Location", bpy.data.objects, "09Hand Left")
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            do_assign(bpy.context.object.pose.bones, "middle_01_l", "Stretch To", bpy.data.objects, "10Hand Left")
            bpy.context.object.pose.bones["middle_01_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["middle_01_l"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['middle_02_l'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            do_assign(bpy.context.object.pose.bones, "middle_02_l", "Copy Location", bpy.data.objects, "10Hand Left")
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            do_assign(bpy.context.object.pose.bones, "middle_02_l", "Stretch To", bpy.data.objects, "11Hand Left")
            bpy.context.object.pose.bones["middle_02_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["middle_02_l"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['middle_03_l'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            do_assign(bpy.context.object.pose.bones, "middle_03_l", "Copy Location", bpy.data.objects, "11Hand Left")
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            do_assign(bpy.context.object.pose.bones, "middle_03_l", "Stretch To", bpy.data.objects, "12Hand Left")
            bpy.context.object.pose.bones["middle_03_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["middle_03_l"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['ring_01_l'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            do_assign(bpy.context.object.pose.bones, "ring_01_l", "Copy Location", bpy.data.objects, "13Hand Left")
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            do_assign(bpy.context.object.pose.bones, "ring_01_l", "Stretch To", bpy.data.objects, "14Hand Left")
            bpy.context.object.pose.bones["ring_01_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["ring_01_l"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['ring_02_l'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            do_assign(bpy.context.object.pose.bones, "ring_02_l", "Copy Location", bpy.data.objects, "14Hand Left")
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            do_assign(bpy.context.object.pose.bones, "ring_02_l", "Stretch To", bpy.data.objects, "15Hand Left")
            bpy.context.object.pose.bones["ring_02_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["ring_02_l"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['ring_03_l'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            do_assign(bpy.context.object.pose.bones, "ring_03_l", "Copy Location", bpy.data.objects, "15Hand Left")
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            do_assign(bpy.context.object.pose.bones, "ring_03_l", "Stretch To", bpy.data.objects, "16Hand Left")
            bpy.context.object.pose.bones["ring_03_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["ring_03_l"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['pinky_01_l'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            do_assign(bpy.context.object.pose.bones, "pinky_01_l", "Copy Location", bpy.data.objects, "17Hand Left")
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            do_assign(bpy.context.object.pose.bones, "pinky_01_l", "Stretch To", bpy.data.objects, "18Hand Left")
            bpy.context.object.pose.bones["pinky_01_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["pinky_01_l"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['pinky_02_l'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            do_assign(bpy.context.object.pose.bones, "pinky_02_l", "Copy Location", bpy.data.objects, "18Hand Left")
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            do_assign(bpy.context.object.pose.bones, "pinky_02_l", "Stretch To", bpy.data.objects, "19Hand Left")
            bpy.context.object.pose.bones["pinky_02_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["pinky_02_l"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['pinky_03_l'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            do_assign(bpy.context.object.pose.bones, "pinky_03_l", "Copy Location", bpy.data.objects, "19Hand Left")
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            do_assign(bpy.context.object.pose.bones, "pinky_03_l", "Stretch To", bpy.data.objects, "20Hand Left")
            bpy.context.object.pose.bones["pinky_03_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["pinky_03_l"].constraints['Stretch To'].rest_length = 0.1

        hide_trackers = ['Body','Hand Left','Hand Right','Face',
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