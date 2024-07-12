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

import os
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

    count = 0

    # loop through all videos in the directory
    for file in os.listdir(file_dir):
        if file.endswith(".mp4"):
            file_path = os.path.join(file_dir, file)
            run_full(file_path)

            # put skeleton on the generated mediapipe
            bpy.ops.pose.skeleton_builder()

            # set rest pose
            bpy.context.scene.frame_set(0)

            # bake the animation of the skeleton with visual transforms without clearing constraints
            bpy.ops.nla.bake(frame_start=0, frame_end=bpy.context.scene.frame_end, only_selected=False, visual_keying=True, clear_constraints=False, clear_parents=True, use_current_action=False, bake_types={'POSE'})

            # bake again with constraints cleared
            bpy.ops.nla.bake(frame_start=0, frame_end=bpy.context.scene.frame_end, only_selected=False, visual_keying=True, clear_constraints=True, clear_parents=True, use_current_action=False, bake_types={'POSE'})

            # export bvh
            bpy.ops.export_anim.bvh(filepath=os.path.join(file_dir, file.replace(".mp4", ".bvh")), check_existing=False, filter_glob="*.bvh", frame_start=0, frame_end=bpy.context.scene.frame_end, rotate_mode='NATIVE')

            # delete all objects
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete(use_global=False)

            count += 1

            if count > 2:
                ShowMessageBox(title="Info", icon='INFO', text="Only 2 files can be converted at a time.")
                break

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

        # set frame_end
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

        bpy.ops.object.armature_add(radius=0.1)

        PosePipe_BodyBones = bpy.context.object
        PosePipe_BodyBones.name = "PosePipe_BodyBones"

        bpy.data.armatures['Armature'].name = "Body_Skeleton"
        Body_Skeleton = bpy.data.armatures["Body_Skeleton"]
        Body_Skeleton.display_type = 'STICK'

        try:
            bpy.data.armatures["Body_Skeleton"].bones["Bone"].name = "mixamorig:Hips"
        except:
            pass

        bpy.ops.object.editmode_toggle()

        def create_bone(name, tail_z, parent_name=None):
            bpy.ops.armature.bone_primitive_add(name=name)
            bone = bpy.context.object.data.edit_bones[name]
            bone.tail[2] = tail_z
            if parent_name:
                bone.parent = bpy.context.object.data.edit_bones[parent_name]
            return bone

        spine01 = create_bone("mixamorig:Spine.001", 0.1, "mixamorig:Hips")
        spine02 = create_bone("mixamorig:Spine1.001", 0.1, "mixamorig:Spine.001")
        spine03 = create_bone("mixamorig:Spine2.001", 0.1, "mixamorig:Spine1.001")
        neck_01 = create_bone("mixamorig:Neck.001", 0.1, "mixamorig:Spine2.001")
        head = create_bone("mixamorig:Head.001", 0.1, "mixamorig:Neck.001")

        thigh_l = create_bone("mixamorig:LeftUpLeg", 0.1, "mixamorig:Hips")
        calf_l = create_bone("mixamorig:LeftLeg", 0.1, "mixamorig:LeftUpLeg")
        foot_l = create_bone("mixamorig:LeftFoot", 0.1, "mixamorig:LeftLeg")

        thigh_r = create_bone("mixamorig:RightUpLeg", 0.1, "mixamorig:Hips")
        calf_r = create_bone("mixamorig:RightLeg", 0.1, "mixamorig:RightUpLeg")
        foot_r = create_bone("mixamorig:RightFoot", 0.1, "mixamorig:RightLeg")

        clavicle_l = create_bone("mixamorig:LeftShoulder.001", 0.1, "mixamorig:Spine2.001")
        upperarm_l = create_bone("mixamorig:LeftArm.001", 0.1, "mixamorig:LeftShoulder.001")
        lowerarm_l = create_bone("mixamorig:LeftForeArm.001", 0.1, "mixamorig:LeftArm.001")

        clavicle_r = create_bone("mixamorig:RightShoulder.001", 0.1, "mixamorig:Spine2.001")
        upperarm_r = create_bone("mixamorig:RightArm.001", 0.1, "mixamorig:RightShoulder.001")
        lowerarm_r = create_bone("mixamorig:RightForeArm.001", 0.1, "mixamorig:RightArm.001")

        if settings.hand_tracking:
            hand_bones = [
                {"name": "mixamorig:LeftHand.001", "tail_z": 0.1, "parent": "mixamorig:LeftForeArm.001"},
                {"name": "mixamorig:LeftHandThumb1.001", "tail_z": 0.1, "parent": "mixamorig:LeftHand.001"},
                {"name": "mixamorig:LeftHandThumb2.001", "tail_z": 0.1, "parent": "mixamorig:LeftHandThumb1.001"},
                {"name": "mixamorig:LeftHandThumb3.001", "tail_z": 0.1, "parent": "mixamorig:LeftHandThumb2.001"},
                {"name": "mixamorig:LeftHandIndex1.001", "tail_z": 0.1, "parent": "mixamorig:LeftHand.001"},
                {"name": "mixamorig:LeftHandIndex2.001", "tail_z": 0.1, "parent": "mixamorig:LeftHandIndex1.001"},
                {"name": "mixamorig:LeftHandIndex3.001", "tail_z": 0.1, "parent": "mixamorig:LeftHandIndex2.001"},
                {"name": "mixamorig:LeftHandMiddle1.001", "tail_z": 0.1, "parent": "mixamorig:LeftHand.001"},
                {"name": "mixamorig:LeftHandMiddle2.001", "tail_z": 0.1, "parent": "mixamorig:LeftHandMiddle1.001"},
                {"name": "mixamorig:LeftHandMiddle3.001", "tail_z": 0.1, "parent": "mixamorig:LeftHandMiddle2.001"},
                {"name": "mixamorig:LeftHandRing1.001", "tail_z": 0.1, "parent": "mixamorig:LeftHand.001"},
                {"name": "mixamorig:LeftHandRing2.001", "tail_z": 0.1, "parent": "mixamorig:LeftHandRing1.001"},
                {"name": "mixamorig:LeftHandRing3.001", "tail_z": 0.1, "parent": "mixamorig:LeftHandRing2.001"},
                {"name": "mixamorig:LeftHandPinky1.001", "tail_z": 0.1, "parent": "mixamorig:LeftHand.001"},
                {"name": "mixamorig:LeftHandPinky2.001", "tail_z": 0.1, "parent": "mixamorig:LeftHandPinky1.001"},
                {"name": "mixamorig:LeftHandPinky3.001", "tail_z": 0.1, "parent": "mixamorig:LeftHandPinky2.001"},
                {"name": "mixamorig:RightHand.001", "tail_z": 0.1, "parent": "mixamorig:RightForeArm.001"},
                {"name": "mixamorig:RightHandThumb1.001", "tail_z": 0.1, "parent": "mixamorig:RightHand.001"},
                {"name": "mixamorig:RightHandThumb2.001", "tail_z": 0.1, "parent": "mixamorig:RightHandThumb1.001"},
                {"name": "mixamorig:RightHandThumb3.001", "tail_z": 0.1, "parent": "mixamorig:RightHandThumb2.001"},
                {"name": "mixamorig:RightHandIndex1.001", "tail_z": 0.1, "parent": "mixamorig:RightHand.001"},
                {"name": "mixamorig:RightHandIndex2.001", "tail_z": 0.1, "parent": "mixamorig:RightHandIndex1.001"},
                {"name": "mixamorig:RightHandIndex3.001", "tail_z": 0.1, "parent": "mixamorig:RightHandIndex2.001"},
                {"name": "mixamorig:RightHandMiddle1.001", "tail_z": 0.1, "parent": "mixamorig:RightHand.001"},
                {"name": "mixamorig:RightHandMiddle2.001", "tail_z": 0.1, "parent": "mixamorig:RightHandMiddle1.001"},
                {"name": "mixamorig:RightHandMiddle3.001", "tail_z": 0.1, "parent": "mixamorig:RightHandMiddle2.001"},
                {"name": "mixamorig:RightHandRing1.001", "tail_z": 0.1, "parent": "mixamorig:RightHand.001"},
                {"name": "mixamorig:RightHandRing2.001", "tail_z": 0.1, "parent": "mixamorig:RightHandRing1.001"},
                {"name": "mixamorig:RightHandRing3.001", "tail_z": 0.1, "parent": "mixamorig:RightHandRing2.001"},
                {"name": "mixamorig:RightHandPinky1.001", "tail_z": 0.1, "parent": "mixamorig:RightHand.001"},
                {"name": "mixamorig:RightHandPinky2.001", "tail_z": 0.1, "parent": "mixamorig:RightHandPinky1.001"},
                {"name": "mixamorig:RightHandPinky3.001", "tail_z": 0.1, "parent": "mixamorig:RightHandPinky2.001"}
            ]
            
            for bone in hand_bones:
                create_bone(bone["name"], bone["tail_z"], bone["parent"])

        bpy.ops.object.posemode_toggle()

        def add_constraint(bone_name, constraint_type, target_name, subtarget=None):
            bone = PosePipe_BodyBones.pose.bones.get(bone_name)
            if not bone:
                print(f"Bone {bone_name} not found.")
                return
            constraint = bone.constraints.new(constraint_type)
            target = bpy.data.objects.get(target_name)
            if not target:
                print(f"Target {target_name} not found.")
                return
            constraint.target = target
            if subtarget:
                constraint.subtarget = subtarget

        add_constraint("mixamorig:Hips", "COPY_LOCATION", "23 left hip")
        add_constraint("mixamorig:Hips", "COPY_LOCATION", "24 right hip")
        PosePipe_BodyBones.pose.bones["mixamorig:Hips"].constraints["Copy Location.001"].influence = 0.5

        PosePipe_BodyBones.pose.bones["mixamorig:Spine.001"].location[1] = 0.1
        PosePipe_BodyBones.pose.bones["mixamorig:Spine1.001"].location[1] = 0.1
        PosePipe_BodyBones.pose.bones["mixamorig:Spine2.001"].location[1] = 0.1
        PosePipe_BodyBones.pose.bones["mixamorig:Neck.001"].location[1] = 0.1
        PosePipe_BodyBones.pose.bones["mixamorig:Head.001"].location[1] = 0.1

        add_constraint("mixamorig:Spine2.001", "IK", "PosePipe_BodyBones")
        PosePipe_BodyBones.pose.bones["mixamorig:Spine2.001"].constraints["IK"].subtarget = "mixamorig:Neck.001"
        PosePipe_BodyBones.pose.bones["mixamorig:Spine2.001"].constraints["IK"].chain_count = 3

        add_constraint("mixamorig:LeftShoulder.001", "COPY_LOCATION", "12 right shoulder")
        add_constraint("mixamorig:LeftShoulder.001", "COPY_LOCATION", "11 left shoulder")
        PosePipe_BodyBones.pose.bones["mixamorig:LeftShoulder.001"].constraints["Copy Location.001"].influence = 0.5
        add_constraint("mixamorig:LeftShoulder.001", "STRETCH_TO", "11 left shoulder")
        PosePipe_BodyBones.pose.bones["mixamorig:LeftShoulder.001"].constraints['Stretch To'].rest_length = 0.1
        PosePipe_BodyBones.pose.bones["mixamorig:LeftShoulder.001"].constraints['Stretch To'].volume = 'NO_VOLUME'
        PosePipe_BodyBones.pose.bones["mixamorig:LeftShoulder.001"].constraints['Stretch To'].keep_axis = 'PLANE_Z'

        add_constraint("mixamorig:LeftArm.001", "COPY_LOCATION", "11 left shoulder")
        add_constraint("mixamorig:LeftArm.001", "STRETCH_TO", "13 left elbow")
        PosePipe_BodyBones.pose.bones["mixamorig:LeftArm.001"].constraints['Stretch To'].volume = 'NO_VOLUME'
        PosePipe_BodyBones.pose.bones["mixamorig:LeftArm.001"].constraints['Stretch To'].rest_length = 0.1

        add_constraint("mixamorig:LeftForeArm.001", "COPY_LOCATION", "13 left elbow")
        if settings.body_tracking and settings.hand_tracking:
            add_constraint("mixamorig:LeftForeArm.001", "STRETCH_TO", "00Hand Left")
        else:
            add_constraint("mixamorig:LeftForeArm.001", "STRETCH_TO", "15 left wrist")
        PosePipe_BodyBones.pose.bones["mixamorig:LeftForeArm.001"].constraints['Stretch To'].volume = 'NO_VOLUME'
        PosePipe_BodyBones.pose.bones["mixamorig:LeftForeArm.001"].constraints['Stretch To'].rest_length = 0.1

        add_constraint("mixamorig:RightShoulder.001", "COPY_LOCATION", "11 left shoulder")
        add_constraint("mixamorig:RightShoulder.001", "COPY_LOCATION", "12 right shoulder")
        add_constraint("mixamorig:RightShoulder.001", "STRETCH_TO", "12 right shoulder")
        PosePipe_BodyBones.pose.bones["mixamorig:RightShoulder.001"].constraints["Copy Location.001"].influence = 0.5
        add_constraint("mixamorig:LeftShoulder.001", "STRETCH_TO", "11 left shoulder")
        PosePipe_BodyBones.pose.bones["mixamorig:RightShoulder.001"].constraints['Stretch To'].rest_length = 0.1
        PosePipe_BodyBones.pose.bones["mixamorig:RightShoulder.001"].constraints['Stretch To'].volume = 'NO_VOLUME'
        PosePipe_BodyBones.pose.bones["mixamorig:RightShoulder.001"].constraints['Stretch To'].keep_axis = 'PLANE_Z'

        add_constraint("mixamorig:RightArm.001", "COPY_LOCATION", "12 right shoulder")
        add_constraint("mixamorig:RightArm.001", "STRETCH_TO", "14 right elbow")
        PosePipe_BodyBones.pose.bones["mixamorig:RightArm.001"].constraints['Stretch To'].volume = 'NO_VOLUME'
        PosePipe_BodyBones.pose.bones["mixamorig:RightArm.001"].constraints['Stretch To'].rest_length = 0.1

        add_constraint("mixamorig:RightForeArm.001", "COPY_LOCATION", "14 right elbow")
        if settings.body_tracking and settings.hand_tracking:
            add_constraint("mixamorig:RightForeArm.001", "STRETCH_TO", "00Hand Right")
        else:
            add_constraint("mixamorig:RightForeArm.001", "STRETCH_TO", "16 right wrist")
        PosePipe_BodyBones.pose.bones["mixamorig:RightForeArm.001"].constraints['Stretch To'].volume = 'NO_VOLUME'
        PosePipe_BodyBones.pose.bones["mixamorig:RightForeArm.001"].constraints['Stretch To'].rest_length = 0.1

        add_constraint("mixamorig:LeftUpLeg", "COPY_LOCATION", "23 left hip")
        add_constraint("mixamorig:LeftUpLeg", "STRETCH_TO", "25 left knee")
        PosePipe_BodyBones.pose.bones["mixamorig:LeftUpLeg"].constraints['Stretch To'].volume = 'NO_VOLUME'
        PosePipe_BodyBones.pose.bones["mixamorig:LeftUpLeg"].constraints['Stretch To'].rest_length = 0.1

        add_constraint("mixamorig:LeftLeg", "COPY_LOCATION", "25 left knee")
        add_constraint("mixamorig:LeftLeg", "STRETCH_TO", "27 left ankle")
        PosePipe_BodyBones.pose.bones["mixamorig:LeftLeg"].constraints['Stretch To'].volume = 'NO_VOLUME'
        PosePipe_BodyBones.pose.bones["mixamorig:LeftLeg"].constraints['Stretch To'].rest_length = 0.1

        add_constraint("mixamorig:LeftFoot", "COPY_LOCATION", "27 left ankle")
        add_constraint("mixamorig:LeftFoot", "STRETCH_TO", "31 left foot index")
        PosePipe_BodyBones.pose.bones["mixamorig:LeftFoot"].constraints['Stretch To'].volume = 'NO_VOLUME'
        PosePipe_BodyBones.pose.bones["mixamorig:LeftFoot"].constraints['Stretch To'].rest_length = 0.1

        add_constraint("mixamorig:RightUpLeg", "COPY_LOCATION", "24 right hip")
        add_constraint("mixamorig:RightUpLeg", "STRETCH_TO", "26 right knee")
        PosePipe_BodyBones.pose.bones["mixamorig:RightUpLeg"].constraints['Stretch To'].volume = 'NO_VOLUME'
        PosePipe_BodyBones.pose.bones["mixamorig:RightUpLeg"].constraints['Stretch To'].rest_length = 0.1

        add_constraint("mixamorig:RightLeg", "COPY_LOCATION", "26 right knee")
        add_constraint("mixamorig:RightLeg", "STRETCH_TO", "28 right ankle")
        PosePipe_BodyBones.pose.bones["mixamorig:RightLeg"].constraints['Stretch To'].volume = 'NO_VOLUME'
        PosePipe_BodyBones.pose.bones["mixamorig:RightLeg"].constraints['Stretch To'].rest_length = 0.1

        add_constraint("mixamorig:RightFoot", "COPY_LOCATION", "28 right ankle")
        add_constraint("mixamorig:RightFoot", "STRETCH_TO", "32 right foot index")
        PosePipe_BodyBones.pose.bones["mixamorig:RightFoot"].constraints['Stretch To'].volume = 'NO_VOLUME'
        PosePipe_BodyBones.pose.bones["mixamorig:RightFoot"].constraints['Stretch To'].rest_length = 0.1

        add_constraint("mixamorig:Neck.001", "COPY_LOCATION", "11 left shoulder")
        add_constraint("mixamorig:Neck.001", "COPY_LOCATION", "12 right shoulder")
        PosePipe_BodyBones.pose.bones["mixamorig:Neck.001"].constraints["Copy Location.001"].influence = 0.5

        add_constraint("mixamorig:Head.001", "COPY_LOCATION", "09 mouth (left)")
        PosePipe_BodyBones.pose.bones["mixamorig:Head.001"].constraints['Copy Location'].use_y = False
        add_constraint("mixamorig:Head.001", "COPY_LOCATION", "10 mouth (right)")
        PosePipe_BodyBones.pose.bones["mixamorig:Head.001"].constraints["Copy Location.001"].influence = 0.5
        PosePipe_BodyBones.pose.bones["mixamorig:Head.001"].constraints["Copy Location.001"].use_y = False
        add_constraint("mixamorig:Head.001", "COPY_LOCATION", "08 right ear")
        PosePipe_BodyBones.pose.bones["mixamorig:Head.001"].constraints["Copy Location.002"].use_x = False
        PosePipe_BodyBones.pose.bones["mixamorig:Head.001"].constraints["Copy Location.002"].use_z = False

        if settings.hand_tracking:

            hand_bones_and_constraints = {
                "RightHand.001": ["00Hand Right", "09Hand Right"],
                "RightHandThumb1.001": ["01Hand Right", "02Hand Right"],
                "RightHandThumb2.001": ["02Hand Right", "03Hand Right"],
                "RightHandThumb3.001": ["03Hand Right", "04Hand Right"],
                "RightHandIndex1.001": ["05Hand Right", "06Hand Right"],
                "RightHandIndex2.001": ["06Hand Right", "07Hand Right"],
                "RightHandIndex3.001": ["07Hand Right", "08Hand Right"],
                "RightHandMiddle1.001": ["09Hand Right", "10Hand Right"],
                "RightHandMiddle2.001": ["10Hand Right", "11Hand Right"],
                "RightHandMiddle3.001": ["11Hand Right", "12Hand Right"],
                "RightHandRing1.001": ["13Hand Right", "14Hand Right"],
                "RightHandRing2.001": ["14Hand Right", "15Hand Right"],
                "RightHandRing3.001": ["15Hand Right", "16Hand Right"],
                "RightHandPinky1.001": ["17Hand Right", "18Hand Right"],
                "RightHandPinky2.001": ["18Hand Right", "19Hand Right"],
                "RightHandPinky3.001": ["19Hand Right", "20Hand Right"],
                "LeftHand.001": ["00Hand Left", "09Hand Left"],
                "LeftHandThumb1.001": ["01Hand Left", "02Hand Left"],
                "LeftHandThumb2.001": ["02Hand Left", "03Hand Left"],
                "LeftHandThumb3.001": ["03Hand Left", "04Hand Left"],
                "LeftHandIndex1.001": ["05Hand Left", "06Hand Left"],
                "LeftHandIndex2.001": ["06Hand Left", "07Hand Left"],
                "LeftHandIndex3.001": ["07Hand Left", "08Hand Left"],
                "LeftHandMiddle1.001": ["09Hand Left", "10Hand Left"],
                "LeftHandMiddle2.001": ["10Hand Left", "11Hand Left"],
                "LeftHandMiddle3.001": ["11Hand Left", "12Hand Left"],
                "LeftHandRing1.001": ["13Hand Left", "14Hand Left"],
                "LeftHandRing2.001": ["14Hand Left", "15Hand Left"],
                "LeftHandRing3.001": ["15Hand Left", "16Hand Left"],
                "LeftHandPinky1.001": ["17Hand Left", "18Hand Left"],
                "LeftHandPinky2.001": ["18Hand Left", "19Hand Left"],
                "LeftHandPinky3.001": ["19Hand Left", "20Hand Left"]
            }

            for bone_name, cstr_objs in hand_bones_and_constraints.items():

                add_constraint(f"mixamorig:{bone_name}", "COPY_LOCATION", cstr_objs[0])
                add_constraint(f"mixamorig:{bone_name}", "STRETCH_TO", cstr_objs[1])
                PosePipe_BodyBones.pose.bones[f"mixamorig:{bone_name}"].constraints['Stretch To'].volume = 'NO_VOLUME'
                PosePipe_BodyBones.pose.bones[f"mixamorig:{bone_name}"].constraints['Stretch To'].rest_length = 0.1

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