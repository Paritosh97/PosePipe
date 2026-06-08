import bpy
from bpy.props import StringProperty, BoolProperty, PointerProperty
from bpy.types import Panel, Operator, PropertyGroup
from bpy_extras.io_utils import ImportHelper
import requests
from bpy.types import Operator
import logging
import traceback


bl_info = {
    "name": "PosePipe Client",
    "version": (1, 0, 0),
    "blender": (2, 80, 0),
    "category": "3D View",
}

class OT_GenerateLandmarks(Operator):
    bl_idname = "posepipe.generate_landmarks"
    bl_label = "Generate Landmarks"
    
    def execute(self, context):
        s = context.scene.posepipe_settings
        
        try:
            # 1. Request data
            response = requests.post(f"{s.server_url}/process_video", json={
                "file_path": s.file_path,
                "use_pose": s.use_pose,
                "use_hand": s.use_hand,
                "use_face": s.use_face
            })
            response.raise_for_status()
            results = response.json().get("results", [])

            settings = bpy.context.scene.posepipe_settings

            if settings.use_pose:
                body = body_setup()
            if settings.use_hand:
                hand_left, hand_right = hands_setup()
            if settings.use_face: 
                face = face_setup()

            for frame_data in results:
                frame_idx = frame_data.get("frame")
                
                # Helper to process and keyframe any landmark type
                def process_type(data_list, type):
                    if not data_list: return
                    
                    try:
                        if type == 'body':
                            # data_list is [33 landmarks]
                            bones = sorted(body.children, key=lambda b: b.name)
                            for k in range(33):
                                b_data = data_list[k]
                                bones[k].location = ((0.5 - b_data['x']), (b_data['z'] / 4), (0.2 - b_data['y']) + 2)
                                bones[k].keyframe_insert(data_path="location", frame=frame_idx)

                        elif type == 'hand':
                            # data_list is [[21 landmarks], [21 landmarks]]
                            # Process Left Hand (index 0)
                            if len(data_list) > 0:
                                bones_l = sorted(hand_left.children, key=lambda b: b.name)
                                for k in range(21):
                                    b_data = data_list[0][k]
                                    bones_l[k].location = ((0.5 - b_data['x']), (b_data['z'] / 4), (0.2 - b_data['y']) + 2)
                                    bones_l[k].keyframe_insert(data_path="location", frame=frame_idx)
                            # Process Right Hand (index 1)
                            if len(data_list) > 1:
                                bones_r = sorted(hand_right.children, key=lambda b: b.name)
                                for k in range(21):
                                    b_data = data_list[1][k]
                                    bones_r[k].location = ((0.5 - b_data['x']), (b_data['z'] / 4), (0.2 - b_data['y']) + 2)
                                    bones_r[k].keyframe_insert(data_path="location", frame=frame_idx)

                        elif type == 'face':
                            # data_list is [468 landmarks]
                            bones = sorted(face.children, key=lambda b: b.name)
                            for k in range(468):
                                b_data = data_list[k]
                                bones[k].location = ((0.5 - b_data['x']), (b_data['z'] / 4), (0.2 - b_data['y']) + 2)
                                bones[k].keyframe_insert(data_path="location", frame=frame_idx)
                                
                    except Exception as e:
                        print(f"Error in process_type ({type}): {e}")
                        

                if 'pose' in frame_data: process_type(frame_data['pose'], 'body')
                if 'hand' in frame_data: process_type(frame_data['hand'], 'hand')
                if 'face' in frame_data: process_type(frame_data['face'], 'face')
            
            context.scene.frame_set(1)
            self.report({'INFO'}, f"Animation complete for {len(results)} frames.")
            
        except Exception as e:
            self.report({'ERROR'}, f"Sync failed: {str(e)}")
            return {'CANCELLED'}
            
        return {'FINISHED'}
    
# --- Settings ---
class PosePipeSettings(PropertyGroup):
    server_url: StringProperty(name="Server URL", default="http://localhost:8000")
    file_path: StringProperty(name="Video Path", subtype="FILE_PATH", default="/home/paritosh97/Desktop/SignMitra/data/vocab_videos/Above.webm")
    use_pose: BoolProperty(name="Pose", default=True)
    use_hand: BoolProperty(name="Hand", default=False)
    use_face: BoolProperty(name="Face", default=False)

# --- Operators ---
class OT_UploadVideo(Operator, ImportHelper):
    bl_idname = "posepipe.upload_video"
    bl_label = "Select Video File"
    
    filter_glob: bpy.props.StringProperty(
        default="*.mp4;*.avi;*.mov;*.mkv",
        options={'HIDDEN'}
    )

    def execute(self, context):
        context.scene.posepipe_settings.file_path = self.filepath
        self.report({'INFO'}, f"Video selected: {self.filepath}")
        return {'FINISHED'}


class OT_CreateSkeleton(Operator):
    bl_idname = "posepipe.create_skeleton"
    bl_label = "Create Mixamo Skeleton"
    
    def execute(self, context):
        # Standard Mixamo-compatible armature creation logic
        bpy.ops.object.armature_add()
        armature = bpy.context.object
        armature.name = "Mixamo_Rig"
        self.report({'INFO'}, "Mixamo Skeleton Created")
        return {'FINISHED'}

# --- UI Panel ---
class VIEW3D_PT_PosePipe(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'PosePipe'
    bl_label = "PosePipe Controls"

    def draw(self, context):
        layout = self.layout
        s = context.scene.posepipe_settings
        
        layout.prop(s, "server_url")
        layout.separator()
        
        box = layout.box()
        box.label(text="Video Selection:")
        box.prop(s, "file_path", text="")
        box.operator("posepipe.upload_video", text="Upload Video", icon='FILE_FOLDER')
        
        layout.separator()
        layout.label(text="Tracking Options:")
        layout.prop(s, "use_pose")
        layout.prop(s, "use_hand")
        layout.prop(s, "use_face")
        
        layout.separator()
        layout.operator("posepipe.generate_landmarks", icon='PLAY')
        layout.operator("pose.skeleton_builder", text="Generate Bones", icon='ARMATURE_DATA')


class OT_SkeletonBuilder(bpy.types.Operator):
    """Builds an armature to use with the mocap data"""
    bl_idname = "pose.skeleton_builder"
    bl_label = "Skeleton Builder"

    def execute(self, context):

        settings = bpy.context.scene.posepipe_settings

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

        if settings.use_hand:
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
        if settings.use_pose and settings.use_hand:
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
        if settings.use_pose and settings.use_hand:
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

        if settings.use_hand:

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

        return {'FINISHED'}

def do_assign(left, leftKey, centerKey, right, rightKey = None):
    success = True
    try:
        if (rightKey == None):
            left[leftKey].constraints[centerKey].target = right
        else:
            left[leftKey].constraints[centerKey].target = right[rightKey]
    except Exception as exception:
        success = False
        logging.error(traceback.format_exc())
    return success

def body_setup():
    """ Setup tracking boxes for body tracking """

    for area in bpy.context.screen.areas: 
        if area.type == 'VIEW_3D':
            for space in area.spaces: 
                if space.type == 'VIEW_3D':
                    space.shading.color_type = 'OBJECT'

    scene_objects = [n for n in bpy.context.scene.objects.keys()]
    setup = "Pose" in scene_objects

    if not setup:
        bpy.ops.object.add(radius=0.1, type='EMPTY')
        pose = bpy.context.active_object
        pose.name = "Pose"
        pose.scale = (-1,1,1)

    pose = bpy.context.scene.objects["Pose"]

    bpy.ops.object.add(radius=0.1, type='EMPTY')
    body = bpy.context.active_object
    body.name = "Body"
    body.parent = pose

    body_names = [
        "00 nose",
        "01 left eye (inner)",
        "02 left eye",
        "03 left eye (outer)",
        "04 right eye (inner)",
        "05 right eye",
        "06 right eye (outer)",
        "07 left ear",
        "08 right ear",
        "09 mouth (left)",
        "10 mouth (right)",
        "11 left shoulder",
        "12 right shoulder",
        "13 left elbow",
        "14 right elbow",
        "15 left wrist",
        "16 right wrist",
        "17 left pinky",
        "18 right pinky",
        "19 left index",
        "20 right index",
        "21 left thumb",
        "22 right thumb",
        "23 left hip",
        "24 right hip",
        "25 left knee",
        "26 right knee",
        "27 left ankle",
        "28 right ankle",
        "29 left heel",
        "30 right heel",
        "31 left foot index",
        "32 right foot index",
    ]

    for k in range(33):
        bpy.ops.mesh.primitive_cube_add()
        box = bpy.context.active_object
        box.name = body_names[k]
        box.scale = [0.003, 0.003, 0.003]
        box.parent = body
        box.color = (0,255,0,255)

    body = bpy.context.scene.objects["Body"]
    return body

def hands_setup():
    """ Setup tracking boxes for hand tracking """

    scene_objects = [n for n in bpy.context.scene.objects.keys()]
    setup = "Pose" in scene_objects

    if not setup:
        bpy.ops.object.add(radius=0.1, type='EMPTY')
        pose = bpy.context.active_object
        pose.name = "Pose"
        pose.scale = (-1,1,1)

    pose = bpy.context.scene.objects["Pose"]

    for area in bpy.context.screen.areas: 
        if area.type == 'VIEW_3D':
            for space in area.spaces: 
                if space.type == 'VIEW_3D':
                    space.shading.color_type = 'OBJECT'

    if "Hand Left" not in scene_objects:
        bpy.ops.object.add(radius=0.1, type='EMPTY')
        hand_left = bpy.context.active_object
        hand_left.name = "Hand Left"
        hand_left.parent = pose

        for k in range(21):
            bpy.ops.mesh.primitive_cube_add()
            box = bpy.context.active_object
            box.name = str(k).zfill(2) + "Hand Left"
            box.scale = (0.005, 0.005, 0.005)
            box.parent = hand_left
            box.color = (0,0,255,255)

    if "Hand Right" not in scene_objects:
        bpy.ops.object.add(radius=0.1, type='EMPTY')
        hand_right = bpy.context.active_object
        hand_right.name = "Hand Right"
        hand_right.parent = pose

        for k in range(21):
            bpy.ops.mesh.primitive_cube_add()
            box = bpy.context.active_object
            box.name = str(k).zfill(2) + "Hand Right"
            box.scale = (0.005, 0.005, 0.005)
            box.parent = hand_right
            box.color = (255,0,0,255)    

    hand_left = bpy.context.scene.objects["Hand Left"]
    hand_right = bpy.context.scene.objects["Hand Right"]
    pose.scale = (-1,1,1)
    return hand_left, hand_right

def face_setup():
    """ Setup tracking boxes for face tracking """

    scene_objects = [n for n in bpy.context.scene.objects.keys()]
    setup = "Pose" in scene_objects

    if not setup:
        bpy.ops.object.add(radius=0.1, type='EMPTY')
        pose = bpy.context.active_object
        pose.name = "Pose"
        pose.scale = (-1,1,1)

    pose = bpy.context.scene.objects["Pose"]

    for area in bpy.context.screen.areas: 
        if area.type == 'VIEW_3D':
            for space in area.spaces: 
                if space.type == 'VIEW_3D':
                    space.shading.color_type = 'OBJECT'

    if "Face" not in scene_objects:
        bpy.ops.object.add(radius=0.1, type='EMPTY')
        face = bpy.context.active_object
        face.name = "Face"
        face.parent = pose

        for k in range(468):
            bpy.ops.mesh.primitive_cube_add()
            box = bpy.context.active_object
            box.name = str(k).zfill(3) + "Face"
            box.scale = (0.002, 0.002, 0.002)
            box.parent = face
            box.color = (255,0,255,255)

    face = bpy.context.scene.objects["Face"]
    pose.scale = (-1,1,1)
    return face

# --- Registration ---
classes = [PosePipeSettings, OT_UploadVideo, OT_GenerateLandmarks, OT_SkeletonBuilder, VIEW3D_PT_PosePipe]

def register():
    for cls in classes: bpy.utils.register_class(cls)
    bpy.types.Scene.posepipe_settings = PointerProperty(type=PosePipeSettings)

def unregister():
    for cls in reversed(classes): bpy.utils.unregister_class(cls)
    del bpy.types.Scene.posepipe_settings

if __name__ == "__main__":
    register()