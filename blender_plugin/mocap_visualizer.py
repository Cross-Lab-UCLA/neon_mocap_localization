# ruff: noqa
bl_info = {
    "name": "MoCap + Neon Render",
    "author": "Rob Ennis",
    "version": (1, 0, 0),
    "blender": (5, 0, 1),
    "category": "Object",
    "warning": "Requires installation of dependencies",
}

import bpy
import numpy as np
import pandas as pd
from bpy_extras import anim_utils
from mathutils import Vector


def apply_curve_material(rayobj):
    # create curve material if not already created.
    # otherwise, assign and return
    if bpy.data.materials.get("curve_material") is None:
        crv_mat = bpy.data.materials.new("curve_material")

        bpy.data.materials["curve_material"].use_nodes = True
        bpy.data.materials["curve_material"].node_tree.nodes["Principled BSDF"].inputs[
            0
        ].default_value = (0.93, 0.31, 0.24, 1)
        bpy.data.materials["curve_material"].node_tree.nodes["Principled BSDF"].inputs[
            26
        ].default_value = (0.93, 0.31, 0.24, 1)
        bpy.data.materials["curve_material"].node_tree.nodes["Principled BSDF"].inputs[
            2
        ].default_value = 1
    else:
        crv_mat = bpy.data.materials["curve_material"]

    if rayobj.data.materials:
        rayobj.data.materials[0] = crv_mat
    else:
        rayobj.data.materials.append(crv_mat)


def create_initial_ray(name="DynamicRay"):
    curve_data = bpy.data.curves.new(name, type="CURVE")
    curve_data.dimensions = "3D"

    curve_data.bevel_depth = 0.0015 * 10  # m
    curve_data.bevel_resolution = 2

    # Create a new spline
    spline = curve_data.splines.new("POLY")
    spline.points.add(count=1)

    # Initialize (x,y,z,w)
    spline.points[0].co = (0.0, 0.0, 0.0, 1.0)
    spline.points[1].co = (1.0, 0.0, 0.0, 1.0)

    obj = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(obj)

    apply_curve_material(obj)

    return obj


def select_single_obj(single_obj):
    for obj in bpy.data.objects:
        obj.select_set(False)

    single_obj.select_set(True)
    bpy.context.view_layer.objects.active = single_obj


def disable_objs(objs):
    for obj in objs:
        obj.hide_render = True
        obj.hide_viewport = True
        obj.hide_select = True


def enable_objs(objs):
    for obj in objs:
        obj.hide_render = False
        obj.hide_viewport = False
        obj.hide_select = False


def register():
    bpy.types.Scene.marker_positions_csv = bpy.props.StringProperty(
        name="marker_positions_csv",
        description="File with the Neon + MoCap data to animate (e.g., marker_positions_w_gaze.csv)",
        default="",
    )
    bpy.utils.register_class(OperatorAnimateNeon)
    bpy.utils.register_class(PANEL_PT_AnimateNeon)
    bpy.types.VIEW3D_MT_object.append(menu_func)

    wm = bpy.context.window_manager


def unregister():
    del bpy.types.Scene.marker_positions_csv
    bpy.utils.unregister_class(OperatorAnimateNeon)
    bpy.utils.unregister_class(PANEL_PT_AnimateNeon)
    bpy.types.VIEW3D_MT_object.remove(menu_func)


def menu_func(self, context):
    self.layout.operator(ObjectAnimateNeon.bl_idname)


class PANEL_PT_AnimateNeon(bpy.types.Panel):
    """MoCap + Neon - Animate"""

    bl_label = "Animate MoCap + Neon data"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "MoCap + Neon"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.label(text="Marker Positions CSV:")

        row = layout.row()
        row.prop(scene, "marker_positions_csv")

        layout.label(text="Frames to animate:")

        row = layout.row()
        row.prop(scene, "frame_start")
        row.prop(scene, "frame_end")

        layout.label(text="Rendered frame rate:")

        row = layout.row()
        row.prop(scene.render, "fps")

        layout.label(text="Animate:")

        row = layout.row()
        row.operator(OperatorAnimateNeon.bl_idname)


class OperatorAnimateNeon(bpy.types.Operator):
    """MoCap + Neon - Animate"""

    bl_idname = "object.neon_animate"
    bl_label = "Apply MoCap data to Neon"
    bl_options = {"REGISTER", "UNDO"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.report({"INFO"}, "MoCap + Neon: Initializing...")

        self.mocap_df = pd.read_csv(bpy.context.scene.marker_positions_csv)

        self.mocap_start_s = self.mocap_df["timestamp [ns]"][0] * 1e-9
        self.last_frame = len(self.mocap_df)

        self.data_frame_rate = np.mean(
            1.0 / np.diff(self.mocap_df["timestamp [ns]"] * 1e-9)
        )

        self.report({"INFO"}, "MoCap + Neon: Loaded data.")

        self.pl_objs = []
        self.ray_obj = []
        self.obj_curves = {}

        self.fps = bpy.context.scene.render.fps
        self.spf = 1.0 / self.fps
        self.frame_step = int(self.data_frame_rate / self.fps)

        self.report(
            {"INFO"},
            "MoCap + Neon: Temporal context calculated and initialized. Preparing keyframe loop...",
        )

    def __del__(self):
        self.report({"INFO"}, "End")
        super().__del__()

    def build_pl_obj_list(self):
        column_names = list(self.mocap_df.keys())

        if "timestamp [ns]" in column_names:
            column_names.remove("timestamp [ns]")
        if "Unnamed: 0" in column_names:
            column_names.remove("Unnamed: 0")
        if "Unnamed: 0.1" in column_names:
            column_names.remove("Unnamed: 0.1")

        n_markers = int(len(column_names) / 3)
        marker_names = column_names[::3]

        self.report({"INFO"}, f"{marker_names}")

        for n, marker_name in enumerate(marker_names):
            marker_name = marker_name[:-2]

            if marker_name == "gaze_dir":
                # Create the ray at the world center initially
                self.ray_obj.append(create_initial_ray(name=marker_name))
            else:
                marker_pos = self.mocap_df[
                    [marker_name + "_X", marker_name + "_Y", marker_name + "_Z"]
                ]

                initial_pos = marker_pos.iloc[0].to_numpy() / 1000

                if np.isnan(initial_pos[0]):
                    bpy.ops.mesh.primitive_uv_sphere_add(
                        radius=0.02, location=(-1000, -1000, -1000)
                    )
                else:
                    bpy.ops.mesh.primitive_uv_sphere_add(
                        radius=0.02,
                        location=(initial_pos[0], initial_pos[1], initial_pos[2]),
                    )

                sphere = bpy.context.active_object
                sphere.name = marker_name

                self.pl_objs.append(sphere)

    def init_fcurves(self):
        for action in bpy.data.actions:
            bpy.data.actions.remove(action)

        for obj in self.pl_objs:
            if obj.animation_data is not None:
                obj.animation_data_clear()

            # Create the action, with a slot for the object, a layer, and a keyframe strip:
            action = bpy.data.actions.new(name=obj.name + "MyAction")
            slot = action.slots.new(obj.id_type, obj.name)
            strip = action.layers.new("MyLayer").strips.new(type="KEYFRAME")

            # Create a channelbag to hold the F-Curves for the slot:
            channelbag = strip.channelbag(slot, ensure=True)

            loc_fcurves = [
                channelbag.fcurves.new(data_path="location", index=index)
                for index in range(3)
            ]

            # Assign the action and the slot to the object:
            adt = obj.animation_data_create()
            adt.action = action
            adt.action_slot = slot

            self.obj_curves[obj.name] = {
                "location": loc_fcurves,
            }

    def insert_fcurve(self, obj):
        obj_fcurves = self.obj_curves[obj.name]

        nframes = int(len(self.mocap_df) / self.frame_step)

        frames = list(range(nframes))

        # data for 'co' [frame, val, frame, val...]
        flat_data = [0] * (nframes * 2)

        for fcurve in obj_fcurves["location"]:
            fcurve.keyframe_points.add(nframes)
            for kf in fcurve.keyframe_points:
                kf.interpolation = "CONSTANT"

        axis_idx_lookup = {"X": 0, "Y": 1, "Z": 2}
        for fcurve, axis in zip(obj_fcurves["location"], ["X", "Y", "Z"]):
            if obj.name == "gaze_origin":
                pos = self.mocap_df[obj.name + "_" + axis].to_numpy().squeeze() / 1000
            else:
                pos = self.mocap_df[obj.name + "_" + axis].to_numpy().squeeze() / 1000

            pos[np.isnan(pos)] = -1000

            values = pos[:: self.frame_step]
            values = values[:nframes]

            flat_data[0::2] = frames
            flat_data[1::2] = values
            fcurve.keyframe_points.foreach_set("co", flat_data)

            fcurve.update()

    # need to find out why this is not being applied in main python script
    # Neon position relative to markers:
    # avg_neon_rel_to_markers = np.array([0.01539802, 0.09746081, 0.04796103])
    def init_insert_raycurve(self):
        curve = self.ray_obj[0].data

        if curve.animation_data is not None:
            curve.animation_data_clear()

        curve.animation_data_create()
        ad = curve.animation_data

        if ad.action is None:
            ad.action = bpy.data.actions.new(name=self.ray_obj[0].name + "Animation")

        if ad.action_slot is None:
            curve.splines[0].points[0].co = (0.0, 0.0, 0.0, 1.0)
            curve.splines[0].points[0].keyframe_insert(data_path="co", index=0, frame=1)

        channelbag = anim_utils.action_get_channelbag_for_slot(
            ad.action, ad.action_slot
        )

        nframes = int(len(self.mocap_df) / self.frame_step)

        frames = list(range(nframes))

        ray_origins = (
            self.mocap_df[["gaze_origin_X", "gaze_origin_Y", "gaze_origin_Z"]]
            .to_numpy()
            .squeeze()
            / 1000
        )
        ray_directions = (
            self.mocap_df[["gaze_dir_X", "gaze_dir_Y", "gaze_dir_Z"]]
            .to_numpy()
            .squeeze()
        )

        ray_lengths = np.zeros((len(ray_directions),))
        unit_directions = np.zeros_like(ray_directions)
        for idx in range(len(ray_origins)):
            ro = ray_origins[idx]

            rd = ray_directions[idx]
            rd /= np.linalg.norm(rd)

            unit_directions[idx] = rd

            closest_isect = bpy.context.scene.ray_cast(self.dg, ro, rd)
            if not closest_isect[0]:
                closest_isect = None

            t = 0.0
            norm_quat = []
            if closest_isect is not None:
                h = closest_isect[1]
                t = np.linalg.norm(h - Vector((ro[0], ro[1], ro[2])))
            else:
                t = 1000

            ray_lengths[idx] = t

        # We need to keyframe Point 0 (Origin) and Point 1 (End)
        # Each point has
        # 3 channels (X, Y, Z)
        axes = ["X", "Y", "Z"]
        gaze_dir_axes = ["X", "Y", "Z"]
        for pt_idx in [0, 1]:
            for char_idx in range(3):  # 0:X, 1:Y, 2:Z
                ray_fcurve = channelbag.fcurves.find(
                    data_path=f"splines[0].points[{pt_idx}].co", index=char_idx
                )
                if ray_fcurve is None:
                    ray_fcurve = channelbag.fcurves.new(
                        data_path=f"splines[0].points[{pt_idx}].co", index=char_idx
                    )

                ray_fcurve.keyframe_points.clear()

                ray_fcurve.keyframe_points.add(nframes)
                for kf in ray_fcurve.keyframe_points:
                    kf.interpolation = "LINEAR"

                flat_data = np.zeros(nframes * 2)

                if pt_idx == 0:
                    pos = (
                        self.mocap_df["gaze_origin_" + axes[char_idx]]
                        .to_numpy()
                        .squeeze()
                        / 1000
                    )
                else:
                    origin = (
                        self.mocap_df["gaze_origin_" + axes[char_idx]]
                        .to_numpy()
                        .squeeze()
                        / 1000
                    )
                    pos = (
                        self.mocap_df["gaze_dir_" + gaze_dir_axes[char_idx]]
                        .to_numpy()
                        .squeeze()
                        / 1000
                    )

                    pos = origin + pos * 1000

                pos[np.isnan(pos)] = -1000

                values = pos[:: self.frame_step]
                values = values[:nframes]

                flat_data[0::2] = frames
                flat_data[1::2] = values
                ray_fcurve.keyframe_points.foreach_set("co", flat_data)

                ray_fcurve.update()

    def execute(self, context):
        self.build_pl_obj_list()

        context.view_layer.update()
        self.dg = context.view_layer.depsgraph
        self.dg.update()

        disable_objs(self.pl_objs)

        self.init_fcurves()

        for obj in self.pl_objs:
            self.insert_fcurve(obj)

        self.init_insert_raycurve()

        enable_objs(self.pl_objs)
        context.scene.frame_set(0)

        self.report(
            {"INFO"},
            f"Mocap + Neon: Prepared {context.scene.frame_end} keyframes.",
        )

        return {"FINISHED"}


if __name__ == "__main__":
    register()
