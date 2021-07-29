import argparse
import math
import os
import pickle
import re
import sys
import time
from glob import glob

# sys.path.append("..")
# sys.path.append("../../")

BLENDER_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.dirname(BLENDER_DIR)
ROOT_DIR = os.path.dirname(DATASETS_DIR)
# SRC_DIR = os.path.join(ROOT_DIR, "src")
# CONFIG_DIR = os.path.join(SRC_DIR, "config")

sys.path.append(BLENDER_DIR)
sys.path.append(DATASETS_DIR)
sys.path.append(ROOT_DIR)
# sys.path.append(SRC_DIR)
# sys.path.append(CONFIG_DIR)

import bpy
import numpy as np
from mathutils import Matrix

from src.config.config import cfg
from datasets.Blender.render_base_utils import (
    get_K_P_from_blender,
    get_3x4_P_matrix_from_blender,
)


def parse_argument():
    parser = argparse.ArgumentParser(
        description="Renders given obj file by rotation a camera around it."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(cfg.PVNET_LINEMOD_DIR, "ape", "ape.ply"),
        help="The cad model to be rendered",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=cfg.TEMP_DIR,
        help="The directory of the output 2d image.",
    )
    parser.add_argument(
        "--bg_imgs",
        type=str,
        default=cfg.SUN_2012_JPEGIMAGES_DIR,
        help="Names of background images stored in a .npy file.",
    )
    parser.add_argument(
        "--poses_path",
        type=str,
        default=cfg.TEMP_DIR,
        help="6d poses(azimuth, euler, theta, x, y, z) stored in a .npy file.",
    )
    parser.add_argument(
        "--use_cycles",
        type=str,
        default="False",
        help="Decide whether to use cycles render or not.",
    )
    parser.add_argument("--azi", type=float, default=0.0, help="Azimuth of camera.")
    parser.add_argument("--ele", type=float, default=0.0, help="Elevation of camera.")
    parser.add_argument(
        "--theta", type=float, default=0.0, help="In-plane rotation angle of camera."
    )
    parser.add_argument(
        "--height", type=float, default=0.0, help="Location z of plane."
    )

    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)
    # args = parser.parse_args()

    return args


def batch_render_with_linemod(args, camera):
    os.system("mkdir -p {}".format(args.output_dir))
    bpy.ops.import_mesh.ply(filepath=args.input)
    obj = bpy.data.objects[os.path.basename(args.input).replace(".ply", "")]
    bpy.context.scene.render.image_settings.file_format = "JPEG"

    if bpy.ops.preferences.addon_enable(module="cycles") == {"FINISHED"}:
        bpy.context.scene.render.engine = "CYCLES"
        bpy.context.preferences.addons[
            "cycles"
        ].preferences.compute_dexice_type = "CUDA"  # or "OPENCL"
        bpy.context.scene.cycles.device = "GPU"
        bpy.context.scene.cycles.sample_clamp_indirect = 1.0
        bpy.context.scene.cycles.blur_glossy = 3.0
        bpy.context.scene.cycles.samples = 100

    for mesh in bpy.data.meshes:
        mesh.use_auto_smooth = True

    _add_shader_on_world()

    mat = _add_shader_on_ply_object(obj)

    bg_imgs = np.load(args.bg_imgs).astype(str)
    bg_imgs = np.random.choice(bg_imgs, size=cfg.NUM_SYN)
    poses = np.load(args.poses_path)
    begin_num_imgs = len(glob(os.path.join(args.output_dir, "*.jpg")))

    for i in range(begin_num_imgs, cfg.NUM_SYN):
        img_name = os.path.basename(bg_imgs[i])
        bpy.data.images.load(bg_imgs[i])
        # overlay an background image and place the object
        bpy.data.worlds["World"].node_tree.nodes[
            "Environment Texture"
        ].image = bpy.data.images[img_name]

        pose = poses[i]
        # x, y = np.random.uniform(-0.15, 0.15, size=2)
        x, y = 0, 0
        obj.location = [x, y, 0]

        obj_name = os.path.basename(args.input).endswith(".ply")
        _set_material_node_parameters(mat, obj_name)

        _render(camera, "{}/{}".format(args.output_dir, i), pose)

        object_to_world_pose = np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, 0]])
        object_to_world_pose = np.append(object_to_world_pose, [[0, 0, 0, 1]], axis=0)
        KRT = get_K_P_from_blender(camera)
        world_to_camera_pose = np.append(KRT["RT"], [[0, 0, 0, 1]], axis=0)
        world_to_camera_pose = np.dot(world_to_camera_pose, object_to_world_pose)[:3]
        with open("{}/{}_RT.pkl".format(args.output_dir, i), "wb") as f:
            pickle.dump({"RT": world_to_camera_pose, "K": KRT["K"]}, f)
        bpy.data.images.remove(bpy.data.images[img_name])


def setup():
    cam_name = "Camera"

    cam_obj = _setup_camera(bpy.context.collection, cam_name)
    bpy.data.cameras[cam_name].clip_end = 10000

    cam_constraint = cam_obj.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    b_empty = _parent_obj_to_camera(cam_obj)
    cam_constraint.target = b_empty

    # configure rendered image's parametersS
    # bpy.context.scene.render.resolution_x = cfg.WIDTH
    # bpy.context.scene.render.resolution_y = cfg.HEIGHT
    bpy.context.scene.render.resolution_x = 640
    bpy.context.scene.render.resolution_y = 480
    bpy.context.scene.render.resolution_percentage = 100
    # bpy.context.scene.render.alpha_mode = 'TRANSPARENT'
    file_format = bpy.context.scene.render.image_settings.file_format
    if file_format is not "PNG":
        bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.image_settings.color_mode = "RGBA"
    # modify the camera intrinsic matrix
    # bpy.data.cameras['Camera'].sensor_width = 39.132693723430386
    # bpy.context.scene.render.pixel_aspect_y = 1.6272340492401836

    # composite node
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    for n in tree.nodes:
        tree.nodes.remove(n)
    rl = tree.nodes.new(type="CompositorNodeRLayers")
    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.base_path = ""
    depth_file_output.format.file_format = "OPEN_EXR"
    depth_file_output.format.color_depth = "32"

    map_node = tree.nodes.new(type="CompositorNodeMapRange")
    # map_node.inputs[1].default_value = cfg.MIN_DEPTH
    # map_node.inputs[2].default_value = cfg.MAX_DEPTH
    map_node.inputs[1].default_value = 0
    map_node.inputs[2].default_value = 2
    map_node.inputs[3].default_value = 0
    map_node.inputs[4].default_value = 1
    links.new(rl.outputs["Depth"], map_node.inputs[0])
    links.new(map_node.outputs[0], depth_file_output.inputs[0])
    # camera = bpy.data.cameras[cam_name]

    return cam_obj, depth_file_output


def _add_shader_on_ply_object(obj):
    # 変更したいオブジェクトをアクティブ化
    bpy.context.view_layer.objects.active = obj

    # 新規マテリアルを作成する
    mat = bpy.data.materials.new("Material")
    # ノードを使用する
    mat.use_nodes = True
    # mat.node_tree.links.clear()

    nodes = mat.node_tree.nodes
    mat_out = nodes["Material Output"]
    # diffuse = nodes["Diffuse BSDF"]

    # 属性ノードを追加する．
    gloss_node = nodes.new(type="ShaderNodeBsdfGlossy")
    attr_node = nodes.new(type="ShaderNodeAttribute")
    # 属性ノードのアトリビュート名を Col (頂点カラー)に変更する
    attr_node.attribute_name = "Col"

    # ノードリンクを取得する
    mat_link = mat.node_tree.links
    # ノードリンクを初期化する
    mat_link.clear()
    mat_link.new(attr_node.outputs["Color"], gloss_node.inputs["Color"])
    mat_link.new(gloss_node.outputs["BSDF"], mat_out.inputs["Surface"])

    # マテリアルスロットを追加する
    bpy.ops.object.material_slot_add()
    # 作成したマテリアルスロットに新規マテリアルを設定する
    bpy.context.object.active_material = mat

    return mat


def _add_shader_on_world(wld_dat=bpy.data.worlds["World"]):
    """
    World に Background ノードを追加する関数．
    既に Background ノードが存在する場合にはそれを削除して新しいものに置き換える．

    Args:
        wld_dat(bpy_struct): World データ
    """
    wld_dat.use_nodes = True
    node_tree = wld_dat.node_tree
    nodes = node_tree.nodes

    # "Background" を名前にもつ Node のリスト
    name_list = ["Background", "World Output", "Environment Texture"]
    for name in name_list:
        _much_node_name_delete(nodes, name)
    # name_list = [node.name for node in nodes if re.match("Background*", node.name)]

    # 名前が存在する場合
    # if name_list:
    #    # すべて削除する
    #    for node in nodes:
    #        nodes.remove(node)

    # Background Node を追加
    bg_node = nodes.new(type="ShaderNodeBackground")
    # Environment Texture Node を追加
    env_node = nodes.new(type="ShaderNodeTexEnvironment")
    output_node = nodes.new(type="ShaderNodeOutputWorld")

    # Env_Node -> Bg_Node へリンク接続
    node_tree.links.new(env_node.outputs["Color"], bg_node.inputs["Color"])
    # Bg_Node -> World_Output_Node へリンク接続
    node_tree.links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])


def _camPosToQuaternion(cx, cy, cz):
    q1a = 0
    q1b = 0
    q1c = math.sqrt(2) / 2
    q1d = math.sqrt(2) / 2
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist
    t = math.sqrt(cx * cx + cy * cy)
    tx = cx / t
    ty = cy / t
    yaw = math.acos(ty)
    if tx > 0:
        yaw = 2 * math.pi - yaw
    pitch = 0
    tmp = min(max(tx * cx + ty * cy, -1), 1)
    # roll = math.acos(tx * cx + ty * cy)
    roll = math.acos(tmp)
    if cz < 0:
        roll = -roll
    print("%f %f %f" % (yaw, pitch, roll))
    q2a, q2b, q2c, q2d = _quaternionFromYawPitchRoll(yaw, pitch, roll)
    q1 = q1a * q2a - q1b * q2b - q1c * q2c - q1d * q2d
    q2 = q1b * q2a + q1a * q2b + q1d * q2c - q1c * q2d
    q3 = q1c * q2a - q1d * q2b + q1a * q2c + q1b * q2d
    q4 = q1d * q2a + q1c * q2b - q1b * q2c + q1a * q2d
    return q1, q2, q3, q4


def _camRotQuaternion(cx, cy, cz, theta):
    theta = theta / 180.0 * math.pi
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = -cx / camDist
    cy = -cy / camDist
    cz = -cz / camDist
    q1 = math.cos(theta * 0.5)
    q2 = -cx * math.sin(theta * 0.5)
    q3 = -cy * math.sin(theta * 0.5)
    q4 = -cz * math.sin(theta * 0.5)
    return q1, q2, q3, q4


def _much_node_name_delete(nodes, name: str):
    """
    Name と 同じ文字列を名前に持つ Node をすべて削除する関数．

    Args:
        nodes: node_tree.nodes
        name(str): サーチする文字列
    """
    name_list = [node.name for node in nodes if re.match(name + "*", node.name)]

    # 名前が存在する場合
    if name_list:
        # すべて削除する
        for node in nodes:
            nodes.remove(node)


def _obj_centened_camera_pos(dist, azimuth_deg, elevation_deg):
    phi = float(elevation_deg) / 180 * math.pi
    theta = float(azimuth_deg) / 180 * math.pi
    x = dist * math.cos(theta) * math.cos(phi)
    y = dist * math.sin(theta) * math.cos(phi)
    z = dist * math.sin(phi)
    return x, y, z


def _obj_location(dist, azi, ele):
    ele = math.radians(ele)
    azi = math.radians(azi)
    x = dist * math.cos(azi) * math.cos(ele)
    y = dist * math.sin(azi) * math.cos(ele)
    z = dist * math.sin(ele)
    return x, y, z


def Plane():
    """
    カーソルの位置を頂点の1つとする平面を作成する関数
    """
    # 平面を形成する頂点と面を定義する
    verts = [(0, 0, 0), (0, 5, 0), (5, 5, 0), (5, 0, 0)]
    faces = [(0, 1, 2, 3)]

    # メッシュを定義する
    mesh = bpy.data.meshes.new("Plane_mesh")
    # 頂点と面のデータからメッシュを生成する
    mesh.from_pydata(verts, [], faces)
    mesh.update(calc_edges=True)

    # メッシュのデータからオブジェクトを定義する
    obj = bpy.data.objects.new("Plane", mesh)
    # オブジェクトの生成場所をカーソルに指定する
    obj.location = bpy.context.scene.cursor.location
    # オブジェクトをシーンにリンク(表示)させる
    bpy.context.scene.collection.objects.link(obj)


def _parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.get("Empty")

    # Empty オブジェクトが存在しない場合
    if b_empty is None:
        b_empty = bpy.data.objects.new("Empty", None)
        b_empty.location = origin
        scn = bpy.context.scene
        scn.collection.objects.link(b_empty)

    b_camera.parent = b_empty  # setup parenting

    scv = bpy.context.view_layer
    scv.objects.active = b_empty
    return b_empty


def _quaternionFromYawPitchRoll(yaw, pitch, roll):
    c1 = math.cos(yaw / 2.0)
    c2 = math.cos(pitch / 2.0)
    c3 = math.cos(roll / 2.0)
    s1 = math.sin(yaw / 2.0)
    s2 = math.sin(pitch / 2.0)
    s3 = math.sin(roll / 2.0)
    q1 = c1 * c2 * c3 + s1 * s2 * s3
    q2 = c1 * c2 * s3 - s1 * s2 * c3
    q3 = c1 * s2 * c3 + s1 * c2 * s3
    q4 = s1 * c2 * c3 - c1 * s2 * s3
    return q1, q2, q3, q4


def _quaternionProduct(qx, qy):
    a = qx[0]
    b = qx[1]
    c = qx[2]
    d = qx[3]
    e = qy[0]
    f = qy[1]
    g = qy[2]
    h = qy[3]
    q1 = a * e - b * f - c * g - d * h
    q2 = a * f + b * e + c * h - d * g
    q3 = a * g - b * h + c * e + d * f
    q4 = a * h + b * g - c * f + d * e
    return q1, q2, q3, q4


def _render(camera, outfile, pose):
    bpy.context.scene.render.filepath = outfile
    depth_file_output.file_slots[0].path = (
        bpy.context.scene.render.filepath + "_depth.png"
    )

    azimuth, elevation, theta = pose[:3]
    cam_dist = 0.5
    cx, cy, cz = _obj_centened_camera_pos(cam_dist, azimuth, elevation)

    q1 = _camPosToQuaternion(cx, cy, cz)
    q2 = _camRotQuaternion(cx, cy, cz, theta)
    q = _quaternionProduct(q2, q1)
    camera.location[0] = cx  # + np.random.uniform(-cfg.pose_noise,g_camPos_noise)
    camera.location[1] = cy  # + np.random.uniform(-g_camPos_noise,g_camPos_noise)
    camera.location[2] = cz  # + np.random.uniform(-g_camPos_noise,g_camPos_noise)
    camera.rotation_mode = "QUATERNION"

    camera.rotation_quaternion[0] = q[0]
    camera.rotation_quaternion[1] = q[1]
    camera.rotation_quaternion[2] = q[2]
    camera.rotation_quaternion[3] = q[3]
    # camera.location = [0, 1, 0]
    # camera.rotation_euler = [np.pi / 2, 0, np.pi]

    _setup_light(bpy.context.collection)

    rotation_matrix = get_K_P_from_blender(camera)["RT"][:, :3]
    camera.location = -np.dot(rotation_matrix.T, pose[3:])
    # カメラオブジェクトをアクティブにする．
    bpy.context.scene.camera = camera
    bpy.ops.render.render(write_still=True)


def _set_material_node_parameters(material, obj_name):
    nodes = material.node_tree.nodes
    if obj_name:
        nodes["Glossy BSDF"].inputs["Roughness"].default_value = np.random.uniform(
            0.8, 1
        )
    else:
        nodes["Diffuse BSDF"].inputs["Roughness"].default_value = np.random.uniform(
            0, 1
        )


def _setup_camera(collection, cam_name="Camera"):
    # 全オブジェクトの名前のリスト
    name_list = [cam.name for cam in bpy.data.cameras if re.match("Camera*", cam.name)]

    # カメラが存在する場合
    if name_list:
        # オブジェクトを削除する
        for cam in bpy.data.cameras:
            bpy.data.cameras.remove(cam)

    # cam_name = "Camera"
    # カメラデータを作成する
    cam_data = bpy.data.cameras.new(name=cam_name)
    # カメラオブジェクトを作成する
    cam_obj = bpy.data.objects.new(name=cam_name, object_data=cam_data)
    # 現在のコンテキストのコレクションのオブジェクトリストにリンク（登録）する（と表示される）
    collection.objects.link(cam_obj)

    # cam_obj = bpy.data.cameras[cam_name]

    return cam_obj


def _setup_light(collection):
    # 全オブジェクトの名前のリスト
    name_list = [lit.name for lit in bpy.data.lights if re.match("Light*", lit.name)]

    # ライトが存在する場合
    if name_list:
        # オブジェクトを削除する
        for lit in bpy.data.lights:
            bpy.data.lights.remove(lit)

    for i in range(2):
        azi = np.random.uniform(0, 360)
        ele = np.random.uniform(0, 40)
        dist = np.random.uniform(1, 2)
        x, y, z = _obj_location(dist, azi, ele)
        lit_name = "Light_{}".format(i)
        # ライトデータを作成する
        lit_data = bpy.data.lights.new(name=lit_name, type="POINT")
        # スポットライトのパワーを編集
        energy = np.random.uniform(low=0.1, high=1)
        lit_data.energy = energy * 1000.0
        # ライトオブジェクトを作成する
        lit_obj = bpy.data.objects.new(name=lit_name, object_data=lit_data)
        lit_obj.location = (x, y, z)
        # 現在のコンテキストのコレクションのオブジェクトリストにリンク（登録）する（と表示される）
        collection.objects.link(lit_obj)


if __name__ == "__main__":
    begin = time.time()

    # DeBug 用(一度作成したオブジェクトをすべて削除する)
    # obj_name = os.path.basename(input_ply).replace(".ply", "")
    # 全オブジェクトの名前のリスト
    name_list = [obj.name for obj in bpy.data.objects]

    # 同名のオブジェクトが存在する場合
    if name_list:
        # オブジェクトを削除する
        for obj in bpy.data.objects:
            bpy.data.objects.remove(obj)

    # マテリアルデータを削除
    for mat in bpy.data.materials:
        bpy.data.materials.remove(mat)

    args = parse_argument()
    camera, depth_file_output = setup()
    if os.path.basename(args.input).endswith(".ply"):
        batch_render_with_linemod(args, camera)
    else:
        batch_render_ycb(args, camera)
    print("cost {} s".format(time.time() - begin))
