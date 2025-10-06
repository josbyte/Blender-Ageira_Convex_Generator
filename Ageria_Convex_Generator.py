bl_info = {
    "name": "Precision Convex Decomposition (CoACD)",
    "description": "High precision convex decomposition using CoACD with optimized geometry (threaded, overlay bottom-left, presets)",
    "author": "Josbyte",
    "version": (1, 0, 1),
    "blender": (4, 5, 2),
    "location": "View3D > Sidebar > CoACD",
    "category": "Object",
}

import sys
import os
import threading
import queue
import time
import inspect

flatpak_site = os.path.expanduser("~/.var/app/org.blender.Blender/data/python/lib/python3.11/site-packages")
is_flatpak = os.path.exists("/.flatpak-info") or bool(os.environ.get("FLATPAK_SANDBOX_DIR")) or bool(os.environ.get("FLATPAK_ID"))

if is_flatpak and os.path.isdir(flatpak_site):
    sys.path.append(flatpak_site)
    # opcional: debug
    # print("Flatpak detected — appended:", flatpak_site)
else:
    # No hacemos nada si no es flatpak (evita añadir rutas inexistentes)
    pass

import bpy
import trimesh
import coacd
import numpy as np  # Para optimización
from mathutils import Matrix
import blf

# ----------------- Compatibilidad blf.size -----------------
def blf_set_size(font_id, size, dpi=None):
    """Wrapper blf.size"""
    try:
        params = inspect.signature(blf.size).parameters
        if len(params) == 2:
            blf.size(font_id, size)
        else:
            if dpi is None:
                try:
                    dpi = bpy.context.preferences.system.dpi
                except Exception:
                    dpi = 72
            blf.size(font_id, size, dpi)
    except Exception:
        try:
            blf.size(font_id, size)
        except Exception:
            pass

# ----------------- Helpers -----------------
def triangulate_faces(faces):
    tri_faces = []
    for face in faces:
        if len(face) == 3:
            tri_faces.append(face)
        elif len(face) > 3:
            for i in range(1, len(face) - 1):
                tri_faces.append([face[0], face[i], face[i + 1]])
    return tri_faces

def validate_mesh(verts, faces):
    valid_faces = []
    verts_np = np.array(verts, dtype=float)
    for face in faces:
        if len(set(face)) != len(face):
            continue
        try:
            v0 = verts_np[face[0]]
            v1 = verts_np[face[1]]
            v2 = verts_np[face[2]]
            edge1 = v1 - v0
            edge2 = v2 - v0
            cross = np.cross(edge1, edge2)
            area = np.linalg.norm(cross) / 2.0
            if area > 1e-6:
                valid_faces.append(face)
        except Exception:
            continue
    return valid_faces

# ----------------- Worker: corre en thread separado -----------------
def coacd_worker(data_list, params, out_q, stop_event):
    for item in data_list:
        if stop_event.is_set():
            break

        name = item['name']
        verts = np.array(item['verts'], dtype=float)
        faces = np.array(item['faces'], dtype=int) if item['faces'] else np.array([], dtype=int)

        if faces.size == 0:
            out_q.put({'name': name, 'error': "No valid faces after validation"})
            continue

        try:
            tm = trimesh.Trimesh(verts, faces, process=True)
        except Exception as e:
            out_q.put({'name': name, 'error': f"Trimesh creation failed: {e}"})
            continue

        try:
            coacd_mesh = coacd.Mesh(tm.vertices, tm.faces)
            parts = coacd.run_coacd(
                coacd_mesh,
                threshold=params.get('threshold', 0.05),
                max_convex_hull=params.get('max_hulls', 20),
                preprocess_mode=params.get('preprocess_mode', 'auto'),
                preprocess_resolution=params.get('preprocess_resolution', 50),
                mcts_iterations=params.get('mcts_iterations', 100),
                decimate=True,
                max_ch_vertex=128,
                merge=params.get('merge', True)
            )
            simple_parts = []
            for p in parts:
                pv = np.array(p[0]).tolist()
                pf = np.array(p[1]).tolist()
                simple_parts.append((pv, pf))
            out_q.put({'name': name, 'parts': simple_parts})
        except Exception as e:
            out_q.put({'name': name, 'error': f"CoACD failed: {e}"})
            continue

    out_q.put({'_done': True})

# ----------------- Operador principal (modal) -----------------
class OBJECT_OT_convex_decomposition(bpy.types.Operator):
    """Perform Convex Decomposition on Selected Objects (threaded, non-blocking)"""
    bl_idname = "object.convex_decomposition"
    bl_label = "Convex Decomposition"
    bl_options = {'REGISTER', 'UNDO'}

    _timer = None
    _draw_handler = None
    _worker_thread = None
    _queue = None
    _stop_event = None

    def draw_overlay(self, context):
        # Esquina inferior izquierda: x=20, y=20
        x = 20
        y = 20

        # Texto: tamaño y contenido
        blf_set_size(0, 14)
        total = getattr(self, "total", 0)
        done = getattr(self, "progress_count", 0)
        # Forzamos float para evitar división entera impredecible
        pct = int((float(done) / float(total)) * 100) if total > 0 else 0
        # Clamp 0..100
        if pct < 0:
            pct = 0
        if pct > 100:
            pct = 100

        # Spinner si el worker sigue vivo
        spinner = ""
        try:
            alive = self._worker_thread.is_alive() if self._worker_thread else False
            if alive:
                t = int(time.time() * 3) % 4
                spinner = [" |", " /", " -", " \\"][t]
        except Exception:
            spinner = ""

        lines = [
            f"CoACD: {done}/{total} objetos ({pct}%) {spinner}",
            f"Object: {getattr(self, 'current_object_name', '')}",
            f"Phase: {getattr(self, 'current_stage', '')}",
            "Press ESC to cancel"
        ]

        offset = 0
        for line in lines:
            blf.position(0, x, y + offset, 0)
            blf.draw(0, line)
            offset += 18

    def invoke(self, context, event):
        # Forzar modo OBJECT
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        selected = [obj for obj in context.selected_objects if obj.type == 'MESH']
        self.total = len(selected)
        if self.total == 0:
            self.report({'WARNING'}, "No mesh objects selected.")
            return {'CANCELLED'}

        # Preparar datos puros para worker (no tocar bpy en worker)
        data_list = []
        for obj in selected:
            mesh = obj.data
            world_matrix = obj.matrix_world
            verts = [(world_matrix @ v.co).to_tuple() for v in mesh.vertices]
            faces = [list(p.vertices) for p in mesh.polygons]
            tri_faces = triangulate_faces(faces)
            tri_faces = validate_mesh(verts, tri_faces)
            if not tri_faces:
                data_list.append({'name': obj.name, 'verts': verts, 'faces': []})
            else:
                data_list.append({'name': obj.name, 'verts': verts, 'faces': tri_faces})

        # Parámetros (mantener defaults optimizados)
        scene = context.scene
        params = {
            'threshold': max(scene.coacd_threshold, 0.01),
            'max_hulls': scene.coacd_max_hulls,
            'preprocess_mode': scene.coacd_preprocess_mode if scene.coacd_preprocess_mode != 'off' else 'auto',
            'preprocess_resolution': max(scene.coacd_preprocess_resolution, 10),
            'mcts_iterations': max(scene.coacd_mcts_iterations, 10),
            'merge': scene.coacd_merge
        }

        # Cola y control de stop
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self.progress_count = 0
        self.current_object_name = ""
        self.current_stage = ""

        # Iniciar worker thread (daemon para que Blender pueda cerrarse limpio)
        self._worker_thread = threading.Thread(
            target=coacd_worker,
            args=(data_list, params, self._queue, self._stop_event),
            daemon=True
        )
        self._worker_thread.start()

        # Iniciar barra de progreso y timer/modal
        context.window_manager.progress_begin(0, self.total)
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.2, window=context.window)
        wm.modal_handler_add(self)

        # Registrar draw handler para overlay (POST_PIXEL)
        self._draw_handler = bpy.types.SpaceView3D.draw_handler_add(self.draw_overlay, (context,), 'WINDOW', 'POST_PIXEL')

        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'ESC':
            # Solicitar cancelación
            self._stop_event.set()
            self.current_stage = "Stopping... (ignoring incoming results)"
            return {'PASS_THROUGH'}

        if event.type == 'TIMER':
            # Procesar la cola (crear mallas en hilo principal)
            while self._queue is not None and not self._queue.empty():
                item = self._queue.get()
                if '_done' in item:
                    # worker terminó
                    continue

                name = item.get('name', '<unknown>')
                self.current_object_name = name

                if 'error' in item:
                    # Registrar fallo y avanzar
                    self.current_stage = f"Error: {item['error']}"
                    self.progress_count += 1
                    context.window_manager.progress_update(self.progress_count)
                    continue

                parts = item.get('parts', [])
                # Si cancelado, descartamos partes
                if self._stop_event.is_set():
                    self.current_stage = f"Discarding {name} (canceled)"
                    self.progress_count += 1
                    context.window_manager.progress_update(self.progress_count)
                    continue

                # Crear colección y mallas (hilo principal)
                collection_name = f"{name}_hulls"
                if collection_name in bpy.data.collections:
                    collection = bpy.data.collections[collection_name]
                else:
                    collection = bpy.data.collections.new(collection_name)
                    context.scene.collection.children.link(collection)

                hull_count = 0
                for i, p in enumerate(parts):
                    pv, pf = p
                    new_mesh = bpy.data.meshes.new(f"{name}_hull_{i}")
                    new_mesh.from_pydata(pv, [], pf)
                    new_mesh.update()
                    new_obj = bpy.data.objects.new(f"{name}_hull_{i}", new_mesh)
                    collection.objects.link(new_obj)
                    new_obj.matrix_world = Matrix()
                    hull_count += 1
                    # breve pausa cooperativa
                    time.sleep(0)

                # Aplicar decimate si está activado en la escena (seguro en hilo principal)
                if getattr(context.scene, "coacd_enable_decimate", False):
                    for o in list(collection.objects):
                        mod = o.modifiers.new(name="Decimate", type='DECIMATE')
                        mod.ratio = context.scene.coacd_decimate_ratio
                        mod.use_collapse_triangulate = True
                        with context.temp_override(object=o):
                            try:
                                bpy.ops.object.modifier_apply(modifier=mod.name)
                            except Exception:
                                pass

                self.current_stage = f"Finished {name}: {hull_count} hulls"
                self.progress_count += 1
                # Actualizar la barra de progreso de Blender (valor entero)
                context.window_manager.progress_update(self.progress_count)

            # Comprobar si worker terminó y cola vacía
            worker_alive = self._worker_thread.is_alive() if self._worker_thread else False
            queue_empty = self._queue.empty() if self._queue else True
            if (not worker_alive) and queue_empty:
                # Cleanup
                self.finish(context)
                if self._stop_event.is_set():
                    self.report({'WARNING'}, "Convex decomposition cancelled.")
                    return {'CANCELLED'}
                else:
                    self.report({'INFO'}, f"Processed {self.progress_count} objects.")
                    return {'FINISHED'}

            # Forzar redraw para overlay (necesario para mostrar spinner y %)
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()

        return {'PASS_THROUGH'}

    def finish(self, context):
        # Cleanup seguro
        wm = context.window_manager
        if self._timer is not None:
            try:
                wm.event_timer_remove(self._timer)
            except Exception:
                pass
            self._timer = None
        if self._draw_handler is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(self._draw_handler, 'WINDOW')
            except Exception:
                pass
            self._draw_handler = None
        try:
            context.window_manager.progress_end()
        except Exception:
            pass
        # Vaciar cola si queda algo
        try:
            while self._queue and not self._queue.empty():
                self._queue.get_nowait()
        except Exception:
            pass

# ----------------- Preset Operators -----------------
class OBJECT_OT_coacd_preset_low(bpy.types.Operator):
    """Preset: Low (fast, low-detail)"""
    bl_idname = "object.coacd_preset_low"
    bl_label = "Low"

    def execute(self, context):
        s = context.scene
        s.coacd_threshold = 0.2
        s.coacd_max_hulls = 8
        s.coacd_preprocess_resolution = 20
        s.coacd_mcts_iterations = 10
        s.coacd_enable_decimate = True
        s.coacd_decimate_ratio = 0.6
        s.coacd_merge = True
        self.report({'INFO'}, "Preset Low applied")
        return {'FINISHED'}

class OBJECT_OT_coacd_preset_mid(bpy.types.Operator):
    """Preset: Mid (balanced)"""
    bl_idname = "object.coacd_preset_mid"
    bl_label = "Mid"

    def execute(self, context):
        s = context.scene
        s.coacd_threshold = 0.1
        s.coacd_max_hulls = 20
        s.coacd_preprocess_resolution = 50
        s.coacd_mcts_iterations = 100
        s.coacd_enable_decimate = False
        s.coacd_decimate_ratio = 0.5
        s.coacd_merge = True
        self.report({'INFO'}, "Preset Mid applied")
        return {'FINISHED'}

class OBJECT_OT_coacd_preset_high(bpy.types.Operator):
    """Preset: High (detailed)"""
    bl_idname = "object.coacd_preset_high"
    bl_label = "High"

    def execute(self, context):
        s = context.scene
        s.coacd_threshold = 0.03
        s.coacd_max_hulls = 50
        s.coacd_preprocess_resolution = 120
        s.coacd_mcts_iterations = 500
        s.coacd_enable_decimate = False
        s.coacd_decimate_ratio = 0.5
        s.coacd_merge = True
        self.report({'INFO'}, "Preset High applied")
        return {'FINISHED'}

class OBJECT_OT_coacd_preset_veryhigh(bpy.types.Operator):
    """Preset: Very High (maximum detail)"""
    bl_idname = "object.coacd_preset_veryhigh"
    bl_label = "Very High"

    def execute(self, context):
        s = context.scene
        s.coacd_threshold = 0.01
        s.coacd_max_hulls = 200
        s.coacd_preprocess_resolution = 300
        s.coacd_mcts_iterations = 2000
        s.coacd_enable_decimate = False
        s.coacd_decimate_ratio = 0.5
        s.coacd_merge = False
        self.report({'INFO'}, "Preset Very High applied")
        return {'FINISHED'}

# ----------------- Panel (ahora en su propia pestaña 'CoACD') -----------------
class VIEW3D_PT_convex_decomposition(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "CoACD"      # <- Aquí: pestaña propia en la barra lateral (N-panel)
    bl_label = "Convex Decomposition"

    def draw(self, context):
        layout = self.layout

        # Preset buttons (fila)
        row = layout.row(align=True)
        row.operator("object.coacd_preset_low", text="Low")
        row.operator("object.coacd_preset_mid", text="Mid")
        row.operator("object.coacd_preset_high", text="High")
        row.operator("object.coacd_preset_veryhigh", text="Very High")

        layout.separator()
        layout.operator("object.convex_decomposition")
        layout.prop(context.scene, "coacd_threshold")
        layout.prop(context.scene, "coacd_max_hulls")
        layout.prop(context.scene, "coacd_preprocess_mode")
        layout.prop(context.scene, "coacd_preprocess_resolution")
        layout.prop(context.scene, "coacd_mcts_iterations")
        layout.prop(context.scene, "coacd_enable_decimate")
        layout.prop(context.scene, "coacd_decimate_ratio")
        layout.prop(context.scene, "coacd_merge")

# ----------------- Registro -----------------
classes = (
    OBJECT_OT_convex_decomposition,
    OBJECT_OT_coacd_preset_low,
    OBJECT_OT_coacd_preset_mid,
    OBJECT_OT_coacd_preset_high,
    OBJECT_OT_coacd_preset_veryhigh,
    VIEW3D_PT_convex_decomposition,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.coacd_threshold = bpy.props.FloatProperty(
        name="Concavity Threshold",
        description="Lower = more detailed decomposition (more pieces)",
        default=0.05,
        min=0.01,
        max=1.0
    )
    bpy.types.Scene.coacd_max_hulls = bpy.props.IntProperty(
        name="Max Hulls",
        description="Maximum number of convex hulls per object",
        default=20,
        min=1,
        max=1024
    )
    bpy.types.Scene.coacd_preprocess_mode = bpy.props.EnumProperty(
        name="Preprocess Mode",
        description="Preprocess mode",
        items=[('auto', 'Auto', ''), ('on', 'On', ''), ('off', 'Off', '')],
        default='auto'
    )
    bpy.types.Scene.coacd_preprocess_resolution = bpy.props.IntProperty(
        name="Preprocess Resolution",
        description="Resolution for preprocessing (higher = more detailed but slower)",
        default=50,
        min=10,
        max=500
    )
    bpy.types.Scene.coacd_mcts_iterations = bpy.props.IntProperty(
        name="MCTS Iterations",
        description="Number of MCTS iterations for better decomposition (higher = slower but better quality)",
        default=100,
        min=10,
        max=10000
    )
    bpy.types.Scene.coacd_enable_decimate = bpy.props.BoolProperty(
        name="Enable Decimate",
        description="Apply decimation modifier to hulls (disabling speeds up)",
        default=False
    )
    bpy.types.Scene.coacd_decimate_ratio = bpy.props.FloatProperty(
        name="Decimate Ratio",
        description="Ratio of faces to keep (0.1 = 10% of original faces, 1.0 = no reduction)",
        default=0.5,
        min=0.01,
        max=1.0
    )
    bpy.types.Scene.coacd_merge = bpy.props.BoolProperty(
        name="Merge Hulls",
        description="Enable merging of convex hulls (disable for more pieces)",
        default=True
    )

def unregister():
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass

    for p in ("coacd_threshold","coacd_max_hulls","coacd_preprocess_mode","coacd_preprocess_resolution",
              "coacd_mcts_iterations","coacd_enable_decimate","coacd_decimate_ratio","coacd_merge"):
        if hasattr(bpy.types.Scene, p):
            try:
                delattr(bpy.types.Scene, p)
            except Exception:
                pass

if __name__ == "__main__":
    try:
        unregister()
    except Exception:
        pass
    register()
