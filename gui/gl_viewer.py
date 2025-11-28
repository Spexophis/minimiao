import ctypes

import numpy as np
from OpenGL import GL
from OpenGL.GL import shaders
from PyQt6.QtCore import QTimer, QSize, pyqtSignal
from PyQt6.QtWidgets import QSizePolicy
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

VERT_SRC = """
#version 330 core
layout (location = 0) in vec2 a_pos;   // -1..1
layout (location = 1) in vec2 a_uv;    //  0..1
uniform vec2 u_scale;                  // aspect-fit scaling
out vec2 v_uv;
void main() {
    v_uv = a_uv;
    vec2 p = a_pos * u_scale;
    gl_Position = vec4(p, 0.0, 1.0);
}
"""

FRAG_SRC = """
#version 330 core
in vec2 v_uv;
out vec4 out_color;

uniform sampler2D u_tex;     // GL_R16 (normalized 0..1 when sampled)
uniform float u_black;       // 0..1
uniform float u_white;       // 0..1
uniform float u_gamma;       // >0

void main() {
    float v = texture(u_tex, v_uv).r;                  // 0..1 from R16
    v = (v - u_black) / max(u_white - u_black, 1e-6);  // normalize by levels
    v = clamp(v, 0.0, 1.0);
    v = pow(v, 1.0 / max(u_gamma, 1e-6));              // gamma
    out_color = vec4(v, v, v, 1.0);
}
"""


class GLGray16Viewer(QOpenGLWidget):
    """
    High-throughput uint16 grayscale viewer.
    Call .set_frame(frame_u16) from the GUI thread (or via Qt signal with QueuedConnection).
    """
    frameConsumed = pyqtSignal(object)   # token
    frameDiscarded = pyqtSignal(object)  # token when replaced before upload

    def __init__(self, use_pbo=True, parent=None):
        super().__init__(parent)
        self.use_pbo = use_pbo
        self.use_pbo = use_pbo
        self._pending = None
        self._pending_token = None
        self._have_new = False

        # CPU-side state (avoid per-frame allocations)
        self._pending = None  # last received frame (uint16 HxW)
        self._have_new = False
        self._img_w = 0
        self._img_h = 0

        # GPU resources
        self._prog = None
        self._vao = None
        self._vbo = None
        self._tex = None

        # PBO ping-pong (optional)
        self._pbo = None
        self._pbo_idx = 0
        self._pbo_nbytes = 0

        # Display controls (in uint16 space)
        self.black_u16 = 0
        self.white_u16 = 65535
        self.gamma = 1.0

        # Drive repaint; vsync usually caps to monitor refresh (~60)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(0)

        # ✅ let layouts stretch it
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # ✅ optional: prevent it from becoming tiny
        self.setMinimumSize(300, 300)

    def sizeHint(self) -> QSize:
        return QSize(700, 700)

    def minimumSizeHint(self) -> QSize:
        return QSize(300, 300)

    def set_levels(self, black_u16: int, white_u16: int, gamma: float = 1.0):
        self.black_u16 = int(np.clip(black_u16, 0, 65535))
        self.white_u16 = int(np.clip(white_u16, 0, 65535))
        self.gamma = float(max(gamma, 1e-6))

    def set_frame(self, frame_u16: np.ndarray, token=None):
        # If a pending frame hasn’t been uploaded yet, discard it (drop frame)
        if self._have_new and self._pending_token is not None:
            self.frameDiscarded.emit(self._pending_token)

        self._pending = np.ascontiguousarray(frame_u16)
        self._pending_token = token
        self._have_new = True
        self.update()  # schedule paintGL

    def initializeGL(self):
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)

        # Shader program
        self._prog = shaders.compileProgram(
            shaders.compileShader(VERT_SRC, GL.GL_VERTEX_SHADER),
            shaders.compileShader(FRAG_SRC, GL.GL_FRAGMENT_SHADER),
        )
        GL.glUseProgram(self._prog)

        # Fullscreen quad (triangle strip): positions + UVs
        # UVs are set so image is NOT vertically flipped (OpenGL tex origin is bottom-left).
        verts = np.array([
            #  x,    y,    u,   v
            -1.0, -1.0, 0.0, 0.0,
            1.0, -1.0, 1.0, 0.0,
            -1.0, 1.0, 0.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
        ], dtype=np.float32)

        self._vao = GL.glGenVertexArrays(1)
        self._vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self._vao)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, verts.nbytes, verts, GL.GL_STATIC_DRAW)

        stride = 4 * 4  # 4 floats per vertex * 4 bytes
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, False, stride, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, False, stride, ctypes.c_void_p(8))

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)

        # Texture (allocated when first frame arrives)
        self._tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

        # sampler binding
        loc = GL.glGetUniformLocation(self._prog, "u_tex")
        GL.glUniform1i(loc, 0)

        if self.use_pbo:
            self._pbo = GL.glGenBuffers(2)

    def _ensure_texture_and_pbo(self, w, h):
        if (w, h) == (self._img_w, self._img_h) and self._img_w > 0:
            return

        self._img_w, self._img_h = w, h

        # Allocate texture storage once (then update via TexSubImage)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0,
            GL.GL_R16, w, h, 0,
            GL.GL_RED, GL.GL_UNSIGNED_SHORT, None
        )

        if self.use_pbo:
            nbytes = w * h * 2
            if nbytes != self._pbo_nbytes:
                self._pbo_nbytes = nbytes
                for i in range(2):
                    GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, self._pbo[i])
                    GL.glBufferData(GL.GL_PIXEL_UNPACK_BUFFER, nbytes, None, GL.GL_STREAM_DRAW)
                GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)

    def _upload_frame(self, frame_u16: np.ndarray):
        h, w = frame_u16.shape
        self._ensure_texture_and_pbo(w, h)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex)

        if not self.use_pbo:
            GL.glTexSubImage2D(
                GL.GL_TEXTURE_2D, 0, 0, 0, w, h,
                GL.GL_RED, GL.GL_UNSIGNED_SHORT, frame_u16
            )
            return

        # Double PBO ping-pong
        pbo_id = self._pbo[self._pbo_idx]
        self._pbo_idx = 1 - self._pbo_idx

        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, pbo_id)
        # Copy CPU -> PBO (still a copy, but driver can pipeline better)
        GL.glBufferSubData(GL.GL_PIXEL_UNPACK_BUFFER, 0, frame_u16.nbytes, frame_u16)

        # Texture update from PBO (last arg is offset into PBO)
        GL.glTexSubImage2D(
            GL.GL_TEXTURE_2D, 0, 0, 0, w, h,
            GL.GL_RED, GL.GL_UNSIGNED_SHORT, ctypes.c_void_p(0)
        )
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)

    def _set_uniforms(self):
        # Levels in normalized [0..1]
        b = self.black_u16 / 65535.0
        w = self.white_u16 / 65535.0
        g = self.gamma

        GL.glUseProgram(self._prog)
        GL.glUniform1f(GL.glGetUniformLocation(self._prog, "u_black"), b)
        GL.glUniform1f(GL.glGetUniformLocation(self._prog, "u_white"), w)
        GL.glUniform1f(GL.glGetUniformLocation(self._prog, "u_gamma"), g)

        # Aspect-fit scaling for quad
        if self._img_w > 0 and self._img_h > 0 and self.height() > 0:
            img_aspect = self._img_w / self._img_h
            win_aspect = self.width() / self.height()
            if img_aspect >= win_aspect:
                sx, sy = 1.0, win_aspect / img_aspect
            else:
                sx, sy = img_aspect / win_aspect, 1.0
        else:
            sx, sy = 1.0, 1.0

        GL.glUniform2f(GL.glGetUniformLocation(self._prog, "u_scale"), sx, sy)

    def paintGL(self):
        GL.glViewport(0, 0, self.width(), self.height())
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        if self._have_new and self._pending is not None:
            self._upload_frame(self._pending)

            if self._pending_token is not None:
                self.frameConsumed.emit(self._pending_token)

            self._have_new = False
            self._pending = None
            self._pending_token = None

        if self._img_w == 0:
            return

        self._set_uniforms()

        GL.glBindVertexArray(self._vao)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        GL.glBindVertexArray(0)
