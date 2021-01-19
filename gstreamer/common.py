# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common utilities."""
import collections
import io
import time

SVG_HEADER = '<svg width="{w}" height="{h}" version="1.1" >'
SVG_RECT = '<rect x="{x}" y="{y}" width="{w}" height="{h}" stroke="{s}" stroke-width="{sw}" fill="none" />'
SVG_TEXT = '''
<text x="{x}" y="{y}" font-size="{fs}" dx="0.05em" dy="0.05em" fill="black">{t}</text>
<text x="{x}" y="{y}" font-size="{fs}" fill="white">{t}</text>
'''
SVG_FOOTER = '</svg>'

def avg_fps_counter(window_size):
    window = collections.deque(maxlen=window_size)
    prev = time.monotonic()
    yield 0.0  # First fps value.

    while True:
        curr = time.monotonic()
        window.append(curr - prev)
        prev = curr
        yield len(window) / sum(window)

class SVG:
    def __init__(self, size):
        self.io = io.StringIO()
        self.io.write(SVG_HEADER.format(w=size[0] , h=size[1]))

    def add_rect(self, x, y, w, h, stroke, stroke_width):
        self.io.write(SVG_RECT.format(x=x, y=y, w=w, h=h, s=stroke, sw=stroke_width))

    def add_text(self, x, y, text, font_size):
        self.io.write(SVG_TEXT.format(x=x, y=y, t=text, fs=font_size))

    def finish(self):
        self.io.write(SVG_FOOTER)
        return self.io.getvalue()
