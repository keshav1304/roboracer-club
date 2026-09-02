/** Decode base64 Int8 occupancy (ROS OccupancyGrid values). */
export function decodeOccB64(b64: string): Int8Array {
  const bin = atob(b64);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Int8Array(u8.buffer, u8.byteOffset, u8.byteLength);
}

/**
 * RViz-style continuous colormap into RGBA ImageData.
 * -1 unknown → cool gray; 0 free → white; 100 occupied → black.
 */
export function occupancyToImageData(
  occ: Int8Array,
  width: number,
  height: number
): ImageData {
  const img = new ImageData(width, height);
  const d = img.data;
  const n = Math.min(occ.length, width * height);
  for (let i = 0; i < n; i++) {
    const o = occ[i];
    const j = i * 4;
    if (o < 0) {
      d[j] = 168;
      d[j + 1] = 174;
      d[j + 2] = 186;
      d[j + 3] = 255;
    } else {
      const t = Math.min(100, Math.max(0, o)) / 100;
      const g = Math.round(255 * (1 - t));
      d[j] = g;
      d[j + 1] = g;
      d[j + 2] = g;
      d[j + 3] = 255;
    }
  }
  return img;
}

/** Paint occupancy onto an offscreen canvas (nearest-neighbor source). */
export function occupancyToCanvas(
  occ: Int8Array,
  width: number,
  height: number
): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = width;
  c.height = height;
  const ctx = c.getContext("2d");
  if (ctx) {
    ctx.putImageData(occupancyToImageData(occ, width, height), 0, 0);
  }
  return c;
}
