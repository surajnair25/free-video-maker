/* eslint-disable @remotion/deterministic-randomness */
import fs from "fs-extra";
import path from "path";
import cuid from "cuid";
import ffmpeg from "fluent-ffmpeg";
import { getOrientationConfig } from "../../components/utils";
import { logger } from "../../logger";
import { OrientationEnum, type Video } from "../../types/shorts";

const defaultTimeoutMs = 30000;
const retryTimes = 3;

// A handful of gentle pan/zoom moves so consecutive scenes don't all look identical.
type KenBurnsMove = {
  zoomStart: number;
  zoomEnd: number;
  xStart: number;
  xEnd: number;
  yStart: number;
  yEnd: number;
};

const kenBurnsMoves: KenBurnsMove[] = [
  { zoomStart: 1.0, zoomEnd: 1.15, xStart: 0.5, xEnd: 0.5, yStart: 0.5, yEnd: 0.45 },
  { zoomStart: 1.15, zoomEnd: 1.0, xStart: 0.45, xEnd: 0.55, yStart: 0.5, yEnd: 0.5 },
  { zoomStart: 1.0, zoomEnd: 1.12, xStart: 0.4, xEnd: 0.5, yStart: 0.5, yEnd: 0.5 },
  { zoomStart: 1.12, zoomEnd: 1.0, xStart: 0.5, xEnd: 0.5, yStart: 0.4, yEnd: 0.5 },
];

export class PollinationsAPI {
  constructor(
    private stylePrompt: string,
    private apiKey?: string,
    private tempDirPath: string = "/tmp",
  ) {}

  private buildImageUrl(prompt: string, width: number, height: number): string {
    const fullPrompt = this.stylePrompt
      ? `${prompt}, ${this.stylePrompt}`
      : prompt;
    // image.pollinations.ai is Pollinations' longstanding no-signup, no-API-key
    // image endpoint. An optional key can be supplied for higher rate limits.
    const seed = Math.floor(Math.random() * 1_000_000);
    const params = new URLSearchParams({
      width: String(width),
      height: String(height),
      nologo: "true",
      seed: String(seed),
    });
    if (this.apiKey) {
      params.set("key", this.apiKey);
    }
    return `https://image.pollinations.ai/prompt/${encodeURIComponent(fullPrompt)}?${params.toString()}`;
  }

  private async downloadImage(url: string, destPath: string, timeout: number): Promise<void> {
    const response = await fetch(url, {
      method: "GET",
      redirect: "follow",
      signal: AbortSignal.timeout(timeout),
    });
    if (!response.ok) {
      throw new Error(`Pollinations API error: ${response.status} ${response.statusText}`);
    }
    const arrayBuffer = await response.arrayBuffer();
    await fs.writeFile(destPath, Buffer.from(arrayBuffer));
  }

  private async imageToKenBurnsVideo(
    imagePath: string,
    videoPath: string,
    durationSeconds: number,
    width: number,
    height: number,
  ): Promise<void> {
    const move = kenBurnsMoves[Math.floor(Math.random() * kenBurnsMoves.length)];
    const fps = 25;
    const totalFrames = Math.ceil(durationSeconds * fps);
    // Oversample so the zoompan filter has room to pan/zoom without revealing edges.
    const scaleW = width * 1.5;
    const scaleH = height * 1.5;

    const zoomExpr = `'${move.zoomStart}+(${move.zoomEnd}-${move.zoomStart})*on/${totalFrames}'`;
    const xExpr = `'(iw-iw/zoom)*${move.xStart}+((iw-iw/zoom)*${move.xEnd}-(iw-iw/zoom)*${move.xStart})*on/${totalFrames}'`;
    const yExpr = `'(ih-ih/zoom)*${move.yStart}+((ih-ih/zoom)*${move.yEnd}-(ih-ih/zoom)*${move.yStart})*on/${totalFrames}'`;

    await new Promise<void>((resolve, reject) => {
      ffmpeg()
        .input(imagePath)
        .inputOptions(["-loop", "1"])
        .complexFilter([
          `scale=${scaleW}:${scaleH}:force_original_aspect_ratio=increase,crop=${scaleW}:${scaleH}`,
          `zoompan=z=${zoomExpr}:x=${xExpr}:y=${yExpr}:d=${totalFrames}:s=${width}x${height}:fps=${fps}`,
        ].join(","))
        .outputOptions([
          "-t",
          String(durationSeconds),
          "-pix_fmt",
          "yuv420p",
          "-movflags",
          "+faststart",
        ])
        .output(videoPath)
        .on("end", () => resolve())
        .on("error", (err: unknown) => reject(err))
        .run();
    });
  }

  private async _findVideo(
    searchTerm: string,
    minDurationSeconds: number,
    orientation: OrientationEnum,
    timeout: number,
  ): Promise<Video> {
    const { width, height } = getOrientationConfig(orientation);
    const id = cuid();
    const imagePath = path.join(this.tempDirPath, `${id}-source.jpg`);
    const videoPath = path.join(this.tempDirPath, `${id}-kenburns.mp4`);

    const imageUrl = this.buildImageUrl(searchTerm, width, height);
    logger.debug({ searchTerm, imageUrl }, "Generating image via Pollinations");

    await this.downloadImage(imageUrl, imagePath, timeout);
    await this.imageToKenBurnsVideo(
      imagePath,
      videoPath,
      minDurationSeconds + 1,
      width,
      height,
    );
    await fs.remove(imagePath).catch(() => {});

    logger.debug({ searchTerm, videoPath }, "Created Ken Burns clip from Pollinations image");

    return {
      id,
      // Prefixed so ShortCreator knows to copy the file locally instead of
      // downloading it over http/https like it does for Pexels results.
      url: `file://${videoPath}`,
      width,
      height,
    };
  }

  async findVideo(
    searchTerms: string[],
    minDurationSeconds: number,
    excludeIds: string[] = [],
    orientation: OrientationEnum = OrientationEnum.portrait,
    timeout: number = defaultTimeoutMs,
    retryCounter: number = 0,
  ): Promise<Video> {
    const searchTerm = searchTerms[0] || "abstract background";
    try {
      return await this._findVideo(searchTerm, minDurationSeconds, orientation, timeout);
    } catch (error: unknown) {
      if (retryCounter < retryTimes) {
        logger.warn(
          { searchTerm, retryCounter },
          "Pollinations generation failed, retrying...",
        );
        return await this.findVideo(
          searchTerms,
          minDurationSeconds,
          excludeIds,
          orientation,
          timeout,
          retryCounter + 1,
        );
      }
      logger.error(error, "Error generating image/video via Pollinations");
      throw error;
    }
  }
}
