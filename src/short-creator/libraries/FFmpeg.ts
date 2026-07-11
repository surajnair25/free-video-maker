import ffmpeg from "fluent-ffmpeg";
import fs from "fs-extra";
import { Readable } from "node:stream";
import { logger } from "../../logger";

export class FFMpeg {
  static async init(): Promise<FFMpeg> {
    return import("@ffmpeg-installer/ffmpeg").then((ffmpegInstaller) => {
      ffmpeg.setFfmpegPath(ffmpegInstaller.path);
      logger.info("FFmpeg path set to:", ffmpegInstaller.path);
      return new FFMpeg();
    });
  }

  async saveNormalizedAudio(
    audio: ArrayBuffer,
    outputPath: string,
  ): Promise<string> {
    logger.debug("Normalizing audio for Whisper");
    const inputStream = new Readable();
    inputStream.push(Buffer.from(audio));
    inputStream.push(null);

    return new Promise((resolve, reject) => {
      ffmpeg()
        .input(inputStream)
        .audioCodec("pcm_s16le")
        .audioChannels(1)
        .audioFrequency(16000)
        .toFormat("wav")
        .on("end", () => {
          logger.debug("Audio normalization complete");
          resolve(outputPath);
        })
        .on("error", (error: unknown) => {
          logger.error(error, "Error normalizing audio:");
          reject(error);
        })
        .save(outputPath);
    });
  }

  async createMp3DataUri(audio: ArrayBuffer): Promise<string> {
    const inputStream = new Readable();
    inputStream.push(Buffer.from(audio));
    inputStream.push(null);
    return new Promise((resolve, reject) => {
      const chunk: Buffer[] = [];

      ffmpeg()
        .input(inputStream)
        .audioCodec("libmp3lame")
        .audioBitrate(128)
        .audioChannels(2)
        .toFormat("mp3")
        .on("error", (err) => {
          reject(err);
        })
        .pipe()
        .on("data", (data: Buffer) => {
          chunk.push(data);
        })
        .on("end", () => {
          const buffer = Buffer.concat(chunk);
          resolve(`data:audio/mp3;base64,${buffer.toString("base64")}`);
        })
        .on("error", (err) => {
          reject(err);
        });
    });
  }

  async saveToMp3(audio: ArrayBuffer, filePath: string): Promise<string> {
    const inputStream = new Readable();
    inputStream.push(Buffer.from(audio));
    inputStream.push(null);
    return new Promise((resolve, reject) => {
      ffmpeg()
        .input(inputStream)
        .audioCodec("libmp3lame")
        .audioBitrate(128)
        .audioChannels(2)
        .toFormat("mp3")
        .save(filePath)
        .on("end", () => {
          logger.debug("Audio conversion complete");
          resolve(filePath);
        })
        .on("error", (err) => {
          reject(err);
        });
    });
  }

  // Normalizes the final rendered video's audio to a standard loudness
  // target. Remotion's <Audio volume={}> prop is a simple linear multiplier,
  // so if the underlying music files are mastered quietly, even "high" volume
  // settings can end up barely audible. This brings the whole mix (voice +
  // music) up to a consistent, properly audible level regardless of how
  // quiet the source assets are, without touching the video stream at all.
  async normalizeAudioLoudness(videoPath: string): Promise<void> {
    const tempOutputPath = `${videoPath}.normalized.mp4`;
    logger.debug({ videoPath }, "Normalizing final audio loudness");

    await new Promise<void>((resolve, reject) => {
      ffmpeg()
        .input(videoPath)
        .videoCodec("copy")
        .audioFilters("loudnorm=I=-14:TP=-1.5:LRA=11")
        .audioCodec("aac")
        .audioBitrate("192k")
        .outputOptions(["-movflags", "+faststart"])
        .save(tempOutputPath)
        .on("end", () => resolve())
        .on("error", (err: unknown) => reject(err));
    });

    await fs.rename(tempOutputPath, videoPath);
    logger.debug({ videoPath }, "Audio loudness normalization complete");
  }
}
