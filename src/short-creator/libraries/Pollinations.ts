/* eslint-disable @remotion/deterministic-randomness */
import fs from "fs-extra";
import path from "path";
import cuid from "cuid";
import ffmpeg from "fluent-ffmpeg";
import { getOrientationConfig } from "../../components/utils";
import { logger } from "../../logger";
import { OrientationEnum, type Video } from "../../types/shorts";

const defaultTimeoutMs = 90000;
const retryTimes = 5;
// Minimum gap enforced between EVERY Pollinations request (successes included),
// since generating 10 scenes back-to-back can trip the API's rate limit even
// when each individual request eventually succeeds. Shared across all
// instances via a static so it holds even across concurrent scene renders.
const minRequestIntervalMs = 6000;

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
  // Straight zoom-in, centered
  { zoomStart: 1.0, zoomEnd: 1.15, xStart: 0.5, xEnd: 0.5, yStart: 0.5, yEnd: 0.45 },
  // Straight zoom-out, centered
  { zoomStart: 1.15, zoomEnd: 1.0, xStart: 0.45, xEnd: 0.55, yStart: 0.5, yEnd: 0.5 },
  // Gentle zoom-in with slight vertical drift
  { zoomStart: 1.0, zoomEnd: 1.12, xStart: 0.4, xEnd: 0.5, yStart: 0.5, yEnd: 0.5 },
  // Gentle zoom-out with slight vertical drift
  { zoomStart: 1.12, zoomEnd: 1.0, xStart: 0.5, xEnd: 0.5, yStart: 0.4, yEnd: 0.5 },
  // Pan left-to-right, no zoom change
  { zoomStart: 1.1, zoomEnd: 1.1, xStart: 0.3, xEnd: 0.7, yStart: 0.5, yEnd: 0.5 },
  // Pan right-to-left, no zoom change
  { zoomStart: 1.1, zoomEnd: 1.1, xStart: 0.7, xEnd: 0.3, yStart: 0.5, yEnd: 0.5 },
  // Diagonal: top-left to bottom-right, zooming in
  { zoomStart: 1.0, zoomEnd: 1.18, xStart: 0.3, xEnd: 0.6, yStart: 0.3, yEnd: 0.6 },
  // Diagonal: bottom-right to top-left, zooming in
  { zoomStart: 1.0, zoomEnd: 1.18, xStart: 0.7, xEnd: 0.4, yStart: 0.7, yEnd: 0.4 },
  // Corner push-in: top-right corner toward center
  { zoomStart: 1.05, zoomEnd: 1.2, xStart: 0.75, xEnd: 0.5, yStart: 0.25, yEnd: 0.45 },
  // Corner push-in: bottom-left corner toward center
  { zoomStart: 1.05, zoomEnd: 1.2, xStart: 0.25, xEnd: 0.5, yStart: 0.75, yEnd: 0.55 },
  // Slow, subtle zoom-in (for calmer/quieter beats)
  { zoomStart: 1.0, zoomEnd: 1.08, xStart: 0.5, xEnd: 0.5, yStart: 0.5, yEnd: 0.5 },
  // Fast, dramatic zoom-in (for punchline/reveal beats)
  { zoomStart: 1.0, zoomEnd: 1.25, xStart: 0.5, xEnd: 0.5, yStart: 0.5, yEnd: 0.4 },
  // Vertical pan upward, mild zoom
  { zoomStart: 1.08, zoomEnd: 1.15, xStart: 0.5, xEnd: 0.5, yStart: 0.7, yEnd: 0.3 },
  // Vertical pan downward, mild zoom
  { zoomStart: 1.08, zoomEnd: 1.15, xStart: 0.5, xEnd: 0.5, yStart: 0.3, yEnd: 0.7 },
];

// Tracks the last move index per-process so consecutive scenes in the same
// video don't repeat the same pan/zoom pattern back to back.
let lastMoveIndex = -1;
function pickKenBurnsMove(): KenBurnsMove {
  let index = Math.floor(Math.random() * kenBurnsMoves.length);
  if (kenBurnsMoves.length > 1) {
    while (index === lastMoveIndex) {
      index = Math.floor(Math.random() * kenBurnsMoves.length);
    }
  }
  lastMoveIndex = index;
  return kenBurnsMoves[index];
}

export class PollinationsAPI {
  // Static so the throttle is shared across every scene/instance in a run,
  // not reset per-scene — that's what actually prevents the burst of 10
  // back-to-back requests from tripping the rate limit.
  private static lastRequestTime = 0;
  private static throttleChain: Promise<void> = Promise.resolve();

  constructor(
    private stylePrompt: string,
    private apiKey?: string,
    private tempDirPath: string = "/tmp",
  ) {}

  // Queues callers so requests are issued strictly one at a time, each
  // waiting out the remaining gap since the previous request actually fired.
  private async throttle(): Promise<void> {
    const runThrottle = async () => {
      const elapsed = Date.now() - PollinationsAPI.lastRequestTime;
      const waitTime = minRequestIntervalMs - elapsed;
      if (waitTime > 0) {
        await new Promise((resolve) => setTimeout(resolve, waitTime));
      }
      PollinationsAPI.lastRequestTime = Date.now();
    };
    const next = PollinationsAPI.throttleChain.then(runThrottle);
    PollinationsAPI.throttleChain = next.catch(() => {});
    return next;
  }

  private buildImageUrl(prompt: string, width: number, height: number): string {
    // Style descriptors go first — diffusion models generally weight earlier
    // tokens more heavily, so leading with style keeps every scene visually
    // consistent instead of drifting toward semi-realistic renders.
    const fullPrompt = this.stylePrompt
      ? `${this.stylePrompt}, ${prompt}, single clean composition, exactly one of each object in the scene, simple relaxed hands not in extreme close-up`
      : prompt;
    // Dedicated negative_prompt field — more reliable than cramming
    // exclusions into the main prompt text, since it's a field the model is
    // specifically trained to treat as "avoid this." Note: this measurably
    // helps hands and garbled in-image text, but does NOT reliably prevent
    // duplicate background objects (e.g. two clocks) — that remains a
    // probabilistic tendency of the free model regardless of prompting.
    const negativePrompt =
      "duplicate objects, multiple identical items, extra limbs, distorted fingers, malformed hands, readable text, writing, letters, words, gibberish text, blurry, photorealistic, distorted anatomy";
    // image.pollinations.ai is Pollinations' longstanding no-signup, no-API-key
    // image endpoint. An optional key can be supplied for higher rate limits.
    const seed = Math.floor(Math.random() * 1_000_000);
    const params = new URLSearchParams({
      model: "flux", // explicit — Pollinations' default model has changed over
      // time and newer defaults render more photorealistic / less consistent
      // results. Flux is free, unlimited, and follows style prompts reliably.
      width: String(width),
      height: String(height),
      nologo: "true",
      seed: String(seed),
      // NOTE: deliberately NOT setting enhance=true. Confirmed via direct
      // testing that it causes Pollinations to return an immediate 500
      // (not a timeout/overload — fails in <1s), independent of prompt
      // content or rate limiting. Our own prompts are already detailed
      // enough that the extra server-side rewrite isn't needed.
      negative_prompt: negativePrompt,
    });
    if (this.apiKey) {
      params.set("key", this.apiKey);
    }
    return `https://image.pollinations.ai/prompt/${encodeURIComponent(fullPrompt)}?${params.toString()}`;
  }

  private async downloadImage(url: string, destPath: string, timeout: number): Promise<void> {
    await this.throttle();
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
    const move = pickKenBurnsMove();
    const fps = 25;
    const totalFrames = Math.ceil(durationSeconds * fps);
    // Oversample generously so zoompan has real pixels to pan across without
    // revealing edges or looking blocky as it zooms in.
    const scaleW = Math.round(width * 2.5);
    const scaleH = Math.round(height * 2.5);

    const zoomExpr = `'${move.zoomStart}+(${move.zoomEnd}-${move.zoomStart})*on/${totalFrames}'`;
    const xExpr = `'(iw-iw/zoom)*${move.xStart}+((iw-iw/zoom)*${move.xEnd}-(iw-iw/zoom)*${move.xStart})*on/${totalFrames}'`;
    const yExpr = `'(ih-ih/zoom)*${move.yStart}+((ih-ih/zoom)*${move.yEnd}-(ih-ih/zoom)*${move.yStart})*on/${totalFrames}'`;

    await new Promise<void>((resolve, reject) => {
      ffmpeg()
        .input(imagePath)
        // -framerate is required alongside -loop 1: without it, ffmpeg only
        // feeds zoompan a single still frame instead of a continuous stream,
        // so the pan/zoom never actually animates (looks like a frozen image).
        .inputOptions(["-loop", "1", "-framerate", String(fps)])
        .complexFilter([
          `scale=${scaleW}:${scaleH}:force_original_aspect_ratio=increase,crop=${scaleW}:${scaleH}`,
          `zoompan=z=${zoomExpr}:x=${xExpr}:y=${yExpr}:d=${totalFrames}:s=${width}x${height}:fps=${fps}`,
          // Re-normalizes timing after zoompan, which otherwise can produce
          // slightly uneven frame pacing when fed from a looped still image.
          `framerate=fps=${fps}`,
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
    // Join all terms into one descriptive prompt — an AI image model needs a
    // coherent scene description, not an isolated keyword the way a stock
    // footage search does.
    const searchTerm = searchTerms.length > 0 ? searchTerms.join(", ") : "abstract background";
    try {
      return await this._findVideo(searchTerm, minDurationSeconds, orientation, timeout);
    } catch (error: unknown) {
      // Pollinations' free tier often returns a bare 500 instead of a
      // proper 429 when it's overloaded/rate-limiting us — so a plain
      // "Internal Server Error" gets treated the same as a rate limit
      // for backoff purposes, not retried on a short timer.
      const isRateLimit =
        error instanceof Error &&
        (error.message.includes("429") || error.message.includes("500"));
      if (retryCounter < retryTimes) {
        // Real breathing room instead of hammering the API immediately,
        // plus jitter so concurrent scene requests don't retry in lockstep.
        const jitter = Math.floor(Math.random() * 1000);
        const backoffMs = isRateLimit
          ? 15000 * Math.pow(2, retryCounter) + jitter // 15s, 30s, 60s, 120s, 240s
          : 5000 * Math.pow(2, retryCounter) + jitter; // 5s, 10s, 20s, 40s, 80s
        logger.warn(
          { searchTerm, retryCounter, isRateLimit, backoffMs },
          "Pollinations generation failed, retrying after backoff...",
        );
        await new Promise((resolve) => setTimeout(resolve, backoffMs));
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
