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
// Cloudflare's free allocation is 10,000 Neurons/day, resetting at 00:00 UTC.
// flux-1-schnell at our target portrait resolution (1080x1920) runs roughly
// 75-80 neurons/image, so even a full day of 4 videos x 10 scenes (~40
// images, ~3,000-3,200 neurons) uses well under a third of the daily
// allowance. No client-side throttle is needed the way Pollinations required
// one — Cloudflare's real production infra doesn't fall over under a burst
// of 10 back-to-back requests the way Pollinations' community-hosted
// endpoint did.
const cloudflareApiBase = "https://api.cloudflare.com/client/v4/accounts";

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

// Two models supported, selectable via constructor param:
// - flux-1-schnell: fast (4 steps), free-tier friendly, but has NO
//   negative_prompt input at all (confirmed against Cloudflare's own model
//   schema) — the only anti-artifact lever available is the positive prompt
//   text itself.
// - stable-diffusion-xl-base-1.0: DOES support negative_prompt, giving a
//   real "avoid this" channel like Pollinations had — at the cost of more
//   steps/neurons and slower generation (default 20 steps vs schnell's 4).
const fluxModel = "@cf/black-forest-labs/flux-1-schnell";
const sdxlModel = "@cf/stabilityai/stable-diffusion-xl-base-1.0";

const negativePrompt =
  "duplicate objects, multiple identical items, extra limbs, distorted fingers, malformed hands, readable text, writing, letters, words, gibberish text, blurry, photorealistic, distorted anatomy";

export class CloudflareAPI {
  private model: string;

  constructor(
    private stylePrompt: string,
    private accountId: string,
    private apiToken: string,
    private tempDirPath: string = "/tmp",
    modelChoice: "flux" | "sdxl" = "flux",
    private fluxSteps: number = 4,
  ) {
    this.model = modelChoice === "sdxl" ? sdxlModel : fluxModel;
  }

  private buildRunUrl(): string {
    return `${cloudflareApiBase}/${this.accountId}/ai/run/${this.model}`;
  }

  private buildPrompt(prompt: string): string {
    // flux-1-schnell has no negative_prompt slot, so these constraints have
    // to work as plain positive instructions. Rewritten as a natural
    // sentence rather than a stacked comma-list of fragments — Flux's own
    // prompting guidance notes it responds better to concise natural
    // language than to "keyword soup," which the previous longer suffix had
    // become. Note: even with ideal prompting, schnell is a distilled model
    // with a documented tendency toward structural errors (extra/missing
    // limbs) — this reduces frequency, it doesn't eliminate the issue.
    return this.stylePrompt
      ? `${this.stylePrompt}. The scene shows ${prompt}. It's a single person with natural two-armed, two-legged anatomy and correct hands, alone in a simple, uncluttered setting with only one of any background object visible. No text or writing anywhere in the image.`
      : prompt;
  }

  private async downloadImage(
    prompt: string,
    width: number,
    height: number,
    destPath: string,
    timeout: number,
  ): Promise<void> {
    const fullPrompt = this.buildPrompt(prompt);
    const seed = Math.floor(Math.random() * 1_000_000);
    const isSdxl = this.model === sdxlModel;

    const body: Record<string, unknown> = {
      prompt: fullPrompt,
      width,
      height,
      seed,
    };
    if (isSdxl) {
      // SDXL has a real negative_prompt channel — the actual fix for
      // duplicate/malformed anatomy, not just stronger wording.
      body.negative_prompt = negativePrompt;
      // Leaving num_steps at its default (20) — SDXL's distilled/lightning
      // variants exist for speed, but base-1.0 is tuned for the default
      // step count; cutting steps here would undercut the quality gain
      // this model is being used for in the first place.
    } else {
      // flux-1-schnell's official range is 1-4 steps. Some community
      // fine-tunes report 4-8 working acceptably, but this is untested for
      // the base BFL schnell hosted here — configurable so it can be A/B'd
      // without a code change, default stays at the documented-safe value.
      body.num_steps = this.fluxSteps;
    }

    const response = await fetch(this.buildRunUrl(), {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.apiToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(timeout),
    });

    if (!response.ok) {
      throw new Error(`Cloudflare Workers AI error: ${response.status} ${response.statusText}`);
    }

    if (isSdxl) {
      // SDXL's REST response is the raw image bytes directly (confirmed via
      // Cloudflare's own docs: "content-type": "image/jpg" on the response),
      // unlike flux-1-schnell's JSON-wrapped base64 — no envelope to parse.
      const arrayBuffer = await response.arrayBuffer();
      await fs.writeFile(destPath, Buffer.from(arrayBuffer));
      return;
    }

    const data = (await response.json()) as {
      success: boolean;
      result?: { image?: string };
      errors?: Array<{ code: number; message: string }>;
    };

    if (!data.success || !data.result?.image) {
      const errMsg = data.errors?.map((e) => `${e.code}: ${e.message}`).join(", ") || "unknown error";
      throw new Error(`Cloudflare Workers AI error: ${errMsg}`);
    }

    // Workers AI returns base64-encoded image bytes in JSON, unlike
    // Pollinations which streams raw bytes directly — decode before writing.
    const buffer = Buffer.from(data.result.image, "base64");
    await fs.writeFile(destPath, buffer);
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
    const scaleW = Math.round(width * 2.5);
    const scaleH = Math.round(height * 2.5);

    const zoomExpr = `'${move.zoomStart}+(${move.zoomEnd}-${move.zoomStart})*on/${totalFrames}'`;
    const xExpr = `'(iw-iw/zoom)*${move.xStart}+((iw-iw/zoom)*${move.xEnd}-(iw-iw/zoom)*${move.xStart})*on/${totalFrames}'`;
    const yExpr = `'(ih-ih/zoom)*${move.yStart}+((ih-ih/zoom)*${move.yEnd}-(ih-ih/zoom)*${move.yStart})*on/${totalFrames}'`;

    await new Promise<void>((resolve, reject) => {
      ffmpeg()
        .input(imagePath)
        .inputOptions(["-loop", "1", "-framerate", String(fps)])
        .complexFilter([
          `scale=${scaleW}:${scaleH}:force_original_aspect_ratio=increase,crop=${scaleW}:${scaleH}`,
          `zoompan=z=${zoomExpr}:x=${xExpr}:y=${yExpr}:d=${totalFrames}:s=${width}x${height}:fps=${fps}`,
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

    logger.debug({ searchTerm }, "Generating image via Cloudflare Workers AI");

    await this.downloadImage(searchTerm, width, height, imagePath, timeout);
    await this.imageToKenBurnsVideo(
      imagePath,
      videoPath,
      minDurationSeconds + 1,
      width,
      height,
    );
    await fs.remove(imagePath).catch(() => {});

    logger.debug({ searchTerm, videoPath }, "Created Ken Burns clip from Cloudflare image");

    return {
      id,
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
    const searchTerm = searchTerms.length > 0 ? searchTerms.join(", ") : "abstract background";
    try {
      return await this._findVideo(searchTerm, minDurationSeconds, orientation, timeout);
    } catch (error: unknown) {
      // Cloudflare surfaces quota/rate issues as 429 (over free daily
      // allocation) or 500/503 (transient backend issues) — treat both
      // tiers the same way the Pollinations provider does.
      const isRateLimit =
        error instanceof Error &&
        (error.message.includes("429") ||
          error.message.includes("500") ||
          error.message.includes("503"));
      if (retryCounter < retryTimes) {
        const jitter = Math.floor(Math.random() * 1000);
        const backoffMs = isRateLimit
          ? 15000 * Math.pow(2, retryCounter) + jitter
          : 5000 * Math.pow(2, retryCounter) + jitter;
        logger.warn(
          { searchTerm, retryCounter, isRateLimit, backoffMs },
          "Cloudflare generation failed, retrying after backoff...",
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
      logger.error(error, "Error generating image/video via Cloudflare Workers AI");
      throw error;
    }
  }
}
